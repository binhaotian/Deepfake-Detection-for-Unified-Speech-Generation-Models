#!/usr/bin/env python3
"""Run resumable batched AuK inference for the G2 restoration pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio

from auk.infer.infer_auk import AukInfer, save_audio


TASKS = {
    "dereverb": {
        "input": Path("dereverb/dereverb_input.wav"),
        "output": Path("dereverb/auk_dereverb.wav"),
        "instruction": "Remove only the room reverberation, preserve everything else, and output audio of the same length.",
        "seed_offset": 3_000_000,
    },
    "full_enhancement": {
        "input": Path("full_enhancement/full_enhance_input.wav"),
        "output": Path("full_enhancement/auk_full_enhance.wav"),
        "instruction": "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.",
        "seed_offset": 4_000_000,
    },
    "channel_restoration": {
        "input": Path("channel_restoration/channel_input.wav"),
        "output": Path("channel_restoration/auk_channel_restore.wav"),
        "instruction": "Repair the telephone effect and restore natural, clear speech.",
        "seed_offset": 5_000_000,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def messages(instruction: str, audio: Path) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "audio", "audio": str(audio)},
    ]}]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def validate(root: Path, rows: list[dict[str, str]]) -> dict:
    errors = []
    warnings = []
    metrics = []
    for row in rows:
        pair_root = root / row["pair_id"]
        for task, spec in TASKS.items():
            input_path = pair_root / spec["input"]
            output_path = pair_root / spec["output"]
            if not output_path.exists():
                errors.append(f"{row['pair_id']} {task}: missing output")
                continue
            output, sr = torchaudio.load(str(output_path))
            input_seconds = duration(input_path)
            output_seconds = output.shape[-1] / sr
            finite = bool(torch.isfinite(output).all())
            peak = float(output.abs().max())
            clipped_fraction = float((output.abs() >= 0.999).float().mean())
            if sr != 24_000 or output.shape[0] != 1:
                errors.append(f"{row['pair_id']} {task}: format={tuple(output.shape)}/{sr}")
            if not finite:
                errors.append(f"{row['pair_id']} {task}: NaN/Inf")
            if abs(output_seconds - input_seconds) > 0.06:
                errors.append(
                    f"{row['pair_id']} {task}: output={output_seconds:.3f}s input={input_seconds:.3f}s"
                )
            if clipped_fraction > 0.001:
                warnings.append(
                    f"{row['pair_id']} {task}: clipped_fraction={clipped_fraction:.6f}"
                )
            metrics.append({
                "pair_id": row["pair_id"],
                "task": task,
                "seconds": output_seconds,
                "peak": peak,
                "clipped_fraction": clipped_fraction,
                "sha256": sha256(output_path),
            })
    return {"errors": errors, "warnings": warnings, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    rows = list(csv.DictReader((args.root / "manifest.tsv").open(), delimiter="\t"))
    if not rows:
        raise RuntimeError("empty pilot manifest")
    config = {
        "created_at_utc": started_at.isoformat(),
        "root": str(args.root.resolve()),
        "model": "AuK Base",
        "config": str(Path("ckpts/AuK/config.yaml").resolve()),
        "checkpoint": str(Path("ckpts/AuK/auk_base.safetensors").resolve()),
        "checkpoint_sha256": sha256(Path("ckpts/AuK/auk_base.safetensors")),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "nfe": args.nfe,
        "cfg": args.cfg,
        "sway": args.sway,
        "task_order": list(TASKS),
        "tasks": {task: {
            "input": str(spec["input"]),
            "output": str(spec["output"]),
            "instruction": spec["instruction"],
            "seed_offset": spec["seed_offset"],
        } for task, spec in TASKS.items()},
        "resume_policy": "skip existing output unless --overwrite",
    }
    (args.root / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    load_started = time.perf_counter()
    engine = AukInfer(
        "ckpts/AuK/config.yaml",
        "ckpts/AuK/auk_base.safetensors",
        dtype=args.dtype,
    )
    sync_cuda()
    model_load_seconds = time.perf_counter() - load_started

    progress_path = args.root / "generation_progress.jsonl"
    task_reports = {}
    generated_total = 0
    skipped_total = 0
    for task, spec in TASKS.items():
        work = []
        for row in rows:
            pair_root = args.root / row["pair_id"]
            input_path = pair_root / spec["input"]
            output_path = pair_root / spec["output"]
            if output_path.exists() and not args.overwrite:
                skipped_total += 1
                continue
            seed = int(row["pilot_seed"]) + int(spec["seed_offset"])
            work.append({
                "pair_id": row["pair_id"],
                "input": input_path,
                "output": output_path,
                "seconds": duration(input_path),
                "seed": seed,
            })
        work.sort(key=lambda item: item["seconds"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        task_started = time.perf_counter()
        task_generated = 0
        batch_count = math.ceil(len(work) / args.batch_size) if work else 0
        for batch_index, begin in enumerate(range(0, len(work), args.batch_size), 1):
            batch = work[begin: begin + args.batch_size]
            outputs, sample_rate = engine.generate_batch(
                [messages(spec["instruction"], item["input"]) for item in batch],
                audios=[str(item["input"]) for item in batch],
                gen_seconds=[None] * len(batch),
                nfe=args.nfe,
                cfg_strength=args.cfg,
                sway_sampling_coef=args.sway,
                seed=[item["seed"] for item in batch],
            )
            for item, output in zip(batch, outputs):
                save_audio(output, sample_rate, str(item["output"]))
                task_generated += 1
                generated_total += 1
            sync_cuda()
            event = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "batch": batch_index,
                "batch_count": batch_count,
                "pair_ids": [item["pair_id"] for item in batch],
                "generated_total": generated_total,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(
                f"[{task}] batch {batch_index}/{batch_count} "
                f"pairs={','.join(event['pair_ids'])} total={generated_total}",
                flush=True,
            )
        sync_cuda()
        elapsed = time.perf_counter() - task_started
        task_reports[task] = {
            "generated": task_generated,
            "wall_seconds": elapsed,
            "seconds_per_output": elapsed / task_generated if task_generated else 0.0,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0,
        }

    validation = validate(args.root, rows)
    finished_at = datetime.now(timezone.utc)
    report = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "total_wall_seconds": (finished_at - started_at).total_seconds(),
        "model_load_seconds": model_load_seconds,
        "generated_outputs": generated_total,
        "skipped_existing_outputs": skipped_total,
        "task_reports": task_reports,
        "validation_error_count": len(validation["errors"]),
        "validation_warning_count": len(validation["warnings"]),
        "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"],
        "output_metrics": validation["metrics"],
    }
    (args.root / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "output_metrics"}, indent=2))
    if validation["errors"]:
        raise SystemExit("generation validation failed")


if __name__ == "__main__":
    main()
