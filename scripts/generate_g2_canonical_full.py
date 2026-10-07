#!/usr/bin/env python3
"""Resumable batched AuK Base inference for full canonical G2 restoration."""
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
    "denoise_10db": {
        "input": Path("denoise_10db/input.wav"),
        "output": Path("denoise_10db/auk_output.wav"),
        "instruction": (
            "Please remove only the background noise from this audio while preserving "
            "the original room reverberation and any colorations. Output a denoised "
            "speech of the same length as the input."
        ),
        "seed_offset": 8_000_000,
    },
    "dereverberation": {
        "input": Path("dereverberation/input.wav"),
        "output": Path("dereverberation/auk_output.wav"),
        "instruction": (
            "Please remove only the room reverberation from this audio while preserving "
            "the original background noise and other colorations. Output a "
            "dereverberated speech of the same length as the input."
        ),
        "seed_offset": 9_000_000,
    },
    "full_enhancement_10db": {
        "input": Path("full_enhancement_10db/input.wav"),
        "output": Path("full_enhancement_10db/auk_output.wav"),
        "instruction": (
            "Preserve all speakers, remove noise and reverberation, and output clean "
            "speech of the same length."
        ),
        "seed_offset": 6_000_000,
    },
    "bandwidth_extension": {
        "input": Path("bandwidth_extension/input.wav"),
        "output": Path("bandwidth_extension/auk_output.wav"),
        "instruction": (
            "This audio suffers from limited bandwidth. Please restore it to a "
            "wideband, clear-sounding speech."
        ),
        "seed_offset": 10_000_000,
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


def output_counts(root: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        task: sum((root / row["pair_id"] / spec["output"]).exists() for row in rows)
        for task, spec in TASKS.items()
    }


def validate(root: Path, rows: list[dict[str, str]]) -> dict:
    errors = []
    warnings = []
    metrics = []
    for row_index, row in enumerate(rows, start=1):
        pair_root = root / row["pair_id"]
        for task, spec in TASKS.items():
            input_path = pair_root / spec["input"]
            output_path = pair_root / spec["output"]
            if not output_path.exists():
                errors.append(f"{row['pair_id']} {task}: missing output")
                continue
            output, sample_rate = torchaudio.load(str(output_path))
            input_seconds = duration(input_path)
            output_seconds = output.shape[-1] / sample_rate
            finite = bool(torch.isfinite(output).all())
            peak = float(output.abs().max())
            clipped_fraction = float((output.abs() >= 0.999).float().mean())
            if sample_rate != 24_000 or output.shape[0] != 1:
                errors.append(f"{row['pair_id']} {task}: format={tuple(output.shape)}/{sample_rate}")
            if not finite:
                errors.append(f"{row['pair_id']} {task}: NaN/Inf")
            if abs(output_seconds - input_seconds) > 0.06:
                errors.append(
                    f"{row['pair_id']} {task}: output={output_seconds:.3f}s "
                    f"input={input_seconds:.3f}s"
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
        if row_index % 250 == 0 or row_index == len(rows):
            print(f"validated {row_index}/{len(rows)} pairs", flush=True)
    return {"errors": errors, "warnings": warnings, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    rows = list(csv.DictReader((args.root / "manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise RuntimeError("empty manifest")

    config = {
        "created_at_utc": started_at.isoformat(),
        "run_id": run_id,
        "root": str(args.root.resolve()),
        "pair_count": len(rows),
        "model": "AuK Base",
        "config": str(Path("ckpts/AuK/config.yaml").resolve()),
        "checkpoint": str(Path("ckpts/AuK/auk_base.safetensors").resolve()),
        "checkpoint_sha256": sha256(Path("ckpts/AuK/auk_base.safetensors")),
        "requested_batch_size": args.batch_size,
        "dtype": args.dtype,
        "nfe": args.nfe,
        "cfg": args.cfg,
        "sway": args.sway,
        "limit_per_task": args.limit_per_task,
        "task_order": list(TASKS),
        "tasks": TASKS,
        "resume_policy": "skip existing output unless --overwrite",
        "oom_policy": "halve batch size and retry the same items",
    }
    config_name = "generation_smoke_config.json" if args.limit_per_task else "generation_config.json"
    (args.root / config_name).write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

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
            work.append({
                "pair_id": row["pair_id"],
                "input": input_path,
                "output": output_path,
                "seconds": duration(input_path),
                "seed": int(row["g2_seed"]) + int(spec["seed_offset"]),
            })
        work.sort(key=lambda item: item["seconds"])
        if args.limit_per_task:
            work = work[: args.limit_per_task]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        task_started = time.perf_counter()
        task_generated = 0
        position = 0
        effective_batch_size = args.batch_size
        batch_index = 0
        while position < len(work):
            batch = work[position: position + effective_batch_size]
            try:
                outputs, sample_rate = engine.generate_batch(
                    [messages(spec["instruction"], item["input"]) for item in batch],
                    audios=[str(item["input"]) for item in batch],
                    gen_seconds=[None] * len(batch),
                    nfe=args.nfe,
                    cfg_strength=args.cfg,
                    sway_sampling_coef=args.sway,
                    seed=[item["seed"] for item in batch],
                )
            except torch.OutOfMemoryError:
                if effective_batch_size <= 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                torch.cuda.empty_cache()
                event = {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "task": task,
                    "event": "oom_reduce_batch",
                    "new_batch_size": effective_batch_size,
                    "position": position,
                }
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                print(f"[{task}] OOM; retrying with batch={effective_batch_size}", flush=True)
                continue

            for item, output in zip(batch, outputs):
                save_audio(output, sample_rate, str(item["output"]))
                task_generated += 1
                generated_total += 1
            sync_cuda()
            position += len(batch)
            batch_index += 1
            elapsed = time.perf_counter() - task_started
            event = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "task": task,
                "batch": batch_index,
                "batch_size": len(batch),
                "effective_batch_size": effective_batch_size,
                "task_generated": task_generated,
                "task_total": len(work),
                "generated_total": generated_total,
                "elapsed_seconds": elapsed,
                "seconds_per_output": elapsed / task_generated,
                "pair_ids": [item["pair_id"] for item in batch],
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            if batch_index % args.log_every == 0 or position == len(work):
                eta = (elapsed / task_generated) * (len(work) - task_generated) if task_generated else 0.0
                print(
                    f"[{task}] {task_generated}/{len(work)} batch={effective_batch_size} "
                    f"sec/output={elapsed / task_generated:.3f} eta_min={eta / 60:.1f} "
                    f"total={generated_total}",
                    flush=True,
                )

        sync_cuda()
        elapsed = time.perf_counter() - task_started
        task_reports[task] = {
            "generated": task_generated,
            "candidate_outputs": len(work),
            "wall_seconds": elapsed,
            "seconds_per_output": elapsed / task_generated if task_generated else 0.0,
            "final_batch_size": effective_batch_size,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0,
        }

    counts = output_counts(args.root, rows)
    full_complete = all(count == len(rows) for count in counts.values())
    validation = {"errors": [], "warnings": [], "metrics": []}
    if full_complete:
        validation = validate(args.root, rows)

    finished_at = datetime.now(timezone.utc)
    report = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "run_id": run_id,
        "total_wall_seconds": (finished_at - started_at).total_seconds(),
        "model_load_seconds": model_load_seconds,
        "generated_outputs": generated_total,
        "skipped_existing_outputs": skipped_total,
        "output_counts": counts,
        "full_complete": full_complete,
        "task_reports": task_reports,
        "validation_error_count": len(validation["errors"]),
        "validation_warning_count": len(validation["warnings"]),
        "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"],
        "output_metrics": validation["metrics"],
    }
    report_name = "generation_smoke_report.json" if args.limit_per_task else "generation_report.json"
    (args.root / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "output_metrics"}, indent=2))
    if validation["errors"]:
        raise SystemExit("generation validation failed")


if __name__ == "__main__":
    main()
