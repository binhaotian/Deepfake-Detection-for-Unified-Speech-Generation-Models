#!/usr/bin/env python3
"""Generate AuK matched-set outputs with batch inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio

from auk.infer.infer_auk import AukInfer, save_audio


INPUT_FILES = (
    "anchor.wav",
    "tts_reference.wav",
    "se_noise.wav",
    "se_input.wav",
    "tse_interferer.wav",
    "tse_input.wav",
)

TASKS = {
    "tts": {
        "input": "tts_reference.wav",
        "output": "auk_tts.wav",
        "seed_offset": 0,
    },
    "se": {
        "input": "se_input.wav",
        "output": "auk_se.wav",
        "seed_offset": 1_000_000,
    },
    "tse": {
        "input": "tse_input.wav",
        "output": "auk_tse.wav",
        "seed_offset": 2_000_000,
    },
}


def audio_duration(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def make_messages(instruction: str, audio: Path) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "audio", "audio": str(audio)},
    ]}]


def task_instruction(task: str, text: str) -> str:
    if task == "tts":
        return f"Say the following with the same voice: '{text}'"
    if task == "se":
        return "Remove only the background noise, preserve everything else, and output audio of the same length."
    if task == "tse":
        return f'Keep only the speaker who says "{text}" and remove all other speakers.'
    raise ValueError(task)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_outputs(root: Path, rows: list[dict[str, str]]) -> dict:
    errors = []
    warnings = []
    metrics = []
    expected_by_task = {
        "auk_tts.wav": "anchor.wav",
        "auk_se.wav": "se_input.wav",
        "auk_tse.wav": "tse_input.wav",
    }
    for row in rows:
        pair = root / row["pair_id"]
        for output_name, expected_name in expected_by_task.items():
            output_path = pair / output_name
            if not output_path.exists():
                errors.append(f"{row['pair_id']}: missing {output_name}")
                continue
            audio, sample_rate = torchaudio.load(str(output_path))
            expected_seconds = audio_duration(pair / expected_name)
            output_seconds = audio.shape[-1] / sample_rate
            finite = bool(torch.isfinite(audio).all())
            peak = float(audio.abs().max())
            clipped_fraction = float((audio.abs() >= 0.999).float().mean())
            if sample_rate != 24_000 or audio.shape[0] != 1:
                errors.append(f"{row['pair_id']}: {output_name} format={tuple(audio.shape)}/{sample_rate}")
            if not finite:
                errors.append(f"{row['pair_id']}: {output_name} contains NaN/Inf")
            if abs(output_seconds - expected_seconds) > 0.06:
                errors.append(
                    f"{row['pair_id']}: {output_name} duration={output_seconds:.3f}, expected={expected_seconds:.3f}"
                )
            if clipped_fraction > 0.001:
                warnings.append(
                    f"{row['pair_id']}: {output_name} clipped_fraction={clipped_fraction:.6f}"
                )
            metrics.append({
                "pair_id": row["pair_id"],
                "file": output_name,
                "seconds": output_seconds,
                "peak": peak,
                "clipped_fraction": clipped_fraction,
                "sha256": sha256(output_path),
            })
    return {"errors": errors, "warnings": warnings, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/matched_v0_snr0"))
    parser.add_argument("--out", type=Path, default=Path("data/matched_full_100_snr0_batch4"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    rows = list(csv.DictReader((args.source / "manifest.tsv").open(), delimiter="\t"))
    if not rows:
        raise RuntimeError("source manifest contains no pairs")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in rows:
        source_pair = args.source / row["pair_id"]
        output_pair = args.out / row["pair_id"]
        output_pair.mkdir(parents=True, exist_ok=True)
        for name in INPUT_FILES:
            source_file = source_pair / name
            if not source_file.exists():
                raise FileNotFoundError(source_file)
            destination_file = output_pair / name
            if source_file.resolve() != destination_file.resolve():
                shutil.copy2(source_file, destination_file)
        updated = dict(row)
        updated["pair_dir"] = row["pair_id"]
        updated["tts_seed"] = str(int(row["seed"]) + TASKS["tts"]["seed_offset"])
        updated["se_seed"] = str(int(row["seed"]) + TASKS["se"]["seed_offset"])
        updated["tse_seed"] = str(int(row["seed"]) + TASKS["tse"]["seed_offset"])
        manifest_rows.append(updated)

    manifest_fields = list(manifest_rows[0])
    with (args.out / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    config = {
        "dataset_name": f"AuK matched set {len(rows)} pairs, SNR 0 dB, SIR 0 dB",
        "created_at_utc": started_at.isoformat(),
        "source_root": str(args.source.resolve()),
        "output_root": str(args.out.resolve()),
        "model": "AuK Base",
        "checkpoint": "ckpts/AuK/auk_base.safetensors",
        "checkpoint_bytes": Path("ckpts/AuK/auk_base.safetensors").stat().st_size,
        "sample_rate": 24_000,
        "dtype": args.dtype,
        "nfe": args.nfe,
        "cfg": args.cfg,
        "sway": args.sway,
        "batch_size": args.batch_size,
        "snr_db": 0.0,
        "sir_db": 0.0,
        "task_order": list(TASKS),
        "duration_bucketing": True,
        "seed_policy": "manifest seed + task-specific fixed offset",
        "task_seed_offsets": {task: spec["seed_offset"] for task, spec in TASKS.items()},
        "instructions": {
            "tts": "Say the following with the same voice: '{anchor transcript}'",
            "se": task_instruction("se", ""),
            "tse": "Keep only the speaker who says \"{anchor transcript}\" and remove all other speakers.",
        },
    }
    (args.out / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    load_started = time.perf_counter()
    engine = AukInfer(
        "ckpts/AuK/config.yaml",
        "ckpts/AuK/auk_base.safetensors",
        dtype=args.dtype,
    )
    sync_cuda()
    model_load_seconds = time.perf_counter() - load_started

    task_reports = {}
    total_generated = 0
    progress_path = args.out / "generation_progress.jsonl"
    for task, spec in TASKS.items():
        work = []
        for row in manifest_rows:
            pair = args.out / row["pair_id"]
            output_path = pair / spec["output"]
            if output_path.exists() and not args.overwrite:
                continue
            input_path = pair / spec["input"]
            target_seconds = audio_duration(pair / "anchor.wav") if task == "tts" else None
            work.append({
                "row": row,
                "input": input_path,
                "output": output_path,
                "target_seconds": target_seconds,
                "input_seconds": audio_duration(input_path),
                "seed": int(row[f"{task}_seed"]),
            })
        work.sort(key=lambda item: item["input_seconds"])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        task_started = time.perf_counter()
        task_generated = 0
        batch_count = math.ceil(len(work) / args.batch_size) if work else 0
        for batch_index, begin in enumerate(range(0, len(work), args.batch_size), 1):
            batch = work[begin : begin + args.batch_size]
            requests = [
                make_messages(task_instruction(task, item["row"]["text"]), item["input"])
                for item in batch
            ]
            outputs, sample_rate = engine.generate_batch(
                requests,
                audios=[str(item["input"]) for item in batch],
                gen_seconds=[item["target_seconds"] for item in batch],
                nfe=args.nfe,
                cfg_strength=args.cfg,
                sway_sampling_coef=args.sway,
                seed=[item["seed"] for item in batch],
            )
            for item, audio in zip(batch, outputs):
                save_audio(audio, sample_rate, str(item["output"]))
                task_generated += 1
                total_generated += 1
            sync_cuda()
            event = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "batch": batch_index,
                "batch_count": batch_count,
                "pair_ids": [item["row"]["pair_id"] for item in batch],
                "generated_total": total_generated,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(
                f"[{task}] batch {batch_index}/{batch_count}: "
                f"{', '.join(event['pair_ids'])} | total={total_generated}",
                flush=True,
            )
        sync_cuda()
        task_elapsed = time.perf_counter() - task_started
        task_reports[task] = {
            "generated": task_generated,
            "wall_seconds": task_elapsed,
            "seconds_per_output": task_elapsed / task_generated if task_generated else 0.0,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0,
        }

    validation = validate_outputs(args.out, manifest_rows)
    finished_at = datetime.now(timezone.utc)
    report = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "total_wall_seconds": (finished_at - started_at).total_seconds(),
        "model_load_seconds": model_load_seconds,
        "generated_outputs": total_generated,
        "task_reports": task_reports,
        "validation_error_count": len(validation["errors"]),
        "validation_warning_count": len(validation["warnings"]),
        "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"],
        "output_metrics": validation["metrics"],
    }
    (args.out / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if validation["errors"]:
        raise SystemExit("Generation completed but validation failed; see generation_report.json")
    print(json.dumps({key: report[key] for key in report if key != "output_metrics"}, indent=2))


if __name__ == "__main__":
    main()
