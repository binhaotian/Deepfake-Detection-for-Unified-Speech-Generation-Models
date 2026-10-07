#!/usr/bin/env python3
"""Generate the 10-pair canonical G2 audio-only listening pilot."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio

from auk.infer.infer_auk import AukInfer, save_audio


SR = 24_000
DENOISE_PROMPT = (
    "Please remove only the background noise from this audio while preserving "
    "the original room reverberation and any colorations. Output a denoised "
    "speech of the same length as the input."
)
DEREVERB_PROMPT = (
    "Please remove only the room reverberation from this audio while preserving "
    "the original background noise and other colorations. Output a "
    "dereverberated speech of the same length as the input."
)
BANDWIDTH_PROMPT = (
    "This audio suffers from limited bandwidth. Please restore it to a "
    "wideband, clear-sounding speech."
)


def load(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    audio = audio.mean(0, keepdim=True).to(torch.float32)
    if sample_rate != SR:
        audio = torchaudio.functional.resample(audio, sample_rate, SR)
    return audio


def fit(audio: torch.Tensor, samples: int) -> torch.Tensor:
    if audio.shape[-1] >= samples:
        return audio[..., :samples]
    return torch.nn.functional.pad(audio, (0, samples - audio.shape[-1]))


def rms(audio: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(audio.double() ** 2) + 1e-18))


def duration(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def messages(prompt: str, audio: Path) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "audio", "audio": str(audio)},
    ]}]


def validate_wav(path: Path, reference: Path | None = None) -> dict:
    audio, sample_rate = torchaudio.load(str(path))
    finite = bool(torch.isfinite(audio).all())
    seconds = audio.shape[-1] / sample_rate
    duration_error = abs(seconds - duration(reference)) if reference else 0.0
    return {
        "path": str(path),
        "sample_rate": sample_rate,
        "channels": audio.shape[0],
        "duration_seconds": seconds,
        "duration_error_seconds": duration_error,
        "finite": finite,
        "peak": float(audio.abs().max()),
        "clipped_fraction": float((audio.abs() >= 0.999).float().mean()),
        "valid": sample_rate == SR and audio.shape[0] == 1 and finite and duration_error <= 0.06,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--listening-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    rows = list(csv.DictReader(args.guide.open(encoding="utf-8"), delimiter="\t"))
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 guide rows, found {len(rows)}")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, row in enumerate(rows, start=1):
        pair_id = row["pair_id"]
        pilot_pair = args.pilot_root / pair_id
        revision_pair = args.revision_root / pair_id
        source_pair = args.source_root / pair_id
        pair_out = args.out / pair_id
        pair_out.mkdir(parents=True, exist_ok=True)

        anchor_path = pair_out / "anchor.wav"
        if not anchor_path.exists() or args.overwrite:
            shutil.copy2(pilot_pair / "anchor.wav", anchor_path)

        denoise_dir = pair_out / "denoise_10db"
        dereverb_dir = pair_out / "dereverberation"
        full_dir = pair_out / "full_enhancement_10db"
        bandwidth_dir = pair_out / "bandwidth_extension"
        for directory in (denoise_dir, dereverb_dir, full_dir, bandwidth_dir):
            directory.mkdir(exist_ok=True)

        anchor = load(anchor_path)
        noise = fit(load(source_pair / "se_noise.wav"), anchor.shape[-1])
        noise *= rms(anchor) / (10.0 ** (10.0 / 20.0) * max(rms(noise), 1e-12))
        denoise_input = anchor + noise
        peak = float(denoise_input.abs().max())
        shared_gain = min(1.0, 0.98 / max(peak, 1e-12))
        denoise_speech = anchor * shared_gain
        denoise_noise = noise * shared_gain
        denoise_input *= shared_gain
        torchaudio.save(str(denoise_dir / "speech.wav"), denoise_speech, SR)
        torchaudio.save(str(denoise_dir / "noise.wav"), denoise_noise, SR)
        torchaudio.save(str(denoise_dir / "input.wav"), denoise_input, SR)
        measured_snr = 20.0 * math.log10(rms(denoise_speech) / max(rms(denoise_noise), 1e-12))

        copy_map = {
            dereverb_dir / "input.wav": pilot_pair / "dereverb/dereverb_input.wav",
            full_dir / "input.wav": revision_pair / "full_enhancement_snr10/full_snr10_input.wav",
            full_dir / "auk_output.wav": revision_pair / "full_enhancement_snr10/auk_full_snr10.wav",
            bandwidth_dir / "input.wav": pilot_pair / "channel_restoration/channel_input.wav",
        }
        for target, source in copy_map.items():
            if not target.exists() or args.overwrite:
                shutil.copy2(source, target)

        manifest.append({
            "sample_index": index,
            "pair_id": pair_id,
            "speaker": row["speaker"],
            "text": row["text"],
            "seed": int(row.get("seed") or 20261007 + index - 1),
            "noise_id": row["noise_id"],
            "denoise_snr_db": measured_snr,
            "estimated_rt60_seconds": float(row["estimated_rt60_seconds"]),
        })

    with (args.out / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)

    tasks = {
        "denoise_10db": {
            "input": Path("denoise_10db/input.wav"),
            "output": Path("denoise_10db/auk_output.wav"),
            "prompt": DENOISE_PROMPT,
            "seed_offset": 8_000_000,
        },
        "dereverberation": {
            "input": Path("dereverberation/input.wav"),
            "output": Path("dereverberation/auk_output.wav"),
            "prompt": DEREVERB_PROMPT,
            "seed_offset": 9_000_000,
        },
        "bandwidth_extension": {
            "input": Path("bandwidth_extension/input.wav"),
            "output": Path("bandwidth_extension/auk_output.wav"),
            "prompt": BANDWIDTH_PROMPT,
            "seed_offset": 10_000_000,
        },
    }
    config = {
        "created_at_utc": started.isoformat(),
        "pair_count": len(manifest),
        "model": "AuK Base",
        "checkpoint": str(Path("ckpts/AuK/auk_base.safetensors").resolve()),
        "dtype": "bf16",
        "nfe": 32,
        "cfg": 2.0,
        "sway": -1.0,
        "batch_size": args.batch_size,
        "prompts": {task: spec["prompt"] for task, spec in tasks.items()},
        "full_enhancement_source": str(args.revision_root.resolve()),
        "resume_policy": "skip existing outputs unless --overwrite",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    engine = AukInfer("ckpts/AuK/config.yaml", "ckpts/AuK/auk_base.safetensors", dtype="bf16")
    progress_path = args.out / "progress.jsonl"
    generated_total = 0
    task_reports = {}
    for task, spec in tasks.items():
        work = []
        for row in manifest:
            pair_out = args.out / row["pair_id"]
            output = pair_out / spec["output"]
            if output.exists() and not args.overwrite:
                continue
            input_path = pair_out / spec["input"]
            work.append({
                "pair_id": row["pair_id"],
                "input": input_path,
                "output": output,
                "seconds": duration(input_path),
                "seed": row["seed"] + spec["seed_offset"],
            })
        work.sort(key=lambda item: item["seconds"])
        task_started = time.perf_counter()
        task_count = 0
        for begin in range(0, len(work), args.batch_size):
            batch = work[begin:begin + args.batch_size]
            outputs, sample_rate = engine.generate_batch(
                [messages(spec["prompt"], item["input"]) for item in batch],
                audios=[str(item["input"]) for item in batch],
                gen_seconds=[None] * len(batch),
                nfe=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                seed=[item["seed"] for item in batch],
            )
            for item, output in zip(batch, outputs):
                save_audio(output, sample_rate, str(item["output"]))
                task_count += 1
                generated_total += 1
            event = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "pair_ids": [item["pair_id"] for item in batch],
                "generated_total": generated_total,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(f"[{task}] {task_count}/{len(work)} total={generated_total}", flush=True)
        elapsed = time.perf_counter() - task_started
        task_reports[task] = {
            "generated": task_count,
            "wall_seconds": elapsed,
            "seconds_per_output": elapsed / task_count if task_count else 0.0,
        }

    validation = []
    errors = []
    for row in manifest:
        pair_out = args.out / row["pair_id"]
        specs = [
            ("denoise_10db", pair_out / "denoise_10db/auk_output.wav", pair_out / "denoise_10db/input.wav"),
            ("dereverberation", pair_out / "dereverberation/auk_output.wav", pair_out / "dereverberation/input.wav"),
            ("full_enhancement_10db", pair_out / "full_enhancement_10db/auk_output.wav", pair_out / "full_enhancement_10db/input.wav"),
            ("bandwidth_extension", pair_out / "bandwidth_extension/auk_output.wav", pair_out / "bandwidth_extension/input.wav"),
        ]
        for task, output, reference in specs:
            if not output.exists():
                errors.append(f"{row['pair_id']} {task}: missing")
                continue
            metrics = validate_wav(output, reference)
            metrics.update({"pair_id": row["pair_id"], "task": task})
            validation.append(metrics)
            if not metrics["valid"]:
                errors.append(f"{row['pair_id']} {task}: invalid")

    if errors:
        raise RuntimeError("; ".join(errors))

    if args.listening_out.exists() and args.overwrite:
        shutil.rmtree(args.listening_out)
    args.listening_out.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        pair_out = args.out / row["pair_id"]
        sample_dir = args.listening_out / f"sample_{row['sample_index']:02d}_{row['pair_id']}"
        sample_dir.mkdir(exist_ok=True)
        copies = {
            "00_real.wav": pair_out / "anchor.wav",
            "01_auk_denoise_10db.wav": pair_out / "denoise_10db/auk_output.wav",
            "02_auk_dereverberation.wav": pair_out / "dereverberation/auk_output.wav",
            "03_auk_full_enhancement_10db.wav": pair_out / "full_enhancement_10db/auk_output.wav",
            "04_auk_bandwidth_extension.wav": pair_out / "bandwidth_extension/auk_output.wav",
        }
        for name, source in copies.items():
            shutil.copy2(source, sample_dir / name)

    report = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_outputs": generated_total,
        "task_reports": task_reports,
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "listening_wav_count": sum(1 for _ in args.listening_out.rglob("*.wav")),
        "validation": validation,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "validation"}, indent=2))


if __name__ == "__main__":
    main()
