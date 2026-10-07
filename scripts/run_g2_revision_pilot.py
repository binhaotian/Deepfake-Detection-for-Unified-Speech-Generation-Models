#!/usr/bin/env python3
"""Prepare and run the 10-pair G2 revision A/B pilot."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchaudio

from auk.infer.infer_auk import AukInfer, save_audio


SR = 24_000
FULL_PROMPT = (
    "Preserve all speakers, remove noise and reverberation, "
    "and output clean speech of the same length."
)
CHANNEL_PROMPT = (
    "Perform speech super-resolution on this bandwidth-limited recording. "
    "Restore the missing high-frequency speech content, preserve the speaker "
    "identity and spoken words, and output audio of the same length."
)


def load(path: Path) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(0, keepdim=True).to(torch.float32)
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    return audio


def rms(audio: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(audio.double() ** 2)))


def fit(audio: torch.Tensor, samples: int) -> torch.Tensor:
    if audio.shape[-1] >= samples:
        return audio[..., :samples]
    return torch.nn.functional.pad(audio, (0, samples - audio.shape[-1]))


def duration(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def messages(prompt: str, audio: Path) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "audio", "audio": str(audio)},
    ]}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    rows = list(csv.DictReader(args.guide.open(), delimiter="\t"))
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 guide rows, found {len(rows)}")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    prep_errors = []
    for index, row in enumerate(rows):
        pair_id = row["pair_id"]
        src = args.pilot_root / pair_id
        original = args.source_root / pair_id
        dst = args.out / pair_id
        dst.mkdir(parents=True, exist_ok=True)
        full = dst / "full_enhancement_snr10"
        channel = dst / "channel_super_resolution"
        full.mkdir(exist_ok=True)
        channel.mkdir(exist_ok=True)

        copy_map = {
            dst / "anchor.wav": src / "anchor.wav",
            full / "full_snr0_input.wav": src / "full_enhancement/full_enhance_input.wav",
            full / "auk_full_snr0.wav": src / "full_enhancement/auk_full_enhance.wav",
            channel / "channel_input.wav": src / "channel_restoration/channel_input.wav",
            channel / "auk_channel_old_prompt.wav": src / "channel_restoration/auk_channel_restore.wav",
        }
        for target, source in copy_map.items():
            if not target.exists() or args.overwrite:
                shutil.copy2(source, target)

        speech = load(src / "dereverb/dereverb_input.wav")
        noise = fit(load(original / "se_noise.wav"), speech.shape[-1])
        noise *= rms(speech) / (10.0 ** (10.0 / 20.0) * max(rms(noise), 1e-12))
        mixture = speech + noise
        peak = float(mixture.abs().max())
        shared_gain = min(1.0, 0.98 / max(peak, 1e-12))
        speech *= shared_gain
        noise *= shared_gain
        mixture *= shared_gain
        torchaudio.save(str(full / "full_snr10_speech.wav"), speech, SR)
        torchaudio.save(str(full / "full_snr10_noise.wav"), noise, SR)
        torchaudio.save(str(full / "full_snr10_input.wav"), mixture, SR)
        measured_snr = 20.0 * math.log10(rms(speech) / max(rms(noise), 1e-12))

        for path in (full / "full_snr10_input.wav", channel / "channel_input.wav"):
            audio, sr = torchaudio.load(str(path))
            if sr != SR or audio.shape[0] != 1 or not torch.isfinite(audio).all():
                prep_errors.append(f"{pair_id}: invalid {path.name}")

        metadata = {
            "pair_id": pair_id,
            "speaker": row["speaker"],
            "text": row["text"],
            "full_enhancement": {
                "requested_snr_db": 10.0,
                "measured_snr_db": measured_snr,
                "same_rir_as_original_pilot": True,
                "same_noise_id_as_original_pair": row["noise_id"],
                "shared_peak_gain": shared_gain,
                "instruction": FULL_PROMPT,
            },
            "channel": {
                "same_input_as_original_pilot": True,
                "old_instruction": "Repair the telephone effect and restore natural, clear speech.",
                "new_instruction": CHANNEL_PROMPT,
            },
        }
        (dst / "revision_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append({
            "pair_id": pair_id,
            "speaker": row["speaker"],
            "text": row["text"],
            "estimated_rt60_seconds": row["estimated_rt60_seconds"],
            "noise_id": row["noise_id"],
            "full_snr_db": f"{measured_snr:.8f}",
            "seed": str(20_261_007 + index),
        })

    with (args.out / "manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(manifest)
    if prep_errors:
        raise RuntimeError("; ".join(prep_errors))

    config = {
        "created_at_utc": started.isoformat(),
        "pilot_root": str(args.pilot_root.resolve()),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.out.resolve()),
        "pair_count": len(manifest),
        "model": "AuK Base",
        "checkpoint": str(Path("ckpts/AuK/auk_base.safetensors").resolve()),
        "dtype": "bf16", "nfe": 32, "cfg": 2.0, "sway": -1.0,
        "batch_size": args.batch_size,
        "full_enhancement_snr_db": 10.0,
        "full_prompt": FULL_PROMPT,
        "channel_prompt": CHANNEL_PROMPT,
        "resume_policy": "skip existing output unless --overwrite",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    engine = AukInfer("ckpts/AuK/config.yaml", "ckpts/AuK/auk_base.safetensors", dtype="bf16")
    tasks = {
        "full_snr10": {
            "input": Path("full_enhancement_snr10/full_snr10_input.wav"),
            "output": Path("full_enhancement_snr10/auk_full_snr10.wav"),
            "prompt": FULL_PROMPT,
            "seed_offset": 6_000_000,
        },
        "channel_explicit": {
            "input": Path("channel_super_resolution/channel_input.wav"),
            "output": Path("channel_super_resolution/auk_channel_explicit_prompt.wav"),
            "prompt": CHANNEL_PROMPT,
            "seed_offset": 7_000_000,
        },
    }
    progress = args.out / "progress.jsonl"
    task_reports = {}
    generated_total = 0
    for task, spec in tasks.items():
        work = []
        for row in manifest:
            pair = args.out / row["pair_id"]
            output = pair / spec["output"]
            if output.exists() and not args.overwrite:
                continue
            input_path = pair / spec["input"]
            work.append({"pair_id": row["pair_id"], "input": input_path, "output": output,
                         "seconds": duration(input_path), "seed": int(row["seed"]) + spec["seed_offset"]})
        work.sort(key=lambda item: item["seconds"])
        task_started = time.perf_counter()
        task_count = 0
        for begin in range(0, len(work), args.batch_size):
            batch = work[begin:begin + args.batch_size]
            outputs, sample_rate = engine.generate_batch(
                [messages(spec["prompt"], item["input"]) for item in batch],
                audios=[str(item["input"]) for item in batch],
                gen_seconds=[None] * len(batch), nfe=32, cfg_strength=2.0,
                sway_sampling_coef=-1.0, seed=[item["seed"] for item in batch],
            )
            for item, output in zip(batch, outputs):
                save_audio(output, sample_rate, str(item["output"]))
                task_count += 1; generated_total += 1
            event = {"time_utc": datetime.now(timezone.utc).isoformat(), "task": task,
                     "pair_ids": [item["pair_id"] for item in batch], "generated_total": generated_total}
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(f"[{task}] {task_count}/{len(work)} total={generated_total}", flush=True)
        elapsed = time.perf_counter() - task_started
        task_reports[task] = {"generated": task_count, "wall_seconds": elapsed,
                              "seconds_per_output": elapsed / task_count if task_count else 0.0}

    errors = []
    output_metrics = []
    for row in manifest:
        pair = args.out / row["pair_id"]
        for task, spec in tasks.items():
            inp = pair / spec["input"]; out = pair / spec["output"]
            if not out.exists(): errors.append(f"{row['pair_id']} {task}: missing"); continue
            x, sr = torchaudio.load(str(out))
            delta = abs(duration(inp) - x.shape[-1] / sr)
            if sr != SR or x.shape[0] != 1 or not torch.isfinite(x).all() or delta > 0.06:
                errors.append(f"{row['pair_id']} {task}: invalid output")
            output_metrics.append({"pair_id": row["pair_id"], "task": task,
                                   "duration_error_seconds": delta, "peak": float(x.abs().max()),
                                   "clipped_fraction": float((x.abs() >= .999).float().mean())})
    report = {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "generated_outputs": generated_total, "task_reports": task_reports,
              "validation_error_count": len(errors), "validation_errors": errors,
              "output_metrics": output_metrics}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "output_metrics"}, indent=2))
    if errors:
        raise SystemExit("revision pilot validation failed")


if __name__ == "__main__":
    main()
