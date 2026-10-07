#!/usr/bin/env python3
"""Generate a 50-pair TTS duration/boundary-control pilot in two variants."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

from auk.infer.infer_auk import AukInfer, save_audio


SR = 24_000
FRAME_MS = 20
HOP_MS = 10
BOUNDARY_DB = -40.0
TTS_BOUNDARY_BUDGET = 0.28
EN_SEC_PER_UTF8_BYTE = 0.0656
SHORT_TEXT_BYTES = 10
SHORT_TEXT_SPEED = 0.3


def load_mono(path: Path) -> tuple[torch.Tensor, int]:
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio.to(torch.float32), sr


def duration(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def text_duration_estimate(text: str) -> float:
    """Replicate the repository's simple F5-style English duration prior."""
    value = str(text or "")
    if not value.strip():
        return 0.3
    # VCTK anchors in this pilot are English. Spaces and punctuation are retained
    # because the repository estimator assigns them the surrounding language rate.
    weight = len(value.encode("utf-8")) * EN_SEC_PER_UTF8_BYTE
    if len(value.encode("utf-8")) < SHORT_TEXT_BYTES:
        weight /= SHORT_TEXT_SPEED
    return max(0.3, weight)


def quantize_seconds(seconds: float) -> float:
    # AuK's VAE hop is 480 samples at 24 kHz = 20 ms.
    return max(0.02, math.ceil(float(seconds) / 0.02) * 0.02)


def frame_rms(audio: torch.Tensor, sr: int) -> tuple[np.ndarray, int, int]:
    x = audio.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    frame = max(1, int(round(sr * FRAME_MS / 1000)))
    hop = max(1, int(round(sr * HOP_MS / 1000)))
    if len(x) <= frame:
        return np.asarray([float(np.sqrt(np.mean(x * x) + 1e-12))]), frame, hop
    n = 1 + (len(x) - frame) // hop
    xx = x * x
    cs = np.concatenate(([0.0], np.cumsum(xx, dtype=np.float64)))
    sums = cs[frame : frame + n * hop : hop] - cs[: n * hop : hop]
    return np.sqrt(sums / frame + 1e-12), frame, hop


def active_bounds(audio: torch.Tensor, sr: int, threshold_db: float = BOUNDARY_DB) -> tuple[int, int]:
    rms, frame, hop = frame_rms(audio, sr)
    threshold = float(rms.max()) * (10.0 ** (threshold_db / 20.0))
    active = np.flatnonzero(rms >= threshold)
    if len(active) == 0:
        return 0, audio.shape[-1]
    start = int(active[0] * hop)
    end = min(audio.shape[-1], int(active[-1] * hop + frame))
    return start, max(start + 1, end)


def aligned_tts(raw: torch.Tensor, anchor: torch.Tensor, sr: int) -> tuple[torch.Tensor, dict]:
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr}")
    a0, a1 = active_bounds(anchor, sr)
    t0, t1 = active_bounds(raw, sr)
    active = raw[..., t0:t1]
    aligned = torch.zeros(1, (a0 + active.shape[-1] + (anchor.shape[-1] - a1)), dtype=torch.float32)
    aligned[..., a0 : a0 + active.shape[-1]] = active
    return aligned, {
        "anchor_leading_silence_sec": a0 / sr,
        "anchor_trailing_silence_sec": (anchor.shape[-1] - a1) / sr,
        "anchor_active_sec": (a1 - a0) / sr,
        "raw_tts_duration_sec": raw.shape[-1] / sr,
        "raw_tts_leading_silence_sec": t0 / sr,
        "raw_tts_trailing_silence_sec": (raw.shape[-1] - t1) / sr,
        "raw_tts_active_sec": (t1 - t0) / sr,
        "aligned_tts_duration_sec": aligned.shape[-1] / sr,
        "boundary_threshold_db": BOUNDARY_DB,
    }


def messages(text: str, audio: Path) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": f"Say the following with the same voice: '{text}'"},
        {"type": "audio", "audio": str(audio)},
    ]}]


def copy_core_files(source_pair: Path, out_pair: Path, *, hardlink: bool = True) -> None:
    out_pair.mkdir(parents=True, exist_ok=True)
    # Keep the canonical names used by the main matched set.
    for source_name, output_name in (
        ("anchor.wav", "anchor.wav"),
        ("auk_se.wav", "auk_se.wav"),
        ("auk_tse.wav", "auk_tse.wav"),
    ):
        source = source_pair / source_name
        target = out_pair / output_name
        if hardlink:
            try:
                os.link(source, target)
                continue
            except FileExistsError:
                continue
            except OSError:
                pass
        shutil.copy2(source, target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/matched_full_5369_snr0_batch4_generated"))
    ap.add_argument("--out", type=Path, default=Path("data/tts_pilot_v2_v3_50_each"))
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy-core", action="store_true", help="copy core WAVs instead of hard-linking them")
    args = ap.parse_args()

    source_rows = list(csv.DictReader((args.source / "manifest.tsv").open(), delimiter="\t"))[: args.count]
    if len(source_rows) != args.count:
        raise RuntimeError(f"expected {args.count} source rows, found {len(source_rows)}")
    args.out.mkdir(parents=True, exist_ok=True)
    raw_root = args.out / "raw_outputs"
    raw_root.mkdir(parents=True, exist_ok=True)

    variants = {
        "v2_active_duration": {"mode": "active_duration_plus_boundary", "seed_offset": 30_000_000},
        "v3_text_duration": {"mode": "text_duration_plus_boundary", "seed_offset": 40_000_000},
    }
    rows_by_variant = {name: [] for name in variants}
    for variant in variants:
        variant_root = args.out / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        for row in source_rows:
            source_pair = args.source / row["pair_id"]
            out_pair = variant_root / row["pair_id"]
            copy_core_files(source_pair, out_pair, hardlink=not args.copy_core)

    engine = AukInfer("ckpts/AuK/config.yaml", "ckpts/AuK/auk_base.safetensors", dtype=args.dtype)
    started = time.time()
    for variant, spec in variants.items():
        variant_root = args.out / variant
        variant_raw = raw_root / variant
        variant_raw.mkdir(parents=True, exist_ok=True)
        work = []
        for index, row in enumerate(source_rows):
            source_pair = args.source / row["pair_id"]
            anchor, anchor_sr = load_mono(source_pair / "anchor.wav")
            if anchor_sr != SR:
                anchor = torchaudio.functional.resample(anchor, anchor_sr, SR)
                anchor_sr = SR
            a0, a1 = active_bounds(anchor, anchor_sr)
            active_sec = (a1 - a0) / SR
            text_sec = text_duration_estimate(row["text"])
            if spec["mode"] == "active_duration_plus_boundary":
                target_sec = quantize_seconds(active_sec + TTS_BOUNDARY_BUDGET)
            else:
                # V3 uses the text-duration prior directly. The 0.28 s boundary
                # estimate is only needed for V2's anchor-duration constraint;
                # V3's generated boundary is removed and replaced during alignment.
                target_sec = quantize_seconds(text_sec)
            work.append({
                "row": row,
                "source_pair": source_pair,
                "out_pair": variant_root / row["pair_id"],
                "raw_path": variant_raw / f"{row['pair_id']}.wav",
                "reference": source_pair / "tts_reference.wav",
                "anchor": anchor,
                "target_sec": target_sec,
                "active_sec": active_sec,
                "text_sec": text_sec,
                "seed": int(row["seed"]) + spec["seed_offset"],
            })

        for begin in range(0, len(work), args.batch_size):
            batch = work[begin : begin + args.batch_size]
            requests = [messages(item["row"]["text"], item["reference"]) for item in batch]
            outputs, sample_rate = engine.generate_batch(
                requests,
                audios=[str(item["reference"]) for item in batch],
                gen_seconds=[item["target_sec"] for item in batch],
                nfe=args.nfe,
                cfg_strength=args.cfg,
                sway_sampling_coef=args.sway,
                seed=[item["seed"] for item in batch],
            )
            for item, raw in zip(batch, outputs):
                raw = raw.to(torch.float32)
                save_audio(raw, sample_rate, str(item["raw_path"]))
                aligned, metrics = aligned_tts(raw, item["anchor"], sample_rate)
                save_audio(aligned, sample_rate, str(item["out_pair"] / "auk_tts.wav"))
                result = dict(item["row"])
                result.update({
                    "variant": variant,
                    "duration_mode": spec["mode"],
                    "tts_seed": item["seed"],
                    "tts_target_duration_sec": item["target_sec"],
                    "text_duration_estimate_sec": item["text_sec"],
                    **metrics,
                })
                rows_by_variant[variant].append(result)
            done = min(begin + len(batch), len(work))
            print(f"[{variant}] {done}/{len(work)}", flush=True)

        fields = list(rows_by_variant[variant][0])
        with (variant_root / "manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows_by_variant[variant])
        (variant_root / "generation_config.json").write_text(json.dumps({
            "variant": variant,
            "mode": spec["mode"],
            "count": len(source_rows),
            "source_root": str(args.source.resolve()),
            "boundary_threshold_db": BOUNDARY_DB,
            "tts_boundary_budget_sec": TTS_BOUNDARY_BUDGET,
            "text_duration_rule": "English UTF-8 bytes * 0.0656 sec; <10 bytes divided by 0.3",
            "model": "AuK Base",
            "checkpoint": "ckpts/AuK/auk_base.safetensors",
            "nfe": args.nfe,
            "cfg": args.cfg,
            "sway": args.sway,
            "dtype": args.dtype,
        }, indent=2) + "\n")

    summary = {
        "created_at_unix": started,
        "source_count": len(source_rows),
        "variants": list(variants),
        "boundary_threshold_db": BOUNDARY_DB,
        "tts_boundary_budget_sec": TTS_BOUNDARY_BUDGET,
        "elapsed_sec": time.time() - started,
    }
    (args.out / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
