#!/usr/bin/env python3
"""Generate the TTS, SE and TSE conditions for prepared matched pairs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torchaudio

from auk.infer.infer_auk import AukInfer, save_audio


def duration(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def messages(instruction: str, audio: Path) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "audio", "audio": str(audio)},
    ]}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/matched_v0"))
    ap.add_argument("--pair", default="pair_0001")
    ap.add_argument("--pairs", default=None, help="comma-separated pair ids; overrides --pair")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    args = ap.parse_args()

    rows = {r["pair_id"]: r for r in csv.DictReader((args.root / "manifest.tsv").open(), delimiter="\t")}
    pair_ids = [p.strip() for p in (args.pairs or args.pair).split(",") if p.strip()]
    engine = AukInfer("ckpts/AuK/config.yaml", "ckpts/AuK/auk_base.safetensors", dtype=args.dtype)
    for pair_index, pair_id in enumerate(pair_ids):
        row = rows[pair_id]
        pair = args.root / pair_id
        text = row["text"]
        jobs = [
            ("auk_tts.wav", pair / "tts_reference.wav", f"Say the following with the same voice: '{text}'", duration(pair / "anchor.wav")),
            ("auk_se.wav", pair / "se_input.wav", "Remove only the background noise, preserve everything else, and output audio of the same length.", None),
            ("auk_tse.wav", pair / "tse_input.wav", f'Keep only the speaker who says "{text}" and remove all other speakers.', None),
        ]
        for offset, (name, audio, instruction, secs) in enumerate(jobs):
            output = pair / name
            if output.exists():
                print(f"skip existing {output}")
                continue
            out, sr = engine.generate(
                messages(instruction, audio), audio=str(audio), gen_seconds=secs,
                nfe=args.nfe, cfg_strength=2.0, sway_sampling_coef=-1.0,
                seed=args.seed + pair_index * len(jobs) + offset,
            )
            save_audio(out, sr, str(output))
            print(f"saved {output} ({out.shape[-1] / sr:.2f}s @ {sr} Hz)")


if __name__ == "__main__":
    main()
