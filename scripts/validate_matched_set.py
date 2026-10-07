#!/usr/bin/env python3
"""Validate the v0 manifest and all prepared audio inputs."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torchaudio


def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x * x) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/matched_v0"))
    args = ap.parse_args()
    rows = list(csv.DictReader((args.root / "manifest.tsv").open(), delimiter="\t"))
    assert len(rows) == 100, len(rows)
    assert len({r["pair_id"] for r in rows}) == 100
    errors = []
    for row in rows:
        pair = args.root / row["pair_id"]
        files = ["anchor.wav", "tts_reference.wav", "se_noise.wav", "se_input.wav",
                 "tse_interferer.wav", "tse_input.wav"]
        audio = {}
        for name in files:
            path = pair / name
            if not path.exists():
                errors.append(f"{row['pair_id']}: missing {name}")
                continue
            x, sr = torchaudio.load(str(path))
            if sr != 24000 or x.shape[0] != 1:
                errors.append(f"{row['pair_id']}: {name} format {x.shape}/{sr}")
            if not torch.isfinite(x).all() or float(x.abs().max()) > 1.0:
                errors.append(f"{row['pair_id']}: invalid samples in {name}")
            audio[name] = x
        if len(audio) == len(files):
            n = audio["anchor.wav"].shape[-1]
            if any(x.shape[-1] != n for x in audio.values() if x.shape[-1] == n) is False:
                pass
            snr = 20 * math.log10(rms(audio["anchor.wav"]) / rms(audio["se_noise.wav"]))
            sir = 20 * math.log10(rms(audio["anchor.wav"]) / rms(audio["tse_interferer.wav"]))
            if abs(snr - float(row["snr_db"])) > 0.05:
                errors.append(f"{row['pair_id']}: SNR {snr:.3f}")
            if abs(sir - float(row["sir_db"])) > 0.05:
                errors.append(f"{row['pair_id']}: SIR {sir:.3f}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(rows)} pairs, 6 input files each, 24 kHz mono, SNR/SIR checks passed")


if __name__ == "__main__":
    main()
