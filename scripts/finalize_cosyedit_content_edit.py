#!/usr/bin/env python3
"""Merge, validate, document, and package the CosyEdit full output set."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torchaudio
import torch


OUT = Path("/data/CosyEdit/content_edit_full_3734_20261006")
PAIR_MANIFEST = Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv")
TEXT_MANIFEST = Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_text_manifest.tsv")
ZIP = Path("/data/CosyEdit/content_edit_full_3734_20261006.zip")


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pair_num(pair_id: str) -> int:
    return int(pair_id.rsplit("_", 1)[-1])


def main():
    accepted = [r for r in rows(PAIR_MANIFEST) if r.get("status") == "accepted"]
    expected = {r["pair_id"]: r for r in accepted}
    wavs = sorted(OUT.glob("pair_*.wav"), key=lambda p: pair_num(p.stem))
    actual = {p.stem for p in wavs}
    missing = sorted(set(expected) - actual, key=pair_num)
    extra = sorted(actual - set(expected), key=pair_num)
    if missing or extra:
        raise RuntimeError(f"file mapping mismatch missing={len(missing)} extra={len(extra)}")

    worker_rows = []
    for path in sorted(OUT.glob("worker_*.tsv")):
        worker_rows.extend(rows(path))
    merged = {r["pair_id"]: r for r in worker_rows if r.get("pair_id")}
    if set(merged) != set(expected):
        missing_manifest = sorted(set(expected) - set(merged), key=pair_num)
        raise RuntimeError(f"worker manifest mismatch missing={len(missing_manifest)}")
    if any(r.get("status") != "ok" for r in merged.values()):
        raise RuntimeError("worker manifest contains non-ok rows")

    bad = []
    durations = []
    sample_rates = {}
    clipped = 0
    for i, path in enumerate(wavs, 1):
        try:
            wav, sr = torchaudio.load(str(path))
            pair_id = path.stem
            if pair_id in merged:
                merged[pair_id]["sample_rate"] = str(sr)
                merged[pair_id]["frames"] = str(int(wav.shape[-1]))
                merged[pair_id]["duration_seconds"] = str(float(wav.shape[-1] / sr))
                if not merged[pair_id].get("seed"):
                    merged[pair_id]["seed"] = str(20261006 + pair_num(pair_id))
            sample_rates[str(sr)] = sample_rates.get(str(sr), 0) + 1
            if wav.numel() == 0 or not torch.isfinite(wav).all():
                bad.append({"file": path.name, "error": "empty_or_nonfinite"})
            if wav.ndim != 2 or wav.shape[0] != 1:
                bad.append({"file": path.name, "error": f"shape={tuple(wav.shape)}"})
            if sr != 22050:
                bad.append({"file": path.name, "error": f"sample_rate={sr}"})
            frac = float((wav.abs() >= 0.999).float().mean()) if wav.numel() else 0.0
            if frac > 0.01:
                clipped += 1
            durations.append(float(wav.shape[-1] / sr))
        except Exception as exc:
            bad.append({"file": path.name, "error": repr(exc)})
        if i % 500 == 0 or i == len(wavs):
            print(f"validated {i}/{len(wavs)}", flush=True)

    validation = {
        "expected_files": len(expected),
        "actual_files": len(wavs),
        "bad_file_count": len(bad),
        "sample_rates": sample_rates,
        "clipping_over_1pct_count": clipped,
        "duration_min_sec": min(durations),
        "duration_max_sec": max(durations),
        "duration_mean_sec": sum(durations) / len(durations),
        "bad_files": bad[:100],
        "status": "pass" if not bad else "warning",
    }
    (OUT / "audio_validation_report.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if bad:
        raise RuntimeError(f"audio validation failed: {len(bad)} issues")

    fields = list(merged[next(iter(merged))].keys())
    with (OUT / "manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged[k] for k in sorted(merged, key=pair_num))

    shutil.copy2(PAIR_MANIFEST, OUT / "content_edit_pair_manifest.tsv")
    shutil.copy2(TEXT_MANIFEST, OUT / "content_edit_text_manifest.tsv")
    readme = f"""# CosyEdit Specialist Content Editing Dataset

This directory contains the full CosyEdit specialist Content Editing output set corresponding to the 3,734 accepted Content Editing pairs.

## Per-pair output

Each `pair_XXXXX.wav` is a CosyEdit edited utterance generated from the original anchor and the source/edited text pair. The original anchor audio remains in:

`/root/AuK/data/matched_full_5369_snr0_batch4_generated/<source_pair_dir>/anchor.wav`

The exact mapping is in `manifest.tsv`. The source-selection and review manifests are also included.

## Generation

- Model: CosyEdit official checkpoint (`/data/CosyEdit/pretrained_models/CosyEdit`)
- Input: `anchor.wav`, `source_text`, `edited_text`
- API: `inference_edit(target_text, original_text, original_speech)`
- Output sample rate: 22,050 Hz
- Four independent model workers on one RTX 4090
- Fixed seed base: 20261006; per-pair seed is recorded in `manifest.tsv`
- Existing outputs were resumed/skipped safely after the parallel worker switch

## Validation

- Expected and generated files: {len(expected)} / {len(wavs)}
- Invalid/empty/non-finite files: {len(bad)}
- Duration range: {min(durations):.3f}–{max(durations):.3f} s
- Mean duration: {sum(durations) / len(durations):.3f} s
- Files with clipping over 1%: {clipped}

The audio was not forcibly time-matched to AuK output. For encoder analysis, resample both systems to the same analysis rate at extraction time.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    ZIP.unlink(missing_ok=True)
    subprocess.run(["zip", "-qr", str(ZIP), OUT.name], cwd=str(OUT.parent), check=True)
    print(json.dumps({"status": "complete", "files": len(wavs), "zip": str(ZIP), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
