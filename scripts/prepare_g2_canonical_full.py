#!/usr/bin/env python3
"""Prepare deterministic full-scale G2 canonical restoration inputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torchaudio

from prepare_g2_restoration_pilot import (
    DEFAULT_SEED,
    EFFECTIVE_CHANNEL_SR,
    TARGET_SR,
    audio_metrics,
    discover_rirs,
    fit_length,
    load_mono,
    polyphase_bandlimit,
    prepare_rir,
    rms,
    save_mono,
    select_balanced_rows,
    select_room_diverse_rirs,
)


SNR_DB = 10.0


def scaled_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    target_noise_rms = rms(speech) / (10.0 ** (snr_db / 20.0))
    return noise * (target_noise_rms / max(rms(noise), 1e-12))


def protect_pair(speech: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mixture = speech + noise
    gain = min(1.0, 0.98 / max(float(np.max(np.abs(mixture))), 1e-12))
    return speech * gain, noise * gain, mixture * gain, gain


def validate_audio(path: Path, expected_samples: int) -> list[str]:
    errors = []
    audio, sample_rate = torchaudio.load(str(path))
    if sample_rate != TARGET_SR or audio.shape != (1, expected_samples):
        errors.append(
            f"{path}: shape={tuple(audio.shape)} sr={sample_rate}; "
            f"expected=(1,{expected_samples})/{TARGET_SR}"
        )
    if not torch.isfinite(audio).all():
        errors.append(f"{path}: NaN/Inf")
    if float(audio.abs().max()) > 1.0:
        errors.append(f"{path}: peak > 1")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/matched_full_5369_snr0_batch4_generated"),
    )
    parser.add_argument("--rir-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rir-pool-size", type=int, default=20)
    parser.add_argument("--rir-scan-limit", type=int, default=2000)
    parser.add_argument("--rt60-low", type=float, default=0.45)
    parser.add_argument("--rt60-high", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_manifest = args.source / "manifest.tsv"
    source_rows = list(csv.DictReader(source_manifest.open(encoding="utf-8"), delimiter="\t"))
    rows = select_balanced_rows(source_rows, len(source_rows))
    args.out.mkdir(parents=True, exist_ok=True)

    eligible_rirs, all_rirs = discover_rirs(
        args.rir_root,
        args.rt60_low,
        args.rt60_high,
        args.rir_scan_limit,
    )
    selected_band = [args.rt60_low, args.rt60_high]
    if len(eligible_rirs) < min(args.rir_pool_size, 10):
        selected_band = [0.35, 0.75]
        eligible_rirs = [
            item for item in all_rirs
            if item["estimated_rt60_seconds"] is not None
            and selected_band[0] <= item["estimated_rt60_seconds"] <= selected_band[1]
        ]
    if not eligible_rirs:
        raise RuntimeError("no eligible RIRs")
    eligible_rirs.sort(key=lambda item: (item["estimated_rt60_seconds"], str(item["path"])))
    rir_pool = select_room_diverse_rirs(
        eligible_rirs,
        args.rir_pool_size,
        center_rt60=sum(selected_band) / 2.0,
    )
    rir_records = []
    prepared_rirs = {}
    for index, item in enumerate(rir_pool, start=1):
        rir_id = f"rir_{index:03d}"
        prepared_rirs[rir_id] = prepare_rir(item)
        rir_records.append({
            "rir_id": rir_id,
            "source_path": str(item["path"].resolve()),
            "source_sha256": item["source_sha256"],
            "room_id": item["room_id"],
            "estimated_rt60_seconds": item["estimated_rt60_seconds"],
            "first_arrival_sample_24k": item["first_arrival_sample_24k"],
        })

    manifest_rows = []
    validation_errors = []
    prepared_count = 0
    skipped_count = 0
    for index, row in enumerate(rows):
        pair_id = row["pair_id"]
        source_pair = args.source / pair_id
        pair_root = args.out / pair_id
        pair_root.mkdir(parents=True, exist_ok=True)
        denoise_path = pair_root / "denoise_10db/input.wav"
        dereverb_path = pair_root / "dereverberation/input.wav"
        full_path = pair_root / "full_enhancement_10db/input.wav"
        bandwidth_path = pair_root / "bandwidth_extension/input.wav"
        metadata_path = pair_root / "input_metadata.json"

        rir_record = rir_records[index % len(rir_records)]
        expected = (denoise_path, dereverb_path, full_path, bandwidth_path, metadata_path)
        needs_prepare = args.overwrite or not all(path.exists() for path in expected)

        if needs_prepare:
            anchor, _ = load_mono(source_pair / "anchor.wav")
            noise, _ = load_mono(source_pair / "se_noise.wav")
            noise = fit_length(noise, len(anchor))

            denoise_noise = scaled_noise(anchor, noise, SNR_DB)
            denoise_speech, denoise_noise, denoise_input, denoise_peak_gain = protect_pair(
                anchor.copy(), denoise_noise
            )

            rir = prepared_rirs[rir_record["rir_id"]]
            reverberant = scipy.signal.fftconvolve(anchor, rir, mode="full")[: len(anchor)]
            reverberant_rms_before = rms(reverberant)
            if reverberant_rms_before <= 1e-12:
                raise RuntimeError(f"silent reverberant signal for {pair_id}")
            reverb_rms_gain = rms(anchor) / reverberant_rms_before
            reverberant *= reverb_rms_gain
            reverb_peak_gain = min(
                1.0,
                0.98 / max(float(np.max(np.abs(reverberant))), 1e-12),
            )
            reverberant *= reverb_peak_gain

            full_noise = scaled_noise(reverberant, noise, SNR_DB)
            full_speech, full_noise, full_input, full_peak_gain = protect_pair(
                reverberant.copy(), full_noise
            )
            bandwidth_input = polyphase_bandlimit(anchor)

            save_mono(denoise_path, denoise_input)
            save_mono(dereverb_path, reverberant)
            save_mono(full_path, full_input)
            save_mono(bandwidth_path, bandwidth_input)

            denoise_measured_snr = 20.0 * math.log10(
                rms(denoise_speech) / max(rms(denoise_noise), 1e-12)
            )
            full_measured_snr = 20.0 * math.log10(
                rms(full_speech) / max(rms(full_noise), 1e-12)
            )
            metadata = {
                "pair_id": pair_id,
                "source_pair": str(source_pair.resolve()),
                "anchor_path": str((source_pair / "anchor.wav").resolve()),
                "noise_path": str((source_pair / "se_noise.wav").resolve()),
                "speaker": row["speaker"],
                "text": row["text"],
                "seed": args.seed + index,
                "rir": rir_record,
                "rt60_selection_band_seconds": selected_band,
                "denoise_10db": {
                    "formula": "anchor + noise",
                    "noise_id": row["noise_id"],
                    "noise_category": row["noise_category"],
                    "requested_snr_db": SNR_DB,
                    "measured_snr_db": denoise_measured_snr,
                    "peak_protection_gain": denoise_peak_gain,
                    "input_metrics": audio_metrics(denoise_input),
                },
                "dereverberation": {
                    "formula": "anchor * rir",
                    "convolution": "scipy.signal.fftconvolve/full_then_crop",
                    "anchor_rms": rms(anchor),
                    "reverberant_rms_before_gain": reverberant_rms_before,
                    "rms_gain": reverb_rms_gain,
                    "peak_protection_gain": reverb_peak_gain,
                    "input_metrics": audio_metrics(reverberant),
                },
                "full_enhancement_10db": {
                    "formula": "(anchor * same rir) + same-identity noise",
                    "noise_id": row["noise_id"],
                    "noise_category": row["noise_category"],
                    "requested_snr_db": SNR_DB,
                    "measured_snr_db": full_measured_snr,
                    "peak_protection_gain": full_peak_gain,
                    "input_metrics": audio_metrics(full_input),
                },
                "bandwidth_extension": {
                    "operation": "24k_to_8k_to_24k_polyphase",
                    "effective_sample_rate": EFFECTIVE_CHANNEL_SR,
                    "effective_nyquist_hz": EFFECTIVE_CHANNEL_SR / 2,
                    "input_metrics": audio_metrics(bandwidth_input),
                },
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            prepared_count += 1
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            denoise_measured_snr = metadata["denoise_10db"]["measured_snr_db"]
            full_measured_snr = metadata["full_enhancement_10db"]["measured_snr_db"]
            skipped_count += 1

        expected_samples = torchaudio.info(str(source_pair / "anchor.wav")).num_frames
        for path in (denoise_path, dereverb_path, full_path, bandwidth_path):
            validation_errors.extend(validate_audio(path, expected_samples))

        manifest_rows.append({
            "pair_id": pair_id,
            "anchor_id": row["anchor_id"],
            "speaker": row["speaker"],
            "gender": row["gender"],
            "text": row["text"],
            "source_seed": row["seed"],
            "g2_seed": args.seed + index,
            "anchor_path": str((source_pair / "anchor.wav").resolve()),
            "noise_id": row["noise_id"],
            "noise_category": row["noise_category"],
            "rir_id": rir_record["rir_id"],
            "estimated_rt60_seconds": f"{rir_record['estimated_rt60_seconds']:.8f}",
            "denoise_requested_snr_db": f"{SNR_DB:.1f}",
            "denoise_measured_snr_db": f"{denoise_measured_snr:.8f}",
            "full_requested_snr_db": f"{SNR_DB:.1f}",
            "full_measured_snr_db": f"{full_measured_snr:.8f}",
            "pair_dir": pair_id,
        })
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"prepared/checked {index + 1}/{len(rows)}", flush=True)

    with (args.out / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "output_root": str(args.out.resolve()),
        "pair_count": len(manifest_rows),
        "speaker_count": len({row["speaker"] for row in manifest_rows}),
        "selection": "all source rows, speaker-balanced round-robin ordering",
        "seed": args.seed,
        "sample_rate": TARGET_SR,
        "snr_db": SNR_DB,
        "rir_source_root": str(args.rir_root.resolve()),
        "rir_pool_size": len(rir_records),
        "rir_scan_limit": args.rir_scan_limit,
        "used_rt60_band_seconds": selected_band,
        "bandwidth_operation": "24k_to_8k_to_24k_polyphase",
    }
    (args.out / "input_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.out / "rir_manifest.json").write_text(json.dumps(rir_records, indent=2) + "\n", encoding="utf-8")
    report = {
        "pair_count": len(manifest_rows),
        "speaker_count": len({row["speaker"] for row in manifest_rows}),
        "prepared_count": prepared_count,
        "skipped_existing_count": skipped_count,
        "rir_pool_size": len(rir_records),
        "rt60_min": min(row["estimated_rt60_seconds"] for row in rir_records),
        "rt60_max": max(row["estimated_rt60_seconds"] for row in rir_records),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
    }
    (args.out / "input_preparation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if validation_errors:
        raise SystemExit("input preparation validation failed")


if __name__ == "__main__":
    main()
