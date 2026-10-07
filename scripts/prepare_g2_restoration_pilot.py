#!/usr/bin/env python3
"""Prepare matched AuK G2 restoration inputs for a small pilot.

The script is deliberately deterministic and never modifies the completed
5,369-pair dataset.  It selects a speaker-balanced anchor subset and prepares:

* dereverberation: anchor convolved with a moderate medium-room RIR;
* full enhancement: the same reverberant anchor plus the pair's existing noise;
* channel restoration: 24 kHz -> 8 kHz -> 24 kHz bandwidth limitation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torchaudio


TARGET_SR = 24_000
EFFECTIVE_CHANNEL_SR = 8_000
DEFAULT_SEED = 20_261_007


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mono(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    return audio.squeeze(0).numpy().astype(np.float64, copy=False), sr


def save_mono(path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    torchaudio.save(str(path), tensor, sr)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def first_arrival_index(rir: np.ndarray, relative_db: float = -20.0) -> int:
    magnitude = np.abs(rir)
    peak = float(magnitude.max(initial=0.0))
    if peak <= 0:
        raise ValueError("RIR is silent")
    threshold = peak * (10.0 ** (relative_db / 20.0))
    candidates = np.flatnonzero(magnitude >= threshold)
    return int(candidates[0]) if len(candidates) else int(np.argmax(magnitude))


def estimate_rt60(rir: np.ndarray, sr: int) -> float | None:
    """Estimate RT60 with a Schroeder energy-decay fit.

    Prefer T30 (-5 to -35 dB), then T20 (-5 to -25 dB).  The estimate is used
    for controlled selection/metadata, not as a ground-truth acoustic label.
    """
    energy = np.square(rir, dtype=np.float64)
    total = float(energy.sum())
    if total <= 1e-14:
        return None
    decay = np.cumsum(energy[::-1])[::-1]
    decay_db = 10.0 * np.log10(np.maximum(decay / decay[0], 1e-12))
    times = np.arange(len(rir), dtype=np.float64) / sr
    for lower in (-35.0, -25.0):
        mask = (decay_db <= -5.0) & (decay_db >= lower)
        if int(mask.sum()) < 20:
            continue
        slope, _ = np.polyfit(times[mask], decay_db[mask], 1)
        if slope < -1e-6:
            estimate = -60.0 / slope
            if 0.05 <= estimate <= 5.0:
                return float(estimate)
    return None


def select_balanced_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    for speaker_rows in by_speaker.values():
        speaker_rows.sort(key=lambda row: row["pair_id"])

    selected: list[dict[str, str]] = []
    level = 0
    speakers = sorted(by_speaker)
    while len(selected) < count:
        added = False
        for speaker in speakers:
            if level < len(by_speaker[speaker]):
                selected.append(by_speaker[speaker][level])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        level += 1
    if len(selected) != count:
        raise RuntimeError(f"requested {count} rows but selected {len(selected)}")
    return selected


def discover_rirs(
    root: Path,
    rt60_low: float,
    rt60_high: float,
    scan_limit: int,
) -> tuple[list[dict], list[dict]]:
    candidates = sorted(root.rglob("*.wav"))
    if not candidates:
        raise FileNotFoundError(f"no RIR WAV files under {root}")
    if scan_limit > 0 and len(candidates) > scan_limit:
        indices = np.linspace(0, len(candidates) - 1, scan_limit, dtype=int)
        candidates = [candidates[int(index)] for index in indices]
    all_metadata = []
    eligible = []
    for path in candidates:
        rir, _ = load_mono(path, TARGET_SR)
        onset = first_arrival_index(rir)
        aligned = rir[onset:]
        estimate = estimate_rt60(aligned, TARGET_SR)
        metadata = {
            "path": path,
            "room_id": path.parent.name,
            "source_sha256": sha256(path),
            "first_arrival_sample_24k": onset,
            "estimated_rt60_seconds": estimate,
        }
        all_metadata.append(metadata)
        if estimate is not None and rt60_low <= estimate <= rt60_high:
            eligible.append(metadata)
    return eligible, all_metadata


def evenly_spaced(items: list[dict], count: int) -> list[dict]:
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(index)] for index in indices]


def select_room_diverse_rirs(items: list[dict], count: int, center_rt60: float) -> list[dict]:
    by_room: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_room[item["room_id"]].append(item)
    representatives = []
    for room_id in sorted(by_room):
        representatives.append(min(
            by_room[room_id],
            key=lambda item: (abs(item["estimated_rt60_seconds"] - center_rt60), str(item["path"])),
        ))
    representatives.sort(key=lambda item: (item["estimated_rt60_seconds"], item["room_id"], str(item["path"])))
    return evenly_spaced(representatives, min(count, len(representatives)))


def prepare_rir(metadata: dict) -> np.ndarray:
    rir, _ = load_mono(metadata["path"], TARGET_SR)
    rir = rir[int(metadata["first_arrival_sample_24k"]):]
    norm = float(np.linalg.norm(rir))
    if norm <= 1e-12:
        raise ValueError(f"silent RIR: {metadata['path']}")
    return rir / norm


def polyphase_bandlimit(audio: np.ndarray) -> np.ndarray:
    down = scipy.signal.resample_poly(audio, EFFECTIVE_CHANNEL_SR, TARGET_SR)
    up = scipy.signal.resample_poly(down, TARGET_SR, EFFECTIVE_CHANNEL_SR)
    return fit_length(up, len(audio))


def audio_metrics(audio: np.ndarray) -> dict[str, float | int | bool]:
    return {
        "samples": int(len(audio)),
        "seconds": float(len(audio) / TARGET_SR),
        "rms": rms(audio),
        "peak": float(np.max(np.abs(audio), initial=0.0)),
        "finite": bool(np.isfinite(audio).all()),
        "clipped_samples": int(np.sum(np.abs(audio) >= 0.999)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/matched_full_5369_snr0_batch4_generated"),
    )
    parser.add_argument("--rir-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--rir-pool-size", type=int, default=20)
    parser.add_argument("--rir-scan-limit", type=int, default=2000)
    parser.add_argument("--rt60-low", type=float, default=0.45)
    parser.add_argument("--rt60-high", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.source / "manifest.tsv"
    rows = list(csv.DictReader(manifest_path.open(), delimiter="\t"))
    selected = select_balanced_rows(rows, args.count)

    eligible_rirs, all_rirs = discover_rirs(
        args.rir_root, args.rt60_low, args.rt60_high, args.rir_scan_limit
    )
    selected_band = [args.rt60_low, args.rt60_high]
    if len(eligible_rirs) < min(args.rir_pool_size, 10):
        fallback_low, fallback_high = 0.35, 0.75
        eligible_rirs = [
            item for item in all_rirs
            if item["estimated_rt60_seconds"] is not None
            and fallback_low <= item["estimated_rt60_seconds"] <= fallback_high
        ]
        selected_band = [fallback_low, fallback_high]
    if not eligible_rirs:
        raise RuntimeError("no RIRs satisfy the requested or fallback RT60 range")
    eligible_rirs.sort(key=lambda item: (item["estimated_rt60_seconds"], str(item["path"])))
    rir_pool = select_room_diverse_rirs(
        eligible_rirs,
        args.rir_pool_size,
        center_rt60=sum(selected_band) / 2,
    )
    rng = random.Random(args.seed)
    rng.shuffle(rir_pool)

    args.out.mkdir(parents=True, exist_ok=True)
    rir_output_root = args.out / "rir_pool"
    rir_output_root.mkdir(exist_ok=True)
    prepared_rirs: dict[str, np.ndarray] = {}
    rir_records = []
    for index, item in enumerate(rir_pool, 1):
        rir_id = f"rir_{index:03d}"
        prepared = prepare_rir(item)
        output_path = rir_output_root / f"{rir_id}.wav"
        save_mono(output_path, prepared)
        prepared_rirs[rir_id] = prepared
        rir_records.append({
            "rir_id": rir_id,
            "source_path": str(item["path"].resolve()),
            "room_id": item["room_id"],
            "source_sha256": item["source_sha256"],
            "prepared_path": str(output_path.relative_to(args.out)),
            "prepared_sha256": sha256(output_path),
            "estimated_rt60_seconds": item["estimated_rt60_seconds"],
            "first_arrival_sample_24k": item["first_arrival_sample_24k"],
        })

    manifest_rows = []
    validation_errors = []
    for index, row in enumerate(selected):
        pair_id = row["pair_id"]
        source_pair = args.source / pair_id
        pair_root = args.out / pair_id
        if pair_root.exists() and args.overwrite:
            shutil.rmtree(pair_root)
        pair_root.mkdir(parents=True, exist_ok=True)

        anchor, _ = load_mono(source_pair / "anchor.wav")
        existing_noise, _ = load_mono(source_pair / "se_noise.wav")
        existing_noise = fit_length(existing_noise, len(anchor))
        anchor_rms = rms(anchor)

        rir_record = rir_records[index % len(rir_records)]
        rir = prepared_rirs[rir_record["rir_id"]]
        reverberant = scipy.signal.fftconvolve(anchor, rir, mode="full")[: len(anchor)]
        reverberant_rms_before = rms(reverberant)
        if reverberant_rms_before <= 1e-12:
            raise RuntimeError(f"silent reverberant signal for {pair_id}")
        reverb_rms_gain = anchor_rms / reverberant_rms_before
        reverberant = reverberant * reverb_rms_gain
        reverb_peak_gain = min(1.0, 0.98 / max(float(np.max(np.abs(reverberant))), 1e-12))
        reverberant = reverberant * reverb_peak_gain

        noise_rms_before = rms(existing_noise)
        full_noise = existing_noise * (rms(reverberant) / max(noise_rms_before, 1e-12))
        full_speech = reverberant.copy()
        full_input = full_speech + full_noise
        full_peak_gain = min(1.0, 0.98 / max(float(np.max(np.abs(full_input))), 1e-12))
        full_speech *= full_peak_gain
        full_noise *= full_peak_gain
        full_input *= full_peak_gain

        channel_input = polyphase_bandlimit(anchor)

        shutil.copy2(source_pair / "anchor.wav", pair_root / "anchor.wav")
        dereverb_root = pair_root / "dereverb"
        full_root = pair_root / "full_enhancement"
        channel_root = pair_root / "channel_restoration"
        save_mono(dereverb_root / "dereverb_input.wav", reverberant)
        save_mono(full_root / "full_enhance_speech.wav", full_speech)
        save_mono(full_root / "full_enhance_noise.wav", full_noise)
        save_mono(full_root / "full_enhance_input.wav", full_input)
        save_mono(channel_root / "channel_input.wav", channel_input)

        measured_snr = 20.0 * math.log10(rms(full_speech) / max(rms(full_noise), 1e-12))
        metadata = {
            "pair_id": pair_id,
            "source_pair": str(source_pair.resolve()),
            "speaker": row["speaker"],
            "text": row["text"],
            "seed": args.seed + index,
            "rir": rir_record,
            "rt60_selection_band_seconds": selected_band,
            "dereverb": {
                "convolution": "scipy.signal.fftconvolve/full_then_crop",
                "rir_resampled_to_hz": TARGET_SR,
                "anchor_rms": anchor_rms,
                "reverberant_rms_before_gain": reverberant_rms_before,
                "rms_gain": reverb_rms_gain,
                "peak_protection_gain": reverb_peak_gain,
                "input_metrics": audio_metrics(reverberant),
            },
            "full_enhancement": {
                "formula": "(anchor * rir) + noise",
                "noise_id": row["noise_id"],
                "noise_category": row["noise_category"],
                "requested_snr_db": 0.0,
                "measured_snr_db": measured_snr,
                "peak_protection_gain": full_peak_gain,
                "input_metrics": audio_metrics(full_input),
            },
            "channel_restoration": {
                "operation": "narrowband_8khz",
                "source_sample_rate": TARGET_SR,
                "effective_sample_rate": EFFECTIVE_CHANNEL_SR,
                "output_sample_rate": TARGET_SR,
                "resampler": "scipy.signal.resample_poly",
                "codec": None,
                "input_metrics": audio_metrics(channel_input),
            },
        }
        (pair_root / "input_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        paths = {
            "anchor_path": pair_root / "anchor.wav",
            "dereverb_input_path": dereverb_root / "dereverb_input.wav",
            "full_enhance_input_path": full_root / "full_enhance_input.wav",
            "channel_input_path": channel_root / "channel_input.wav",
        }
        for label, path in paths.items():
            audio, sr = torchaudio.load(str(path))
            if sr != TARGET_SR or audio.shape != (1, len(anchor)):
                validation_errors.append(
                    f"{pair_id} {label}: shape={tuple(audio.shape)} sr={sr}, expected=(1,{len(anchor)})/{TARGET_SR}"
                )
            if not torch.isfinite(audio).all():
                validation_errors.append(f"{pair_id} {label}: NaN/Inf")
            if float(audio.abs().max()) > 1.0:
                validation_errors.append(f"{pair_id} {label}: peak > 1")

        manifest_rows.append({
            "pair_id": pair_id,
            "source_pair_id": pair_id,
            "anchor_id": row["anchor_id"],
            "speaker": row["speaker"],
            "gender": row["gender"],
            "text": row["text"],
            "source_seed": row["seed"],
            "pilot_seed": args.seed + index,
            "rir_id": rir_record["rir_id"],
            "estimated_rt60_seconds": f"{rir_record['estimated_rt60_seconds']:.8f}",
            "noise_id": row["noise_id"],
            "noise_category": row["noise_category"],
            "requested_snr_db": "0.0",
            "measured_snr_db": f"{measured_snr:.8f}",
            "channel_operation": "narrowband_8khz",
            "pair_dir": pair_id,
        })

    fields = list(manifest_rows[0])
    with (args.out / "manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "output_root": str(args.out.resolve()),
        "pair_count": len(manifest_rows),
        "speaker_count": len({row["speaker"] for row in manifest_rows}),
        "selection": "speaker-balanced round robin over sorted pair IDs",
        "seed": args.seed,
        "sample_rate": TARGET_SR,
        "rir_source_root": str(args.rir_root.resolve()),
        "rir_pool_size": len(rir_records),
        "rir_scan_limit": args.rir_scan_limit,
        "requested_rt60_band_seconds": [args.rt60_low, args.rt60_high],
        "used_rt60_band_seconds": selected_band,
        "full_enhancement_snr_db": 0.0,
        "channel_effective_sample_rate": EFFECTIVE_CHANNEL_SR,
        "instructions": {
            "dereverb": "Remove only the room reverberation, preserve everything else, and output audio of the same length.",
            "full_enhancement": "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.",
            "channel_restoration": "Repair the telephone effect and restore natural, clear speech.",
        },
    }
    (args.out / "input_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.out / "rir_manifest.json").write_text(json.dumps(rir_records, indent=2) + "\n", encoding="utf-8")
    report = {
        "pair_count": len(manifest_rows),
        "speaker_count": len({row["speaker"] for row in manifest_rows}),
        "rir_candidates_total": len(all_rirs),
        "rir_candidates_in_used_band": len(eligible_rirs),
        "rir_pool_size": len(rir_records),
        "rt60_min": min(record["estimated_rt60_seconds"] for record in rir_records),
        "rt60_max": max(record["estimated_rt60_seconds"] for record in rir_records),
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
