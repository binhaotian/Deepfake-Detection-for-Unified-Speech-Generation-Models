#!/usr/bin/env python3
"""Construct all eligible ASVspoof LA eval target-speaker matched pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torchaudio


SAMPLE_RATE = 24_000
SEED = 20260923
NOISE_CATEGORIES = ("engine", "rain", "sea_waves", "thunderstorm", "water_drops", "wind")


def load_meta(path: Path) -> dict[str, dict[str, str]]:
    with tarfile.open(path) as archive:
        stream = archive.extractfile("ASVspoof2019_LA_VCTK_MetaInfo.tsv")
        if stream is None:
            raise RuntimeError("ASVspoof LA/VCTK mapping TSV not found in archive")
        return {
            row["ASVspoof_ID"]: row
            for row in csv.DictReader((line.decode("utf-8") for line in stream), delimiter="\t")
        }


def load_cm_protocol(path: Path) -> dict[str, str]:
    labels = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields:
            labels[fields[1]] = fields[-1]
    return labels


def load_asv_protocols(directory: Path) -> dict[str, list[str]]:
    enrollment = {}
    for path in sorted(directory.glob("*.eval.*.trn.txt")):
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                enrollment[fields[0]] = fields[1].split(",")
    return enrollment


def load_speakers(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    gender = {}
    details = {}
    for line in path.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 3:
            gender[fields[0]] = fields[2]
            details[fields[0]] = line.strip()
    return gender, details


def rms(audio: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(audio.square()) + 1e-12))


def read_mono(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = torchaudio.load(str(path))
    return audio.mean(dim=0, keepdim=True).to(torch.float32), sample_rate


def to_target_rate(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)
    return audio


def fit_length(audio: torch.Tensor, samples: int) -> tuple[torch.Tensor, int]:
    if audio.shape[-1] == 0:
        raise ValueError("empty source audio")
    repeats = max(1, math.ceil(samples / audio.shape[-1]))
    if repeats > 1:
        audio = audio.repeat(1, repeats)
    return audio[..., :samples], repeats


def write_wav(path: Path, audio: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.to(torch.float32), SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)


def select_median_reference(ids: list[str], audio_root: Path) -> str:
    candidates = []
    for audio_id in ids:
        path = audio_root / f"{audio_id}.flac"
        if path.exists():
            info = torchaudio.info(str(path))
            candidates.append((info.num_frames / info.sample_rate, audio_id))
    if not candidates:
        raise RuntimeError("target has no readable enrollment audio")
    candidates.sort()
    return candidates[len(candidates) // 2][1]


def choose_duration_matched_enrollment(ids: list[str], audio_root: Path, target_seconds: float) -> tuple[str, int]:
    choices = []
    for audio_id in ids:
        path = audio_root / f"{audio_id}.flac"
        if not path.exists():
            continue
        info = torchaudio.info(str(path))
        duration = info.num_frames / info.sample_rate
        choices.append((duration, audio_id))
    if not choices:
        raise RuntimeError("interferer target has no readable enrollment audio")
    # Prefer no looping. If no clip is long enough, use the longest clip and record repeats.
    long_enough = [choice for choice in choices if choice[0] >= target_seconds]
    if long_enough:
        duration, audio_id = min(long_enough, key=lambda choice: (choice[0] - target_seconds, choice[1]))
    else:
        duration, audio_id = max(choices, key=lambda choice: (choice[0], choice[1]))
    repeats = max(1, math.ceil(target_seconds / duration))
    return audio_id, repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--la", type=Path, default=Path("/root/LA"))
    parser.add_argument("--vctk", type=Path, default=Path("data/VCTK-Corpus-0.92"))
    parser.add_argument("--mapping", type=Path, default=Path("ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz"))
    parser.add_argument("--noise-dir", type=Path, default=Path("data/noise_esc50_full"))
    parser.add_argument("--out", type=Path, default=Path("data/matched_full_5369_snr0_batch4"))
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--sir-db", type=float, default=0.0)
    args = parser.parse_args()

    meta = load_meta(args.mapping)
    labels = load_cm_protocol(args.la / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    enrollment = load_asv_protocols(args.la / "ASVspoof2019_LA_asv_protocols")
    gender, speaker_details = load_speakers(args.vctk / "speaker-info.txt")
    la_audio_dir = args.la / "ASVspoof2019_LA_eval/flac"

    target_speaker = {}
    target_enrollment = {}
    for target_id, audio_ids in enrollment.items():
        speakers = {
            meta[audio_id]["VCTK_ID"].split("_")[0]
            for audio_id in audio_ids
            if audio_id in meta and meta[audio_id]["VCTK_ID"] != "-"
        }
        if len(speakers) == 1:
            speaker = next(iter(speakers))
            target_speaker[target_id] = speaker
            target_enrollment[speaker] = audio_ids

    bona_fide = []
    excluded = []
    seen_vctk = set()
    for audio_id, label in labels.items():
        if label != "bonafide":
            continue
        row = meta.get(audio_id)
        if row is None or row["VCTK_ID"] == "-":
            excluded.append({"anchor_id": audio_id, "reason": "no VCTK utterance mapping"})
            continue
        vctk_id = row["VCTK_ID"]
        speaker = vctk_id.split("_")[0]
        if speaker not in target_enrollment:
            excluded.append({"anchor_id": audio_id, "vctk_id": vctk_id, "reason": "speaker has no ASV target enrollment"})
            continue
        text_path = args.vctk / "txt" / speaker / f"{vctk_id}.txt"
        anchor_path = la_audio_dir / f"{audio_id}.flac"
        if not text_path.exists():
            excluded.append({"anchor_id": audio_id, "vctk_id": vctk_id, "reason": "official VCTK transcript missing"})
            continue
        if not anchor_path.exists():
            excluded.append({"anchor_id": audio_id, "vctk_id": vctk_id, "reason": "ASVspoof bona fide audio missing"})
            continue
        if vctk_id in seen_vctk:
            raise RuntimeError(f"duplicate VCTK anchor mapping: {vctk_id}")
        seen_vctk.add(vctk_id)
        info = torchaudio.info(str(anchor_path))
        bona_fide.append({
            "anchor_id": audio_id,
            "vctk_id": vctk_id,
            "speaker": speaker,
            "target_id": next(target for target, spk in target_speaker.items() if spk == speaker),
            "gender": gender.get(speaker, "U"),
            "text": " ".join(text_path.read_text(encoding="utf-8").split()),
            "duration": info.num_frames / info.sample_rate,
            "anchor_source": anchor_path,
            "tts_reference_id": "",
            "interferer_target_id": "",
            "interferer_speaker": "",
            "interferer_id": "",
        })
    bona_fide.sort(key=lambda row: (row["speaker"], row["anchor_id"]))

    # Fix one same-speaker, independent enrollment reference per target speaker.
    references = {
        speaker: select_median_reference(ids, la_audio_dir)
        for speaker, ids in target_enrollment.items()
    }

    # Fix three same-gender interferer speakers for each target speaker.
    target_speakers = sorted(target_enrollment)
    interferer_pool = {}
    for speaker in target_speakers:
        compatible = [other for other in target_speakers if other != speaker and gender.get(other) == gender.get(speaker)]
        if len(compatible) < 3:
            compatible = [other for other in target_speakers if other != speaker]
        shift = int.from_bytes(hashlib.sha256(speaker.encode()).digest()[:4], "big") % len(compatible)
        rotated = compatible[shift:] + compatible[:shift]
        interferer_pool[speaker] = rotated[:3]

    noise_rows = list(csv.DictReader((args.noise_dir / "esc50_subset.csv").open(encoding="utf-8")))
    noise_by_category = defaultdict(list)
    for row in noise_rows:
        noise_by_category[row["category"]].append(row)
    for category in NOISE_CATEGORIES:
        noise_by_category[category].sort(key=lambda row: row["filename"])
        if len(noise_by_category[category]) != 40:
            raise RuntimeError(f"expected 40 noise clips for {category}, found {len(noise_by_category[category])}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    speaker_anchor_index = Counter()
    noise_counts = Counter()
    for index, row in enumerate(bona_fide):
        speaker = row["speaker"]
        within_speaker = speaker_anchor_index[speaker]
        speaker_anchor_index[speaker] += 1
        interferer_speaker = interferer_pool[speaker][within_speaker % 3]
        interferer_target_id = next(target for target, spk in target_speaker.items() if spk == interferer_speaker)
        target_duration = row["duration"]
        interferer_id, interferer_repeats = choose_duration_matched_enrollment(
            target_enrollment[interferer_speaker], la_audio_dir, target_duration
        )
        category = NOISE_CATEGORIES[index % len(NOISE_CATEGORIES)]
        noise_index = (index // len(NOISE_CATEGORIES)) % len(noise_by_category[category])
        noise_row = noise_by_category[category][noise_index]
        noise_counts[noise_row["filename"]] += 1
        tts_reference_id = references[speaker]

        pair_id = f"pair_{index + 1:05d}"
        pair_dir = args.out / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        anchor, anchor_sr = read_mono(row["anchor_source"])
        anchor = to_target_rate(anchor, anchor_sr)
        reference, ref_sr = read_mono(la_audio_dir / f"{tts_reference_id}.flac")
        reference = to_target_rate(reference, ref_sr)
        interferer, int_sr = read_mono(la_audio_dir / f"{interferer_id}.flac")
        interferer = to_target_rate(interferer, int_sr)
        noise, noise_sr = read_mono(args.noise_dir / noise_row["filename"])
        noise = to_target_rate(noise, noise_sr)

        sample_count = anchor.shape[-1]
        noise, noise_repeats = fit_length(noise, sample_count)
        interferer, actual_interferer_repeats = fit_length(interferer, sample_count)
        noise *= rms(anchor) / (10 ** (args.snr_db / 20) * max(rms(noise), 1e-8))
        interferer *= rms(anchor) / (10 ** (args.sir_db / 20) * max(rms(interferer), 1e-8))
        se_input = anchor + noise
        tse_input = anchor + interferer

        # A shared scale preserves both exact ratios and within-pair source identity.
        peak = max(float(audio.abs().max()) for audio in (anchor, reference, noise, interferer, se_input, tse_input))
        if peak > 0.98:
            scale = 0.98 / peak
            anchor, reference, noise, interferer, se_input, tse_input = [
                audio * scale for audio in (anchor, reference, noise, interferer, se_input, tse_input)
            ]

        write_wav(pair_dir / "anchor.wav", anchor)
        write_wav(pair_dir / "tts_reference.wav", reference)
        write_wav(pair_dir / "se_noise.wav", noise)
        write_wav(pair_dir / "se_input.wav", se_input)
        write_wav(pair_dir / "tse_interferer.wav", interferer)
        write_wav(pair_dir / "tse_input.wav", tse_input)

        manifest_rows.append({
            "pair_id": pair_id,
            "anchor_id": row["anchor_id"],
            "anchor_vctk_id": row["vctk_id"],
            "target_id": row["target_id"],
            "speaker": speaker,
            "gender": row["gender"],
            "text": row["text"],
            "anchor_source_id": row["anchor_id"],
            "tts_reference_id": tts_reference_id,
            "interferer_target_id": interferer_target_id,
            "interferer_speaker": interferer_speaker,
            "interferer_id": interferer_id,
            "noise_id": noise_row["filename"],
            "noise_category": category,
            "snr_db": args.snr_db,
            "sir_db": args.sir_db,
            "measured_snr_db": 20 * math.log10(rms(anchor) / rms(noise)),
            "measured_sir_db": 20 * math.log10(rms(anchor) / rms(interferer)),
            "anchor_duration_seconds": target_duration,
            "noise_repeats": noise_repeats,
            "interferer_repeats": actual_interferer_repeats,
            "selected_interferer_nominal_repeats": interferer_repeats,
            "seed": SEED + index + 1,
            "tts_seed": SEED + index + 1,
            "se_seed": SEED + 1_000_000 + index + 1,
            "tse_seed": SEED + 2_000_000 + index + 1,
            "gender_metadata": speaker_details.get(speaker, ""),
            "noise_source": f"noise_source_metadata.csv::{noise_row['filename']}",
            "pair_dir": pair_id,
        })
        if (index + 1) % 250 == 0 or index + 1 == len(bona_fide):
            print(f"prepared {index + 1}/{len(bona_fide)} pairs", flush=True)

    if len(manifest_rows) != 5369:
        print(f"notice: generated {len(manifest_rows)} eligible anchors (expected approximately 5369)")
    fields = list(manifest_rows[0])
    with (args.out / "manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (args.out / "excluded_anchors.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("anchor_id", "vctk_id", "reason"), delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded)
    shutil_noise = args.out / "noise_source_metadata.csv"
    with shutil_noise.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=noise_rows[0].keys())
        writer.writeheader()
        writer.writerows(noise_rows)
    summary = {
        "pair_count": len(manifest_rows),
        "unique_target_speakers": len({row["speaker"] for row in manifest_rows}),
        "excluded_count": len(excluded),
        "exclusions": excluded,
        "noise_category_counts": dict(Counter(row["noise_category"] for row in manifest_rows)),
        "distinct_noise_clips_used": len(noise_counts),
        "max_noise_clip_reuse": max(noise_counts.values()),
        "snr_db": args.snr_db,
        "sir_db": args.sir_db,
        "sample_rate": SAMPLE_RATE,
        "seed": SEED,
    }
    (args.out / "preparation_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
