#!/usr/bin/env python3
"""Package 15 raw-output listening samples from the specialist experiments."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


SELECTION_MANIFEST = Path("data/audition_15_auk_20261006/selection_manifest.tsv")
OUT_ROOT = Path("data/audition_15_specialist_20261006")
SAMPLE_COUNT = 15

SPECIALISTS = {
    "cosyvoice3": {
        "display_name": "CosyVoice3 TTS",
        "root": Path("/data/specialist_outputs/cosyvoice3"),
        "filename": "specialist_tts.wav",
        "zip": Path("data/audition_15_specialist_cosyvoice3_20261006.zip"),
    },
    "mossformer_gan_se": {
        "display_name": "MossFormerGAN-SE",
        "root": Path("/data/specialist_outputs/mossformer_gan_se"),
        "filename": "specialist_se.wav",
        "zip": Path("data/audition_15_specialist_mossformer_se_20261006.zip"),
    },
    "dae_tse": {
        "display_name": "DAE-TSE",
        "root": Path("/data/specialist_outputs/dae_tse"),
        "filename": "specialist_tse.wav",
        "zip": Path("data/audition_15_specialist_dae_tse_20261006.zip"),
    },
}


def read_selection() -> list[dict[str, str]]:
    with SELECTION_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != SAMPLE_COUNT or len({row["speaker"] for row in rows}) != SAMPLE_COUNT:
        raise RuntimeError("selection manifest must contain 15 distinct speakers")
    return rows


def read_status_maps() -> dict[str, dict[str, dict[str, str]]]:
    maps: dict[str, dict[str, dict[str, str]]] = {name: {} for name in SPECIALISTS}

    moss_manifest = Path("/data/specialist_outputs/mossformer_gan_se/manifest.jsonl")
    if moss_manifest.is_file():
        for line in moss_manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            maps["mossformer_gan_se"][row["pair_id"]] = row

    dae_manifest = Path("/data/specialist_outputs/dae_tse/manifest.tsv")
    if dae_manifest.is_file():
        with dae_manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                maps["dae_tse"][row["pair_id"]] = row

    return maps


def probe_audio(path: Path) -> dict[str, str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "codec": stream.get("codec_name", ""),
        "sample_rate_hz": str(stream.get("sample_rate", "")),
        "channels": str(stream.get("channels", "")),
        "duration_seconds": str(payload.get("format", {}).get("duration", "")),
    }


def zip_directory(source_dir: Path, zip_path: Path, archive_root: str) -> None:
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(archive_root) / path.relative_to(source_dir))


def write_task_package(
    task: str,
    config: dict[str, object],
    selected: list[dict[str, str]],
    status_map: dict[str, dict[str, str]],
) -> None:
    task_dir = OUT_ROOT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for index, selection in enumerate(selected, 1):
        pair_id = selection["pair_id"]
        source = Path(str(config["root"])) / pair_id / str(config["filename"])
        if not source.is_file():
            raise FileNotFoundError(source)
        output_name = f"sample_{index:02d}_{pair_id}.wav"
        shutil.copy2(source, task_dir / output_name)
        media = probe_audio(source)
        record = status_map.get(pair_id, {})
        manifest_rows.append(
            {
                "sample_index": str(index),
                "pair_id": pair_id,
                "speaker": selection["speaker"],
                "gender": selection["gender"],
                "text": selection["text"],
                "noise_category": selection["noise_category"],
                "output_file": output_name,
                "model_output_source": str(source),
                "codec": media["codec"],
                "sample_rate_hz": media["sample_rate_hz"],
                "channels": media["channels"],
                "duration_seconds": media["duration_seconds"],
                "generation_status": record.get("status", "filesystem_verified"),
                "generation_warning": record.get("warning", record.get("error_or_warning", "")),
            }
        )

    fields = list(manifest_rows[0].keys())
    with (task_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    rates = sorted({row["sample_rate_hz"] for row in manifest_rows})
    (task_dir / "README.md").write_text(
        f"""# {config['display_name']} audition sample

This directory contains 15 raw outputs from the completed specialist-model experiment, using the same 15 target speakers as the other audition packages.

- Model: `{config['display_name']}`
- Output source: `{config['root']}`
- Audio files: 15 mono WAV files
- Sample rate(s): {', '.join(rates)} Hz
- No resampling, normalization, trimming, or other audio processing was applied when packaging.

Use `manifest.tsv` to match each file to its pair, speaker, transcript, and generation status.
""",
        encoding="utf-8",
    )
    zip_directory(task_dir, config["zip"], f"audition_15_specialist_{task}_20261006")


def main() -> None:
    selected = read_selection()
    status_maps = read_status_maps()
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    shutil.copy2(SELECTION_MANIFEST, OUT_ROOT / "selection_manifest.tsv")

    (OUT_ROOT / "README.md").write_text(
        """# Specialist-model 15-sample listening package

This package contains raw outputs from the three completed specialist-model experiments:

- CosyVoice3 TTS
- MossFormerGAN-SE
- DAE-TSE

All three tasks use the same 15 pair IDs and 15 different target speakers. The files are copied from `/data/specialist_outputs`; the package preserves each model's original sample rate and PCM encoding.

These are the specialist outputs, not the AuK outputs from `data/matched_full_5369_snr0_batch4_generated`.
""",
        encoding="utf-8",
    )

    for task, config in SPECIALISTS.items():
        write_task_package(task, config, selected, status_maps[task])

    zip_directory(OUT_ROOT, Path("data/audition_15_specialist_tasks_20261006.zip"), "audition_15_specialist_tasks_20261006")

    for task, config in SPECIALISTS.items():
        count = len(list((OUT_ROOT / task).glob("*.wav")))
        if count != SAMPLE_COUNT:
            raise RuntimeError(f"{task} package contains {count} WAV files")
        print(f"{task}: {count} samples -> {config['zip']}")
    print("combined: data/audition_15_specialist_tasks_20261006.zip")


if __name__ == "__main__":
    main()
