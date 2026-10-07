#!/usr/bin/env python3
"""Package 15 comparable listening samples for AuK TTS, SE, and TSE."""
from __future__ import annotations

import csv
import random
import shutil
import zipfile
from pathlib import Path


SEED = 20261006
SAMPLE_COUNT = 15
SOURCE_ROOT = Path("data/matched_full_5369_snr0_batch4_generated")
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.tsv"
OUT_ROOT = Path("data/audition_15_auk_20261006")

TASKS = {
    "tts": "auk_tts.wav",
    "se": "auk_se.wav",
    "tse": "auk_tse.wav",
}


def read_manifest() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def pair_number(pair_id: str) -> int:
    return int(pair_id.rsplit("_", 1)[-1])


def select_samples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    available = [
        row
        for row in rows
        if all((SOURCE_ROOT / row["pair_id"] / filename).is_file() for filename in TASKS.values())
    ]
    by_speaker: dict[str, list[dict[str, str]]] = {}
    for row in available:
        by_speaker.setdefault(row["speaker"], []).append(row)

    if len(by_speaker) < SAMPLE_COUNT:
        raise RuntimeError(f"Only {len(by_speaker)} speakers are available; need {SAMPLE_COUNT}.")

    rng = random.Random(SEED)
    speakers = rng.sample(sorted(by_speaker), SAMPLE_COUNT)
    selected = [rng.choice(by_speaker[speaker]) for speaker in speakers]
    return sorted(selected, key=lambda row: pair_number(row["pair_id"]))


def write_manifest(path: Path, selected: list[dict[str, str]], task: str | None = None) -> None:
    fields = [
        "sample_index",
        "pair_id",
        "speaker",
        "gender",
        "text",
        "anchor_duration_seconds",
        "noise_category",
        "output_file",
        "source_pair",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            output_file = f"sample_{index:02d}_{row['pair_id']}_{task}.wav" if task else ""
            writer.writerow(
                {
                    "sample_index": index,
                    "pair_id": row["pair_id"],
                    "speaker": row["speaker"],
                    "gender": row["gender"],
                    "text": row["text"],
                    "anchor_duration_seconds": row["anchor_duration_seconds"],
                    "noise_category": row["noise_category"],
                    "output_file": output_file,
                    "source_pair": f"{SOURCE_ROOT}/{row['pair_id']}",
                }
            )


def zip_directory(source_dir: Path, zip_path: Path, archive_root: str) -> None:
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(archive_root) / path.relative_to(source_dir))


def main() -> None:
    rows = read_manifest()
    selected = select_samples(rows)
    if len(selected) != SAMPLE_COUNT:
        raise RuntimeError(f"Selected {len(selected)} samples instead of {SAMPLE_COUNT}.")

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    write_manifest(OUT_ROOT / "selection_manifest.tsv", selected)
    (OUT_ROOT / "README.md").write_text(
        f"""# AuK 15-sample listening package

This package contains the same 15 source pairs for AuK TTS, speech enhancement (SE), and target-speaker extraction (TSE). The samples were selected with a fixed seed and cover 15 different target speakers.

Each task directory contains exactly 15 generated 24 kHz mono WAV files and a TSV manifest. The WAV files are AuK Base outputs from `matched_full_5369_snr0_batch4_generated`. The completed specialist-model outputs are stored separately under `/data/specialist_outputs` and are packaged by `scripts/package_specialist_audition_15.py`.

The source pair directory in the manifest contains the corresponding anchor/input audio for direct comparison if needed.

Selection seed: `{SEED}`.
""",
        encoding="utf-8",
    )

    for task, source_filename in TASKS.items():
        task_dir = OUT_ROOT / task
        task_dir.mkdir()
        write_manifest(task_dir / "manifest.tsv", selected, task)
        for index, row in enumerate(selected, 1):
            source = SOURCE_ROOT / row["pair_id"] / source_filename
            target = task_dir / f"sample_{index:02d}_{row['pair_id']}_{task}.wav"
            shutil.copy2(source, target)
        (task_dir / "README.md").write_text(
            f"# AuK {task.upper()} audition\n\n"
            f"15 generated samples from the AuK `{task}` task. Listen in sample-number order to compare across the three task packages.\n",
            encoding="utf-8",
        )
        zip_directory(task_dir, Path(f"data/audition_15_auk_{task}_20261006.zip"), f"audition_15_auk_{task}_20261006")

    zip_directory(OUT_ROOT, Path("data/audition_15_auk_tasks_20261006.zip"), "audition_15_auk_tasks_20261006")

    for task in TASKS:
        count = len(list((OUT_ROOT / task).glob("*.wav")))
        if count != SAMPLE_COUNT:
            raise RuntimeError(f"{task} package contains {count} WAV files")
        print(f"{task}: {count} samples -> data/audition_15_auk_{task}_20261006.zip")
    print("combined: data/audition_15_auk_tasks_20261006.zip")
    print("pairs: " + ", ".join(row["pair_id"] for row in selected))


if __name__ == "__main__":
    main()
