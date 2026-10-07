#!/usr/bin/env python3
"""Create a deterministic 200-pair content-edit review package."""
from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 20261001
N = 200
TEXT_MANIFEST = Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_text_manifest.tsv")
PAIR_MANIFEST = Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv")
SOURCE_ROOT = Path("/root/AuK/data/matched_full_5369_snr0_batch4_generated")
GENERATED_ROOT = Path("/data/AuK_content_edit_full_20261001")
OUT_ROOT = Path("/data/AuK_content_edit_review_sample_200_20261001")
ZIP_PATH = Path("/data/AuK_content_edit_review_sample_200_20261001.zip")


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    pairs = [r for r in read_tsv(PAIR_MANIFEST) if r.get("status") == "accepted"]
    # Prefer pairs whose generated audio is already present, so the review archive
    # can be listened to immediately while the full generation continues elsewhere.
    available = [
        r for r in pairs
        if (GENERATED_ROOT / r["pair_id"] / "auk_content_edit.wav").is_file()
    ]
    if len(available) < N:
        raise RuntimeError(f"Only {len(available)} generated pairs are available; need {N}.")

    # Deterministic stratified sampling over route and POS, with a final fill step.
    strata = defaultdict(list)
    for row in available:
        strata[(row.get("route", ""), row.get("part_of_speech", ""))].append(row)
    rng = random.Random(SEED)
    for rows in strata.values():
        rng.shuffle(rows)
    ordered_keys = sorted(strata)
    chosen = []
    # Round-robin gives small strata a chance while preserving broad coverage.
    while len(chosen) < N and any(strata.values()):
        for key in ordered_keys:
            if strata[key] and len(chosen) < N:
                chosen.append(strata[key].pop())
        if len(chosen) >= N:
            break
    chosen.sort(key=lambda r: int(r["pair_id"].split("_")[-1]))

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    selected_ids = {r["pair_id"] for r in chosen}
    text_rows = [r for r in read_tsv(TEXT_MANIFEST) if r.get("status") == "accepted" and r.get("text_group_id") in {x["text_group_id"] for x in chosen}]

    # Write a compact review manifest with one row per selected pair.
    manifest_path = OUT_ROOT / "review_manifest.tsv"
    fields = list(chosen[0].keys()) + ["anchor_path", "generated_audio_path", "audio_available"]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in chosen:
            src = SOURCE_ROOT / row["source_pair_dir"] / "anchor.wav"
            gen = GENERATED_ROOT / row["pair_id"] / "auk_content_edit.wav"
            out = dict(row)
            out["anchor_path"] = "pairs/" + row["pair_id"] + "/anchor.wav"
            out["generated_audio_path"] = "pairs/" + row["pair_id"] + "/auk_content_edit.wav"
            out["audio_available"] = str(gen.is_file()).lower()
            writer.writerow(out)

            pair_out = OUT_ROOT / "pairs" / row["pair_id"]
            pair_out.mkdir(parents=True)
            if src.is_file():
                shutil.copy2(src, pair_out / "anchor.wav")
            shutil.copy2(gen, pair_out / "auk_content_edit.wav")
            (pair_out / "instruction.txt").write_text(row["auk_instruction"] + "\n", encoding="utf-8")

    # Include the source-level text decisions relevant to the selected pairs.
    text_ids = {r["text_group_id"] for r in chosen}
    with (OUT_ROOT / "selected_text_manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        all_rows = read_tsv(TEXT_MANIFEST)
        rows = [r for r in all_rows if r.get("text_group_id") in text_ids]
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    route_counts = defaultdict(int)
    pos_counts = defaultdict(int)
    for row in chosen:
        route_counts[row.get("route", "")] += 1
        pos_counts[row.get("part_of_speech", "")] += 1
    readme = f"""# Content-edit review sample (200 pairs)\n\n"
This archive is a deterministic, stratified sample of 200 accepted content-edit pairs from the full AuK content-edit candidate set.\n\n"
Each pair contains:\n- `anchor.wav`: original bona fide anchor\n- `auk_content_edit.wav`: AuK output after a single-word replacement\n- `instruction.txt`: exact generation instruction\n\n"
The TSV manifests contain the original sentence, replacement word, edited sentence, route, part of speech, speaker, and pair metadata.\n\n"
Sampling seed: `{SEED}`. All 200 selected pairs had generated audio available at packaging time.\n\n"
Routes: {dict(sorted(route_counts.items()))}\n\nPart-of-speech counts: {dict(sorted(pos_counts.items()))}\n"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=OUT_ROOT.parent, base_dir=OUT_ROOT.name)
    print(f"selected={len(chosen)}")
    print(f"output={OUT_ROOT}")
    print(f"zip={ZIP_PATH}")
    print(f"routes={dict(sorted(route_counts.items()))}")
    print(f"pos={dict(sorted(pos_counts.items()))}")


if __name__ == "__main__":
    main()
