#!/usr/bin/env python3
"""Package a broad 15-pair listening sample for AuK content editing."""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PAIR_MANIFEST = Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv")
TEXT_MANIFEST = Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_text_manifest.tsv")
SOURCE_ROOT = Path("/root/AuK/data/matched_full_5369_snr0_batch4_generated")
GENERATED_ROOT = Path("/data/AuK_content_edit_full_20261001")
OUT_ROOT = Path("/data/AuK_content_edit_review_sample15_20261006")
ZIP_PATH = Path("/data/AuK_content_edit_review_sample15_20261006.zip")


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pair_num(row):
    return int(row["pair_id"].split("_")[-1])


def delta_bucket(row):
    try:
        d = int(row.get("syllable_delta_est", "0"))
    except ValueError:
        d = 0
    return "shorter" if d < 0 else "longer" if d > 0 else "same_syllable_estimate"


def choose_representative(rows):
    # Prefer a distinct speaker; then use the lowest pair id for reproducibility.
    return sorted(rows, key=lambda r: (r.get("speaker", ""), pair_num(r)))[0]


def main():
    pairs = [r for r in read_tsv(PAIR_MANIFEST) if r.get("status") == "accepted"]
    texts = [r for r in read_tsv(TEXT_MANIFEST) if r.get("status") == "accepted"]
    pair_by_group = defaultdict(list)
    for r in pairs:
        if (GENERATED_ROOT / r["pair_id"] / "auk_content_edit.wav").is_file():
            pair_by_group[r["text_group_id"]].append(r)
    candidates = []
    for t in texts:
        group = pair_by_group.get(t["text_group_id"], [])
        if not group:
            continue
        r = choose_representative(group)
        merged = dict(r)
        # Text-level estimates/review fields are useful in the listening sheet.
        for k, v in t.items():
            merged[f"text_{k}"] = v
        merged["delta_bucket"] = delta_bucket(t)
        candidates.append(merged)

    by_pos = defaultdict(list)
    for r in candidates:
        by_pos[r.get("part_of_speech", "")].append(r)
    selected = []
    used_groups = set()
    used_speakers = set()
    covered_routes = set()
    covered_deltas = set()

    # Cover every POS represented in the accepted text set. There are 13 POS
    # categories; this deliberately favors breadth over frequency.
    for pos in sorted(by_pos, key=lambda p: (len(by_pos[p]), p)):
        options = [r for r in by_pos[pos] if r["text_group_id"] not in used_groups]
        if not options:
            continue
        options.sort(key=lambda r: (
            r["route"] in covered_routes,
            r["delta_bucket"] in covered_deltas,
            r.get("speaker", "") in used_speakers,
            pair_num(r),
        ))
        r = options[0]
        selected.append(r)
        used_groups.add(r["text_group_id"])
        used_speakers.add(r.get("speaker", ""))
        covered_routes.add(r["route"])
        covered_deltas.add(r["delta_bucket"])

    # Add two common-category examples to bring the sample to 15 and provide
    # stronger coverage of both routes with ordinary utterances.
    remaining = [r for r in candidates if r["text_group_id"] not in used_groups]
    remaining.sort(key=lambda r: (
        r["route"] in covered_routes,
        r["delta_bucket"] in covered_deltas,
        r.get("speaker", "") in used_speakers,
        0 if r.get("part_of_speech") in {"adjective", "noun", "verb", "adverb"} else 1,
        pair_num(r),
    ))
    while len(selected) < 15 and remaining:
        r = remaining.pop(0)
        selected.append(r)
        used_groups.add(r["text_group_id"])
        used_speakers.add(r.get("speaker", ""))
        covered_routes.add(r["route"])
        covered_deltas.add(r["delta_bucket"])

    if len(selected) != 15:
        raise RuntimeError(f"selected {len(selected)} instead of 15")
    selected.sort(key=pair_num)

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    ZIP_PATH.unlink(missing_ok=True)

    # Copy audio and per-pair human-readable details.
    for i, r in enumerate(selected, 1):
        pair_out = OUT_ROOT / "pairs" / f"sample_{i:02d}_{r['pair_id']}"
        pair_out.mkdir(parents=True)
        anchor = SOURCE_ROOT / r["source_pair_dir"] / "anchor.wav"
        edited = GENERATED_ROOT / r["pair_id"] / "auk_content_edit.wav"
        shutil.copy2(anchor, pair_out / "anchor.wav")
        shutil.copy2(edited, pair_out / "auk_content_edit.wav")
        (pair_out / "instruction.txt").write_text(r["auk_instruction"] + "\n", encoding="utf-8")
        detail = {
            "sample_index": i,
            "pair_id": r["pair_id"],
            "text_group_id": r["text_group_id"],
            "speaker": r["speaker"],
            "gender": r["gender"],
            "anchor_id": r["anchor_id"],
            "anchor_vctk_id": r["anchor_vctk_id"],
            "source_text": r["source_text"],
            "old_word": r["old_word"],
            "new_word": r["new_word"],
            "edited_text": r["edited_text"],
            "route": r["route"],
            "part_of_speech": r["part_of_speech"],
            "confidence": r["confidence"],
            "auk_instruction": r["auk_instruction"],
            "anchor_duration_seconds": r["anchor_duration_seconds"],
            "old_syllables_est": r.get("text_old_syllables_est", ""),
            "new_syllables_est": r.get("text_new_syllables_est", ""),
            "syllable_delta_est": r.get("text_syllable_delta_est", ""),
            "delta_bucket": r["delta_bucket"],
            "audio": {"anchor": "anchor.wav", "content_edit": "auk_content_edit.wav"},
        }
        (pair_out / "details.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "sample_index", "pair_id", "text_group_id", "speaker", "gender", "anchor_id", "anchor_vctk_id",
        "source_text", "old_word", "new_word", "edited_text", "route", "part_of_speech", "confidence",
        "auk_instruction", "anchor_duration_seconds", "old_syllables_est", "new_syllables_est",
        "syllable_delta_est", "delta_bucket",
    ]
    with (OUT_ROOT / "listening_manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i, r in enumerate(selected, 1):
            out = {k: r.get(k, "") for k in fields}
            out["sample_index"] = i
            out["old_syllables_est"] = r.get("text_old_syllables_est", "")
            out["new_syllables_est"] = r.get("text_new_syllables_est", "")
            out["syllable_delta_est"] = r.get("text_syllable_delta_est", "")
            writer.writerow(out)

    # Include text-level review evidence for these groups.
    group_ids = {r["text_group_id"] for r in selected}
    text_rows = [r for r in texts if r["text_group_id"] in group_ids]
    with (OUT_ROOT / "selected_text_review.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=texts[0].keys(), delimiter="\t")
        writer.writeheader(); writer.writerows(text_rows)

    route_counts = Counter(r["route"] for r in selected)
    pos_counts = Counter(r["part_of_speech"] for r in selected)
    delta_counts = Counter(r["delta_bucket"] for r in selected)
    readme = f"""# AuK Content Editing — 15-pair listening sample

This is a broad manual-listening sample from the accepted full Content Editing set generated on 2026-10-01.

## How to listen

Each directory under `pairs/` contains:

- `anchor.wav`: original bona fide speech used as the source anchor;
- `auk_content_edit.wav`: AuK output after one word was edited;
- `instruction.txt`: exact AuK instruction;
- `details.json`: all metadata for this sample.

The intended comparison is to listen to `anchor.wav` first, then `auk_content_edit.wav`, while reading `listening_manifest.tsv`.

## Selection logic

The sample intentionally covers every part-of-speech category present in the accepted text set, then adds two ordinary high-frequency examples. Selection favors distinct speakers, both editing routes, and different estimated syllable changes. It is not a random estimate of dataset-level performance.

- Number of pairs: {len(selected)}
- Routes: {dict(route_counts)}
- Part-of-speech categories: {dict(pos_counts)}
- Estimated syllable-change buckets: {dict(delta_counts)}
- Unique speakers: {len({r['speaker'] for r in selected})}

Important: `old_syllables_est` and `new_syllables_est` are text-selection estimates, not forced acoustic durations. The model was instructed only to replace the specified word; it was not given the original waveform as a reconstruction target.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=OUT_ROOT.parent, base_dir=OUT_ROOT.name)
    print(f"selected={len(selected)}")
    print(f"routes={dict(route_counts)}")
    print(f"pos={dict(pos_counts)}")
    print(f"delta={dict(delta_counts)}")
    print(f"zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
