#!/usr/bin/env python3
"""Three-way task probe on specialist-model outputs only.

Classes are TTS (CosyVoice3), SE (MossFormerGAN), and TSE (DAE-TSE).
The encoder is frozen; only a standardized multinomial linear probe is fit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STREAMS = {"tts": 2, "se": 4, "tse": 6}
LABELS = ["tts", "se", "tse"]


def split_speakers(speakers: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(speakers)))
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    result = {}
    for speaker in unique[: n - n_val - n_test]:
        result[str(speaker)] = "train"
    for speaker in unique[n - n_val - n_test : n - n_test]:
        result[str(speaker)] = "validation"
    for speaker in unique[n - n_test :]:
        result[str(speaker)] = "test"
    return result


def evaluate(model, x, y, split_name: str) -> dict:
    pred = model.predict(x)
    return {
        "split": split_name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--max-iter", type=int, default=1500)
    args = ap.parse_args()

    features = np.load(args.embedding_root / "features.npy", mmap_mode="r")
    with (args.embedding_root / "manifest.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if features.shape[0] != len(rows) or features.shape[1] != 7:
        raise ValueError(f"unexpected feature shape {features.shape} for {len(rows)} rows")
    speakers = [r.get("speaker", "") for r in rows]
    split_by_speaker = split_speakers(speakers, args.seed)
    split_by_pair = {r["pair_id"]: split_by_speaker[r["speaker"]] for r in rows}

    x_parts, y_parts, split_parts = [], [], []
    for row_index, row in enumerate(rows):
        for label, stream_index in STREAMS.items():
            x_parts.append(np.asarray(features[row_index, stream_index], dtype=np.float32))
            y_parts.append(label)
            split_parts.append(split_by_pair[row["pair_id"]])
    x = np.stack(x_parts)
    y = np.asarray(y_parts)
    split = np.asarray(split_parts)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=args.max_iter,
            C=1.0,
            multi_class="multinomial",
            random_state=args.seed,
        ),
    )
    model.fit(x[split == "train"], y[split == "train"])
    metrics = {
        name: evaluate(model, x[split == name], y[split == name], name)
        for name in ("train", "validation", "test")
    }

    args.out.mkdir(parents=True, exist_ok=True)
    split_rows = [
        {"pair_id": row["pair_id"], "speaker": row["speaker"], "split": split_by_pair[row["pair_id"]]}
        for row in rows
    ]
    with (args.out / "split.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "speaker", "split"], delimiter="\t")
        writer.writeheader()
        writer.writerows(split_rows)
    report = {
        "embedding_root": str(args.embedding_root.resolve()),
        "feature_shape": list(features.shape),
        "streams": {"tts": "specialist_tts", "se": "specialist_se", "tse": "specialist_tse"},
        "classes": LABELS,
        "probe": "StandardScaler + multinomial LogisticRegression",
        "seed": args.seed,
        "max_iter": args.max_iter,
        "split_protocol": "speaker-disjoint",
        "speaker_counts": {name: len({r["speaker"] for r in split_rows if r["split"] == name}) for name in ("train", "validation", "test")},
        "pair_counts": {name: sum(r["split"] == name for r in split_rows) for name in ("train", "validation", "test")},
        "metrics": metrics,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
