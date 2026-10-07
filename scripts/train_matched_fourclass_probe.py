#!/usr/bin/env python3
"""Four-way matched probe: Real vs TTS vs SE vs TSE.

Runs either the AuK streams or the specialist streams from the common
(pair, 7-stream, 1024-d) feature matrix, using the same speaker split.
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


LABELS = ["real", "tts", "se", "tse"]
STREAMS = {
    "auk": {"real": 0, "tts": 1, "se": 3, "tse": 5},
    "specialist": {"real": 0, "tts": 2, "se": 4, "tse": 6},
}


def speaker_split(speakers: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(speakers)))
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    out = {}
    for s in unique[: n - n_val - n_test]: out[str(s)] = "train"
    for s in unique[n - n_val - n_test : n - n_test]: out[str(s)] = "validation"
    for s in unique[n - n_test :]: out[str(s)] = "test"
    return out


def metrics(model, x, y, name):
    pred = model.predict(x)
    return {
        "split": name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-root", type=Path, required=True)
    ap.add_argument("--variant", choices=tuple(STREAMS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--max-iter", type=int, default=1500)
    args = ap.parse_args()

    features = np.load(args.embedding_root / "features.npy", mmap_mode="r")
    with (args.embedding_root / "manifest.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if features.shape[:2] != (len(rows), 7):
        raise ValueError(f"unexpected shape {features.shape}, rows={len(rows)}")

    split_spk = speaker_split([r["speaker"] for r in rows], args.seed)
    split_by_pair = {r["pair_id"]: split_spk[r["speaker"]] for r in rows}
    stream_map = STREAMS[args.variant]

    xs, ys, splits = [], [], []
    for i, row in enumerate(rows):
        for label in LABELS:
            xs.append(np.asarray(features[i, stream_map[label]], dtype=np.float32))
            ys.append(label)
            splits.append(split_by_pair[row["pair_id"]])
    x, y, split = np.stack(xs), np.asarray(ys), np.asarray(splits)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=args.max_iter, C=1.0, multi_class="multinomial", random_state=args.seed),
    )
    model.fit(x[split == "train"], y[split == "train"])
    result = {s: metrics(model, x[split == s], y[split == s], s) for s in ("train", "validation", "test")}

    args.out.mkdir(parents=True, exist_ok=True)
    split_rows = [{"pair_id": r["pair_id"], "speaker": r["speaker"], "split": split_by_pair[r["pair_id"]]} for r in rows]
    with (args.out / "split.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "speaker", "split"], delimiter="\t")
        w.writeheader(); w.writerows(split_rows)
    report = {
        "embedding_root": str(args.embedding_root.resolve()),
        "variant": args.variant,
        "stream_indices": stream_map,
        "feature_shape": list(features.shape),
        "classes": LABELS,
        "probe": "StandardScaler + multinomial LogisticRegression",
        "split_protocol": "speaker-disjoint",
        "seed": args.seed,
        "max_iter": args.max_iter,
        "speaker_counts": {s: len({r["speaker"] for r in split_rows if r["split"] == s}) for s in ("train", "validation", "test")},
        "pair_counts": {s: sum(r["split"] == s for r in split_rows) for s in ("train", "validation", "test")},
        "metrics": result,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
