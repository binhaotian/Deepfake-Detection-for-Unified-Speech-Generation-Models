#!/usr/bin/env python3
"""Train/evaluate a class-balanced linear probe on ASVspoof embeddings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ["bonafide", "spoof"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    rows = read_rows(root / f"{split}.tsv")
    x = np.stack([np.load(root / row["embedding"]) for row in rows]).astype(np.float32)
    y = np.array([row["label"] for row in rows])
    return x, y, rows


def eer(y_true: np.ndarray, score_bonafide: np.ndarray) -> float:
    y_binary = (y_true == "bonafide").astype(np.int32)
    fpr, tpr, _ = roc_curve(y_binary, score_bonafide)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def evaluate(model, x: np.ndarray, y: np.ndarray, split: str) -> dict:
    pred = model.predict(x)
    probabilities = model.predict_proba(x)
    bonafide_index = list(model.classes_).index("bonafide")
    score = probabilities[:, bonafide_index]
    return {
        "split": split,
        "n": int(len(y)),
        "class_counts": {label: int(np.sum(y == label)) for label in LABELS},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro")),
        "roc_auc_bonafide": float(roc_auc_score((y == "bonafide").astype(int), score)),
        "eer": eer(y, score),
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    x_train, y_train, train_rows = load_split(args.embedding_root, "train")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, class_weight="balanced", C=1.0, random_state=args.seed),
    )
    model.fit(x_train, y_train)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    split_counts = {}
    for split in ("train", "dev", "eval"):
        x, y, rows = load_split(args.embedding_root, split)
        results[split] = evaluate(model, x, y, split)
        split_counts[split] = {"total": len(rows), "targets": len({r["target"] for r in rows})}
    report = {
        "embedding_root": str(args.embedding_root.resolve()),
        "probe": "StandardScaler + class-balanced binary LogisticRegression",
        "train_split": "ASVspoof2019 LA train CM protocol",
        "evaluation_splits": ["dev", "eval"],
        "seed": args.seed,
        "split_counts": split_counts,
        "metrics": results,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
