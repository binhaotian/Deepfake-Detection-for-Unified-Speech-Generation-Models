#!/usr/bin/env python3
"""Five-class AuK probe: real, TTS, SE, TSE, and content editing."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ["real", "tts", "se", "tse", "content_edit"]
OLD_FILES = {"real": "anchor_embedding", "tts": "auk_tts_embedding", "se": "auk_se_embedding", "tse": "auk_tse_embedding"}


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def split_speakers(speakers: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(speakers)))
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * 0.15))
    n_val = max(1, round(len(unique) * 0.15))
    out = {}
    for s in unique[: len(unique) - n_val - n_test]:
        out[str(s)] = "train"
    for s in unique[len(unique) - n_val - n_test : len(unique) - n_test]:
        out[str(s)] = "validation"
    for s in unique[len(unique) - n_test :]:
        out[str(s)] = "test"
    return out


def metric(model, x, y, split_name):
    pred = model.predict(x)
    return {
        "split": split_name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "labels": LABELS,
        "classification_report": classification_report(y, pred, labels=LABELS, target_names=LABELS, output_dict=True, zero_division=0),
    }, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("wav2vec2", "hubert"), required=True)
    ap.add_argument("--old-embedding-root", type=Path, required=True)
    ap.add_argument("--content-embedding-root", type=Path, required=True)
    ap.add_argument("--source-manifest", type=Path, default=Path("data/matched_full_5369_snr0_batch4_generated/manifest.tsv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    old_rows = {r["pair_id"]: r for r in read_tsv(args.old_embedding_root / "manifest.tsv")}
    content_rows = {r["pair_id"]: r for r in read_tsv(args.content_embedding_root / "manifest.tsv")}
    source_rows = {r["pair_id"]: r for r in read_tsv(args.source_manifest)}
    pair_ids = sorted(set(old_rows) & set(content_rows) & set(source_rows), key=lambda x: int(x.split("_")[-1]))
    if not pair_ids:
        raise RuntimeError("No pair IDs overlap between old and content-edit embeddings")

    first_old = np.load(args.old_embedding_root / old_rows[pair_ids[0]][OLD_FILES["real"]])
    content_features = np.load(args.content_embedding_root / "content_edit_embeddings.npy", mmap_mode="r")
    content_index = {r["pair_id"]: i for i, r in enumerate(read_tsv(args.content_embedding_root / "manifest.tsv"))}
    dim = int(first_old.shape[0])
    x_rows, y_rows, split_rows = [], [], []
    speakers = [source_rows[p]["speaker"] for p in pair_ids]
    split_by_speaker = split_speakers(speakers, args.seed)
    pair_split = {p: split_by_speaker[source_rows[p]["speaker"]] for p in pair_ids}

    for pair_id in pair_ids:
        old = old_rows[pair_id]
        for label in LABELS[:-1]:
            value = np.load(args.old_embedding_root / old[OLD_FILES[label]]).astype(np.float32)
            if value.shape != (dim,):
                raise ValueError(f"bad shape for {pair_id}/{label}: {value.shape}")
            x_rows.append(value); y_rows.append(label); split_rows.append(pair_split[pair_id])
        value = np.asarray(content_features[content_index[pair_id]], dtype=np.float32)
        if value.shape != (dim,):
            raise ValueError(f"bad shape for {pair_id}/content_edit: {value.shape}")
        x_rows.append(value); y_rows.append("content_edit"); split_rows.append(pair_split[pair_id])

    x = np.stack(x_rows)
    y = np.asarray(y_rows)
    split = np.asarray(split_rows)
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite feature value")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=args.max_iter, C=1.0, multi_class="multinomial", random_state=args.seed),
    )
    model.fit(x[split == "train"], y[split == "train"])
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for split_name in ("train", "validation", "test"):
        result, pred = metric(model, x[split == split_name], y[split == split_name], split_name)
        results[split_name] = result

    with (args.out / "split.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "speaker", "split"], delimiter="\t")
        w.writeheader()
        for p in pair_ids:
            w.writerow({"pair_id": p, "speaker": source_rows[p]["speaker"], "split": pair_split[p]})

    report = {
        "encoder": args.encoder,
        "old_embedding_root": str(args.old_embedding_root.resolve()),
        "content_embedding_root": str(args.content_embedding_root.resolve()),
        "classes": LABELS,
        "pair_count": len(pair_ids),
        "sample_count": int(len(y)),
        "embedding_dim": dim,
        "probe": "StandardScaler + multinomial LogisticRegression",
        "split_protocol": "speaker-disjoint",
        "seed": args.seed,
        "max_iter": args.max_iter,
        "speaker_counts": {s: len({source_rows[p]['speaker'] for p in pair_ids if pair_split[p] == s}) for s in ("train", "validation", "test")},
        "pair_counts": {s: sum(pair_split[p] == s for p in pair_ids) for s in ("train", "validation", "test")},
        "class_counts": {label: int(np.sum(y == label)) for label in LABELS},
        "metrics": results,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
