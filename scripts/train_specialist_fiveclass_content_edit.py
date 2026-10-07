#!/usr/bin/env python3
"""Specialist five-class probe with CosyEdit Content Editing."""
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


LABELS = ["real", "tts", "se", "tse", "content_edit"]


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def split_speakers(speakers, seed):
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(speakers)))
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * 0.15))
    n_val = max(1, round(len(unique) * 0.15))
    result = {}
    for s in unique[: len(unique) - n_val - n_test]: result[str(s)] = "train"
    for s in unique[len(unique) - n_val - n_test : len(unique) - n_test]: result[str(s)] = "validation"
    for s in unique[len(unique) - n_test :]: result[str(s)] = "test"
    return result


def evaluate(model, x, y, split):
    pred = model.predict(x)
    return {
        "split": split,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("wav2vec2", "hubert"), required=True)
    ap.add_argument("--old-root", type=Path, required=True)
    ap.add_argument("--content-root", type=Path, required=True)
    ap.add_argument("--source-manifest", type=Path, default=Path("data/matched_full_5369_snr0_batch4_generated/manifest.tsv"))
    ap.add_argument("--content-manifest", type=Path, default=Path("data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--max-iter", type=int, default=5000)
    args = ap.parse_args()

    old_features = np.load(args.old_root / "features.npy", mmap_mode="r")
    old_rows = read(args.old_root / "manifest.tsv")
    old_idx = {r["pair_id"]: i for i, r in enumerate(old_rows)}
    source = {r["pair_id"]: r for r in read(args.source_manifest)}
    content_rows = [r for r in read(args.content_manifest) if r.get("status") == "accepted"]
    content_features = np.load(args.content_root / "features.npy", mmap_mode="r")
    content_emb_rows = read(args.content_root / "manifest.tsv")
    content_idx = {r["pair_id"]: i for i, r in enumerate(content_emb_rows)}
    pair_ids = [r["pair_id"] for r in content_rows if r["pair_id"] in old_idx and r["pair_id"] in content_idx]
    if len(pair_ids) != len(content_rows):
        raise RuntimeError(f"pair intersection {len(pair_ids)} != accepted {len(content_rows)}")
    dim = int(old_features.shape[-1])
    speakers = [source[p]["speaker"] for p in pair_ids]
    split_by_speaker = split_speakers(speakers, args.seed)
    pair_split = {p: split_by_speaker[source[p]["speaker"]] for p in pair_ids}

    xs, ys, splits = [], [], []
    for p in pair_ids:
        i = old_idx[p]
        # Existing specialist feature order: real, AuK-TTS, specialist-TTS,
        # AuK-SE, specialist-SE, AuK-TSE, specialist-TSE.
        for label, stream in (("real", 0), ("tts", 2), ("se", 4), ("tse", 6)):
            xs.append(np.asarray(old_features[i, stream], dtype=np.float32)); ys.append(label); splits.append(pair_split[p])
        xs.append(np.asarray(content_features[content_idx[p], 1], dtype=np.float32)); ys.append("content_edit"); splits.append(pair_split[p])
    x, y, split = np.stack(xs), np.asarray(ys), np.asarray(splits)
    if x.shape[1] != dim or not np.isfinite(x).all():
        raise RuntimeError(f"invalid features shape={x.shape}")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=args.max_iter, C=1.0, multi_class="multinomial", random_state=args.seed))
    model.fit(x[split == "train"], y[split == "train"])
    results = {s: evaluate(model, x[split == s], y[split == s], s) for s in ("train", "validation", "test")}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "split.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "speaker", "split"], delimiter="\t"); w.writeheader()
        for p in pair_ids: w.writerow({"pair_id": p, "speaker": source[p]["speaker"], "split": pair_split[p]})
    report = {
        "encoder": args.encoder,
        "old_embedding_root": str(args.old_root.resolve()),
        "content_embedding_root": str(args.content_root.resolve()),
        "classes": LABELS,
        "stream_mapping": {"real": 0, "tts": 2, "se": 4, "tse": 6, "content_edit": "content_features[:,1]"},
        "pair_count": len(pair_ids), "sample_count": len(y), "embedding_dim": dim,
        "probe": "StandardScaler + multinomial LogisticRegression",
        "split_protocol": "speaker-disjoint", "seed": args.seed, "max_iter": args.max_iter,
        "speaker_counts": {s: len({source[p]['speaker'] for p in pair_ids if pair_split[p] == s}) for s in ("train", "validation", "test")},
        "pair_counts": {s: sum(pair_split[p] == s for p in pair_ids) for s in ("train", "validation", "test")},
        "class_counts": {label: int(np.sum(y == label)) for label in LABELS}, "metrics": results,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
