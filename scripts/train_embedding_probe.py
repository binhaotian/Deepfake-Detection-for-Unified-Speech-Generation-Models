#!/usr/bin/env python3
"""Train/evaluate frozen-embedding linear probes for AuK task identity."""
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


TASKS = ("real", "tts", "se", "tse")
EMB_COLS = {"real": "anchor_embedding", "tts": "auk_tts_embedding", "se": "auk_se_embedding", "tse": "auk_tse_embedding"}
NEW_EMB_COLS = {"real": "real_embedding", "tts": "tts_embedding", "se": "se_embedding", "tse": "tse_embedding"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def split_speakers(speakers: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(speakers)))
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    result = {}
    for s in unique[: n - n_val - n_test]: result[s] = "train"
    for s in unique[n - n_val - n_test : n - n_test]: result[s] = "validation"
    for s in unique[n - n_test :]: result[s] = "test"
    return result


def split_pairs_by_speaker(rows: list[dict[str, str]], data_rows: dict[str, dict[str, str]], seed: int) -> dict[str, str]:
    """Within-speaker 80/20 split; useful as a speaker-overlap comparison."""
    rng = np.random.default_rng(seed)
    by_speaker: dict[str, list[str]] = {}
    for row in rows:
        pair = row["pair_id"]
        by_speaker.setdefault(data_rows[pair]["speaker"], []).append(pair)
    result = {}
    for speaker, pairs in by_speaker.items():
        pairs = list(pairs)
        rng.shuffle(pairs)
        n_test = max(1, round(len(pairs) * 0.20))
        for pair in pairs[:-n_test]: result[pair] = "train"
        for pair in pairs[-n_test:]: result[pair] = "test"
    return result


def load_embedding(root: Path, name: str, expected_dim: int | None = None) -> np.ndarray:
    value = np.load(root / name)
    if value.ndim != 1 or (expected_dim is not None and value.shape[0] != expected_dim):
        raise ValueError(f"unexpected embedding shape for {name}: {value.shape}")
    return value.astype(np.float32)


def evaluate(model, x, y, split_name: str, labels: list[str]) -> dict:
    pred = model.predict(x)
    return {
        "split": split_name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "labels": labels,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-root", type=Path, default=Path("data/embeddings/wav2vec2-large-full"))
    ap.add_argument("--data-manifest", type=Path, default=Path("data/matched_full_5369_snr0_batch4_generated/manifest.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/probe_results/wav2vec2-large"))
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--split-protocol", choices=("speaker-disjoint", "speaker-overlap"), default="speaker-disjoint")
    args = ap.parse_args()

    emb_rows = read_rows(args.embedding_root / "manifest.tsv")
    data_rows = {r["pair_id"]: r for r in read_rows(args.data_manifest)}
    available = set(emb_rows[0])
    emb_cols = EMB_COLS if set(EMB_COLS.values()).issubset(available) else NEW_EMB_COLS
    first_name = emb_rows[0][emb_cols["real"]]
    embedding_dim = int(np.load(args.embedding_root / first_name).shape[0])
    speakers = [data_rows[r["pair_id"]]["speaker"] for r in emb_rows]
    if args.split_protocol == "speaker-disjoint":
        split_map = split_speakers(speakers, args.seed)
        split_by_pair = {r["pair_id"]: split_map[data_rows[r["pair_id"]]["speaker"]] for r in emb_rows}
    else:
        split_by_pair = split_pairs_by_speaker(emb_rows, data_rows, args.seed)
        split_map = {s: "overlap" for s in set(speakers)}
    args.out.mkdir(parents=True, exist_ok=True)

    split_rows = []
    for row in emb_rows:
        pair = row["pair_id"]
        speaker = data_rows[pair]["speaker"]
        split_rows.append({"pair_id": pair, "speaker": speaker, "split": split_by_pair[pair]})
    with (args.out / "split.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "speaker", "split"], delimiter="\t")
        w.writeheader(); w.writerows(split_rows)

    x_by_task, y_by_task, split_by_task = [], [], []
    for row in emb_rows:
        data = data_rows[row["pair_id"]]
        for task in TASKS:
            x_by_task.append(load_embedding(args.embedding_root, row[emb_cols[task]], embedding_dim))
            y_by_task.append(task)
            split_by_task.append(split_by_pair[row["pair_id"]])
    x = np.stack(x_by_task)
    y = np.array(y_by_task)
    split = np.array(split_by_task)
    labels = list(TASKS)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, C=1.0, multi_class="multinomial", random_state=args.seed))
    model.fit(x[split == "train"], y[split == "train"])
    results = {}
    for name in ("train", "validation", "test"):
        if np.any(split == name):
            results[name] = evaluate(model, x[split == name], y[split == name], name, labels)
    report = {
        "embedding_root": str(args.embedding_root.resolve()),
        "embedding_dim": embedding_dim,
        "probe": "StandardScaler + multinomial LogisticRegression",
        "classes": labels,
        "split_protocol": args.split_protocol,
        "seed": args.seed,
        "speaker_counts": {name: len({r["speaker"] for r in split_rows if r["split"] == name}) for name in ("train", "validation", "test") if any(r["split"] == name for r in split_rows)},
        "pair_counts": {name: sum(r["split"] == name for r in split_rows) for name in ("train", "validation", "test") if any(r["split"] == name for r in split_rows)},
        "metrics": results,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
