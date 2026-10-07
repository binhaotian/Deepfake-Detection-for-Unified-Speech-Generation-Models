#!/usr/bin/env python3
"""Distribution and nuisance-factor analysis for matched-set embeddings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torchaudio
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

CLASSES = ("real", "tts", "se", "tse")
COLS = {"real": "anchor_embedding", "tts": "auk_tts_embedding", "se": "auk_se_embedding", "tse": "auk_tse_embedding"}
FILES = {"real": "anchor.wav", "tts": "auk_tts.wav", "se": "auk_se.wav", "tse": "auk_tse.wav"}


def rows(path: Path):
    with path.open() as f: return list(csv.DictReader(f, delimiter="\t"))


def audio_stats(path: Path):
    x, sr = torchaudio.load(str(path)); x = x.float()
    rms = float(torch_rms(x))
    return x.shape[-1] / sr, rms, float(x.abs().max())


def torch_rms(x):
    return (x.square().mean().sqrt()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/matched_full_5369_snr0_batch4_generated"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    erows = rows(args.embedding_root / "manifest.tsv"); drows = {r["pair_id"]: r for r in rows(args.data_root / "manifest.tsv")}
    if args.max_pairs: erows = erows[:args.max_pairs]
    x_by = {c: [] for c in CLASSES}; nuisance = {c: [] for c in CLASSES}
    for r in erows:
        pair = r["pair_id"]
        for c in CLASSES:
            x_by[c].append(np.load(args.embedding_root / r[COLS[c]]))
            nuisance[c].append(audio_stats(args.data_root / pair / FILES[c]))
    x_by = {c: np.stack(v).astype(np.float32) for c, v in x_by.items()}
    all_x = np.concatenate([x_by[c] for c in CLASSES])
    pca = PCA(n_components=2, random_state=0).fit(all_x)
    coords = {c: pca.transform(x_by[c]).tolist() for c in CLASSES}
    centroids = {c: x_by[c].mean(0) for c in CLASSES}
    centroid_dist = {a: {b: float(np.linalg.norm(centroids[a]-centroids[b])) for b in CLASSES} for a in CLASSES}
    within = {c: float(pairwise_distances(x_by[c][: min(1000, len(x_by[c]))]).mean()) for c in CLASSES}
    nuisance_summary = {}
    for c in CLASSES:
        arr = np.asarray(nuisance[c])
        nuisance_summary[c] = {"duration_mean": float(arr[:,0].mean()), "duration_std": float(arr[:,0].std()), "rms_mean": float(arr[:,1].mean()), "rms_std": float(arr[:,1].std()), "peak_mean": float(arr[:,2].mean())}
    result = {"embedding_root": str(args.embedding_root.resolve()), "pair_count": len(erows), "classes": list(CLASSES), "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "centroid_distance": centroid_dist, "within_class_mean_distance_first1000": within, "audio_nuisance": nuisance_summary, "pca_coordinates": coords}
    (args.out / "distribution_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "pca_coordinates"}, indent=2))


if __name__ == "__main__": main()
