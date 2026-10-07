#!/usr/bin/env python
"""Plot test-set confusion matrices before/after RMS normalization."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

models = [
    (
        "Wav2Vec2 Large",
        ROOT / "data/probe_results/wav2vec2-large-fourclass/report.json",
        ROOT / "data/probe_results/wav2vec2-large-rmsnorm-fourclass/report.json",
    ),
    (
        "HuBERT Large",
        ROOT / "data/probe_results/hubert-large-fourclass/report.json",
        ROOT / "data/probe_results/hubert-large-rmsnorm-fourclass/report.json",
    ),
    (
        "Whisper Large-v3",
        ROOT / "data/probe_results/whisper-large-fourclass/report.json",
        ROOT / "data/probe_results/whisper-large-rmsnorm-fourclass/report.json",
    ),
]
labels = ["Real", "TTS", "SE", "TSE"]


def load(path: Path):
    obj = json.loads(path.read_text())
    test = obj["metrics"]["test"]
    return np.asarray(test["confusion_matrix"], dtype=int), test["accuracy"]


def annotate(ax, cm, normalized=False):
    vmax = cm.max() if not normalized else 1.0
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f"{cm[i, j]:.1%}" if normalized else f"{cm[i, j]:d}"
            color = "white" if cm[i, j] > 0.55 * vmax else "black"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")


def make_plot(normalized=False):
    fig, axes = plt.subplots(3, 2, figsize=(10, 13), constrained_layout=True)
    for row, (name, raw_path, rms_path) in enumerate(models):
        raw, raw_acc = load(raw_path)
        rms, rms_acc = load(rms_path)
        if normalized:
            raw_plot = raw / raw.sum(axis=1, keepdims=True)
            rms_plot = rms / rms.sum(axis=1, keepdims=True)
        else:
            raw_plot, rms_plot = raw, rms
        annotate(axes[row, 0], raw_plot, normalized)
        annotate(axes[row, 1], rms_plot, normalized)
        axes[row, 0].set_title(f"{name}\nOriginal — Acc {raw_acc:.2%}")
        axes[row, 1].set_title(f"{name}\nRMS-normalized — Acc {rms_acc:.2%}")
    suffix = "row_normalized" if normalized else "counts"
    fig.suptitle(
        "Speaker-disjoint four-class test confusion matrices\n"
        "Real / TTS / SE / TSE",
        fontsize=15,
    )
    path = OUT / f"confusion_matrices_original_vs_rmsnorm_{suffix}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_plot(False))
    print(make_plot(True))
