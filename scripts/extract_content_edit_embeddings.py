#!/usr/bin/env python3
"""Extract paired frozen-encoder embeddings for the full content-edit set.

For every accepted pair, the two rows are:
  0: the original bona-fide anchor
  1: the AuK content-edited output

Embeddings are mean pooled over valid encoder frames (padding is excluded).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    return wav


def frame_mask(model, hidden_len: int, attention_mask: torch.Tensor) -> torch.Tensor:
    # Both Wav2Vec2Model and HubertModel expose this helper through the base model.
    return model._get_feature_vector_attention_mask(hidden_len, attention_mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("wav2vec2", "hubert"), required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pair-manifest", type=Path, default=Path("data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv"))
    ap.add_argument("--source-root", type=Path, default=Path("data/matched_full_5369_snr0_batch4_generated"))
    ap.add_argument("--generated-root", type=Path, default=Path("/data/AuK_content_edit_full_20261001"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.pair_manifest.open(encoding="utf-8"), delimiter="\t") if r.get("status") == "accepted"]
    if not rows:
        raise RuntimeError("No accepted rows in pair manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = Wav2Vec2Model if args.encoder == "wav2vec2" else HubertModel
    model = model_cls.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    hidden_size = int(model.config.hidden_size)
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "encoder": args.encoder,
        "model": str(args.model.resolve()),
        "device": str(device),
        "dtype": args.dtype,
        "sample_rate": 16000,
        "pooling": "attention-mask-weighted mean over last_hidden_state",
        "pair_count": len(rows),
        "batch_size": args.batch_size,
        "feature_shape": [len(rows), 2, hidden_size],
        "order": ["anchor", "content_edit"],
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    features = np.empty((len(rows), 2, hidden_size), dtype=np.float32)
    out_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        wavs = []
        for row in batch_rows:
            anchor = args.source_root / row["source_pair_dir"] / "anchor.wav"
            edited = args.generated_root / row["pair_id"] / "auk_content_edit.wav"
            if not anchor.is_file():
                raise FileNotFoundError(anchor)
            if not edited.is_file():
                raise FileNotFoundError(edited)
            wavs.extend([load_audio(anchor), load_audio(edited)])
        inputs = processor([x.numpy() for x in wavs], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                hidden = model(input_values=input_values, attention_mask=attention_mask).last_hidden_state
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                valid = frame_mask(model, hidden.shape[1], attention_mask).to(hidden.dtype)
                pooled = (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = pooled.float().cpu().numpy()
        if not np.isfinite(pooled).all():
            raise RuntimeError(f"non-finite embedding in batch starting at {start}")
        for j, row in enumerate(batch_rows):
            idx = start + j
            features[idx, 0] = pooled[2 * j]
            features[idx, 1] = pooled[2 * j + 1]
            out_rows.append({
                "pair_id": row["pair_id"],
                "text_group_id": row.get("text_group_id", ""),
                "speaker": row.get("speaker", ""),
                "anchor_id": row.get("anchor_id", ""),
                "source_text": row.get("source_text", ""),
                "old_word": row.get("old_word", ""),
                "new_word": row.get("new_word", ""),
                "edited_text": row.get("edited_text", ""),
                "route": row.get("route", ""),
                "part_of_speech": row.get("part_of_speech", ""),
            })
        done = min(start + len(batch_rows), len(rows))
        print(f"[{args.encoder}] processed {done}/{len(rows)} pairs", flush=True)

    np.save(args.out / "features.npy", features)
    np.save(args.out / "anchor_embeddings.npy", features[:, 0])
    np.save(args.out / "content_edit_embeddings.npy", features[:, 1])
    with (args.out / "manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"saved {args.out} shape={features.shape}")


if __name__ == "__main__":
    main()
