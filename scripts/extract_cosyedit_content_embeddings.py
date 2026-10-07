#!/usr/bin/env python3
"""Extract paired anchor/CosyEdit-CE embeddings for accepted pairs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import (
    HubertModel,
    WhisperFeatureExtractor,
    WhisperModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("wav2vec2", "hubert", "whisper"), required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pair-manifest", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--cosyedit-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    args = ap.parse_args()

    with args.pair_manifest.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r.get("status") == "accepted"]
    if not rows:
        raise RuntimeError("no accepted rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.encoder == "wav2vec2":
        model = Wav2Vec2Model.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    elif args.encoder == "hubert":
        model = HubertModel.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    else:
        model = WhisperModel.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
        processor = WhisperFeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    hidden_size = int(model.config.hidden_size)
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.out.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(args.out / "features.npy", mode="w+", dtype=np.float32, shape=(len(rows), 2, hidden_size))
    out_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        paths = []
        for row in batch:
            anchor = args.source_root / row["source_pair_dir"] / "anchor.wav"
            edit = args.cosyedit_root / f"{row['pair_id']}.wav"
            if not anchor.is_file():
                raise FileNotFoundError(anchor)
            if not edit.is_file():
                raise FileNotFoundError(edit)
            paths.extend([anchor, edit])
        if args.encoder == "whisper":
            inputs = processor(
                [load_audio(p).numpy() for p in paths],
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=3000,
                truncation=True,
                return_attention_mask=True,
            )
        else:
            inputs = processor([load_audio(p).numpy() for p in paths], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                if args.encoder == "whisper":
                    hidden = model.encoder(input_features=inputs.input_features.to(device)).last_hidden_state
                    mask = None
                else:
                    attention_mask = inputs.attention_mask.to(device) if getattr(inputs, "attention_mask", None) is not None else None
                    hidden = model(input_values=inputs.input_values.to(device), attention_mask=attention_mask).last_hidden_state
                    mask = attention_mask
            if mask is not None and hasattr(model, "_get_feature_vector_attention_mask"):
                valid = model._get_feature_vector_attention_mask(hidden.shape[1], mask).to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            else:
                pooled = hidden.mean(dim=1)
            pooled = pooled.float().cpu().numpy()
        if pooled.shape != (len(paths), hidden_size) or not np.isfinite(pooled).all():
            raise RuntimeError(f"invalid embeddings at batch {start}: {pooled.shape}")
        for j, row in enumerate(batch):
            i = start + j
            features[i, 0] = pooled[2 * j]
            features[i, 1] = pooled[2 * j + 1]
            out_rows.append({"pair_id": row["pair_id"], "speaker": row.get("speaker", ""), "text": row.get("source_text", ""), "old_word": row.get("old_word", ""), "new_word": row.get("new_word", ""), "edited_text": row.get("edited_text", "")})
        features.flush()
        print(f"[{args.encoder}] processed {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    with (args.out / "manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter="\t"); w.writeheader(); w.writerows(out_rows)
    (args.out / "anchor_embeddings.npy").write_bytes((args.out / "features.npy").read_bytes()) if False else None
    (args.out / "config.json").write_text(json.dumps({"encoder": args.encoder, "model": str(args.model.resolve()), "hidden_size": hidden_size, "sample_rate": 16000, "pooling": "attention-mask-aware mean over encoder last_hidden_state; Whisper mean", "dtype": args.dtype, "device": str(device), "batch_size": args.batch_size, "pair_count": len(rows), "order": ["anchor", "cosyedit_content_edit"]}, indent=2) + "\n")
    del features


if __name__ == "__main__":
    main()
