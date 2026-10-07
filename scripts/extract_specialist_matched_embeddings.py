#!/usr/bin/env python3
"""Extract frozen encoder features for the AuK/specialist matched set.

The output is one dense feature matrix per encoder.  Rows follow the source
manifest and columns follow STREAMS, so later probes can compare task and
system effects without re-reading thousands of individual numpy files.
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


STREAMS = (
    "real",
    "auk_tts",
    "specialist_tts",
    "auk_se",
    "specialist_se",
    "auk_tse",
    "specialist_tse",
)


def stream_path(pair_id: str, stream: str, auk_root: Path, specialist_root: Path) -> Path:
    if stream == "real":
        return auk_root / pair_id / "anchor.wav"
    if stream == "auk_tts":
        return auk_root / pair_id / "auk_tts.wav"
    if stream == "auk_se":
        return auk_root / pair_id / "auk_se.wav"
    if stream == "auk_tse":
        return auk_root / pair_id / "auk_tse.wav"
    if stream == "specialist_tts":
        return specialist_root / "cosyvoice3" / pair_id / "specialist_tts.wav"
    if stream == "specialist_se":
        return specialist_root / "mossformer_gan_se" / pair_id / "specialist_se.wav"
    if stream == "specialist_tse":
        return specialist_root / "dae_tse" / pair_id / "specialist_tse.wav"
    raise KeyError(stream)


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    audio = audio.mean(dim=0)
    if sample_rate != 16_000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16_000)
    return audio.contiguous()


def pooled_hidden(model, inputs, device, amp_dtype, use_amp):
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            hidden = model(
                input_values=inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device)
                if getattr(inputs, "attention_mask", None) is not None
                else None,
            ).last_hidden_state
    mask = getattr(inputs, "attention_mask", None)
    if mask is not None and hasattr(model, "_get_feature_vector_attention_mask"):
        mask = model._get_feature_vector_attention_mask(hidden.shape[1], mask.to(device))
        mask = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return hidden.mean(dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("wav2vec2", "hubert"), required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--auk-root", type=Path, required=True)
    ap.add_argument("--specialist-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-pairs", type=int)
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = ap.parse_args()

    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if args.max_pairs is not None:
        rows = rows[: args.max_pairs]
    if not rows:
        raise RuntimeError("manifest contains no rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = Wav2Vec2Model if args.encoder == "wav2vec2" else HubertModel
    model = model_cls.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    hidden_size = int(model.config.hidden_size)

    args.out.mkdir(parents=True, exist_ok=True)
    features_path = args.out / "features.npy"
    features = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float32, shape=(len(rows), len(STREAMS), hidden_size)
    )

    output_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        for stream_index, stream in enumerate(STREAMS):
            paths = [stream_path(r["pair_id"], stream, args.auk_root, args.specialist_root) for r in batch_rows]
            missing = [str(p) for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(missing[0])
            audios = [load_audio(p).numpy() for p in paths]
            inputs = processor(
                audios,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            embedding = pooled_hidden(model, inputs, device, amp_dtype, use_amp)
            embedding = embedding.float().cpu().numpy()
            if embedding.shape != (len(batch_rows), hidden_size) or not np.isfinite(embedding).all():
                raise RuntimeError(f"invalid {stream} embedding at batch {start}")
            features[start : start + len(batch_rows), stream_index, :] = embedding
        for row in batch_rows:
            output_rows.append({"pair_id": row["pair_id"], "speaker": row.get("speaker", ""), "text": row.get("text", "")})
        features.flush()
        print(f"processed {min(start + args.batch_size, len(rows))}/{len(rows)} pairs", flush=True)

    with (args.out / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "speaker", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    config = {
        "encoder": args.encoder,
        "model": str(args.model.resolve()),
        "hidden_size": hidden_size,
        "sample_rate_hz": 16000,
        "pooling": "attention-mask-aware mean over last_hidden_state",
        "dtype": args.dtype,
        "device": str(device),
        "batch_size": args.batch_size,
        "pair_count": len(rows),
        "streams": list(STREAMS),
        "source_manifest": str(args.manifest.resolve()),
        "auk_root": str(args.auk_root.resolve()),
        "specialist_root": str(args.specialist_root.resolve()),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    del features


if __name__ == "__main__":
    main()
