#!/usr/bin/env python3
"""Extract frozen utterance embeddings for the four-class V2/V3 datasets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model


AUDIO = {
    "real": "anchor.wav",
    "tts": "auk_tts.wav",
    "se": "auk_se.wav",
    "tse": "auk_tse.wav",
}


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    audio = audio.mean(dim=0)
    if sample_rate != 16_000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16_000)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("wav2vec2", "hubert"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = parser.parse_args()

    rows = list(csv.DictReader((args.data_root / "manifest.tsv").open(), delimiter="\t"))
    if args.max_pairs is not None:
        rows = rows[: args.max_pairs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = Wav2Vec2Model if args.encoder == "wav2vec2" else HubertModel
    model = model_cls.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    hidden_size = int(model.config.hidden_size)

    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "encoder": args.encoder,
        "model": str(args.model.resolve()),
        "hidden_size": hidden_size,
        "sample_rate": 16_000,
        "pooling": "mean over encoder last_hidden_state",
        "dtype": args.dtype,
        "device": str(device),
        "pair_count": len(rows),
        "classes": list(AUDIO),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    output_rows = []
    for index, row in enumerate(rows, 1):
        pair_id = row["pair_id"]
        pair_dir = args.data_root / pair_id
        output = {"pair_id": pair_id, "speaker": row.get("speaker", ""), "text": row.get("text", "")}
        for label, filename in AUDIO.items():
            path = pair_dir / filename
            if not path.exists():
                raise FileNotFoundError(path)
            audio = load_audio(path)
            inputs = processor(audio.numpy(), sampling_rate=16_000, return_tensors="pt")
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    hidden = model(input_values=inputs.input_values.to(device)).last_hidden_state
                embedding = hidden.mean(dim=1).float().cpu().numpy()[0]
            if embedding.shape != (hidden_size,) or not np.isfinite(embedding).all():
                raise RuntimeError(f"invalid embedding: {pair_id}/{label}: {embedding.shape}")
            filename_out = f"{pair_id}__{label}.npy"
            np.save(args.out / filename_out, embedding.astype(np.float32))
            output[f"{label}_embedding"] = filename_out
        output_rows.append(output)
        if index % 25 == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)} pairs", flush=True)

    with (args.out / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
