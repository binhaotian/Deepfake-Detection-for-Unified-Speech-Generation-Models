#!/usr/bin/env python3
"""Extract utterance-level Wav2Vec2 embeddings for the AuK matched set."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


AUDIO_NAMES = (
    "anchor.wav", "tts_reference.wav", "auk_tts.wav", "se_input.wav",
    "auk_se.wav", "tse_input.wav", "auk_tse.wav",
)


def load_audio(path: Path, target_sr: int = 16_000) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(dim=0)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/matched_full_5369_snr0_batch4_generated"))
    ap.add_argument("--model", type=Path, default=Path("encoders/wav2vec2-large-960h-lv60-self"))
    ap.add_argument("--out", type=Path, default=Path("data/embeddings/wav2vec2-large-960h-lv60-self"))
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = ap.parse_args()

    rows = list(csv.DictReader((args.data_root / "manifest.tsv").open(), delimiter="\t"))
    if args.max_pairs is not None:
        rows = rows[: args.max_pairs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2Model.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": str(args.model.resolve()),
        "hidden_size": int(model.config.hidden_size),
        "sample_rate": 16_000,
        "pooling": "mean over last_hidden_state frames",
        "device": str(device),
        "dtype": args.dtype,
        "pair_count": len(rows),
        "audio_names": list(AUDIO_NAMES),
    }
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    manifest = []
    for row in rows:
        pair_id = row["pair_id"]
        pair_dir = args.data_root / pair_id
        output = {"pair_id": pair_id, "speaker_id": row.get("speaker_id", ""), "text": row.get("text", "")}
        for name in AUDIO_NAMES:
            path = pair_dir / name
            if not path.exists():
                raise FileNotFoundError(path)
            audio = load_audio(path)
            inputs = processor(audio.numpy(), sampling_rate=16_000, return_tensors="pt")
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        hidden = model(inputs.input_values.to(device)).last_hidden_state
                else:
                    hidden = model(inputs.input_values.to(device)).last_hidden_state
                embedding = hidden.mean(dim=1).float().cpu().numpy()[0]
            if not np.isfinite(embedding).all():
                raise RuntimeError(f"non-finite embedding: {pair_id}/{name}")
            key = Path(name).stem
            np.save(args.out / f"{pair_id}__{key}.npy", embedding.astype(np.float32))
            output[f"{key}_embedding"] = f"{pair_id}__{key}.npy"
        manifest.append(output)
        if len(manifest) % 10 == 0 or len(manifest) == len(rows):
            print(f"processed {len(manifest)}/{len(rows)} pairs", flush=True)

    fields = list(manifest[0]) if manifest else ["pair_id"]
    with (args.out / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
