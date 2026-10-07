#!/usr/bin/env python3
"""Extract utterance embeddings from local HuBERT/Wav2Vec2/Whisper encoders."""
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

AUDIO_NAMES = ("anchor.wav", "tts_reference.wav", "auk_tts.wav", "se_input.wav", "auk_se.wav", "tse_input.wav", "auk_tse.wav")


def load_audio(path: Path) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(0)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    return audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("hubert", "whisper"), required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/matched_full_5369_snr0_batch4_generated"))
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int)
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = ap.parse_args()
    if args.model is None:
        args.model = Path("encoders/hubert-large-ll60k" if args.encoder == "hubert" else "encoders/whisper-large-v3")

    rows = list(csv.DictReader((args.data_root / "manifest.tsv").open(), delimiter="\t"))
    if args.max_pairs is not None:
        rows = rows[:args.max_pairs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.encoder == "hubert":
        model = HubertModel.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    else:
        model = WhisperModel.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
        processor = WhisperFeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.out.mkdir(parents=True, exist_ok=True)
    hidden_size = int(model.config.hidden_size)
    config = {"encoder": args.encoder, "model": str(args.model.resolve()), "hidden_size": hidden_size, "sample_rate": 16000, "pooling": "mean over encoder last_hidden_state", "dtype": args.dtype, "device": str(device), "pair_count": len(rows), "audio_names": list(AUDIO_NAMES)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = []
    for index, row in enumerate(rows, 1):
        pair_id = row["pair_id"]
        pair_dir = args.data_root / pair_id
        out_row = {"pair_id": pair_id, "speaker_id": row.get("speaker_id", ""), "text": row.get("text", "")}
        for name in AUDIO_NAMES:
            audio = load_audio(pair_dir / name)
            inputs = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    if args.encoder == "hubert":
                        hidden = model(input_values=inputs.input_values.to(device)).last_hidden_state
                    else:
                        hidden = model.encoder(input_features=inputs.input_features.to(device)).last_hidden_state
                embedding = hidden.mean(1).float().cpu().numpy()[0]
            if embedding.shape != (hidden_size,) or not np.isfinite(embedding).all():
                raise RuntimeError(f"invalid embedding {pair_id}/{name}: {embedding.shape}")
            file_name = f"{pair_id}__{Path(name).stem}.npy"
            np.save(args.out / file_name, embedding.astype(np.float32))
            out_row[f"{Path(name).stem}_embedding"] = file_name
        manifest.append(out_row)
        if index % 10 == 0 or index == len(rows): print(f"processed {index}/{len(rows)} pairs", flush=True)
    fields = list(manifest[0])
    with (args.out / "manifest.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader(); w.writerows(manifest)


if __name__ == "__main__":
    main()
