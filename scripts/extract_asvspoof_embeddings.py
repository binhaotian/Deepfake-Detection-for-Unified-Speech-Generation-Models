#!/usr/bin/env python3
"""Extract frozen Wav2Vec2/HuBERT embeddings for ASVspoof LA CM trials."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model


SPLITS = {
    "train": ("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train/flac"),
    "dev": ("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev/flac"),
    "eval": ("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval/flac"),
}


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    audio = audio.mean(dim=0)
    if sample_rate != 16_000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16_000)
    return audio


def masked_mean(hidden: torch.Tensor, input_mask: torch.Tensor, model) -> torch.Tensor:
    frame_mask = model._get_feature_vector_attention_mask(hidden.shape[1], input_mask)
    weights = frame_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def read_protocol(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"unexpected protocol row: {line!r}")
            target, audio_id, _, attack, label = fields
            rows.append({"target": target, "audio_id": audio_id, "attack": attack, "label": label})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("wav2vec2", "hubert"), required=True)
    parser.add_argument("--la-root", type=Path, default=Path("/root/LA"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-per-split", type=int)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = Wav2Vec2Model if args.encoder == "wav2vec2" else HubertModel
    model = model_cls.from_pretrained(str(args.model), local_files_only=True).eval().to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.out.mkdir(parents=True, exist_ok=True)

    config = {
        "encoder": args.encoder,
        "model": str(args.model.resolve()),
        "sample_rate": 16_000,
        "pooling": "attention-mask-aware mean over last_hidden_state",
        "dtype": args.dtype,
        "device": str(device),
        "batch_size": args.batch_size,
        "classes": ["bonafide", "spoof"],
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for split, (protocol_rel, audio_rel) in SPLITS.items():
        rows = read_protocol(args.la_root / protocol_rel)
        if args.max_per_split is not None:
            rows = rows[: args.max_per_split]
        audio_dir = args.la_root / audio_rel
        manifest_rows = []
        for begin in range(0, len(rows), args.batch_size):
            batch_rows = rows[begin : begin + args.batch_size]
            audios = [load_audio(audio_dir / f"{row['audio_id']}.flac") for row in batch_rows]
            inputs = processor([audio.numpy() for audio in audios], sampling_rate=16_000, return_tensors="pt", padding=True)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    hidden = model(input_values=inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).last_hidden_state
                embeddings = masked_mean(hidden.float(), inputs.attention_mask.to(device), model).cpu().numpy()
            for row, embedding in zip(batch_rows, embeddings):
                if embedding.ndim != 1 or not np.isfinite(embedding).all():
                    raise RuntimeError(f"invalid embedding: {split}/{row['audio_id']}")
                filename = f"{row['audio_id']}.npy"
                np.save(args.out / filename, embedding.astype(np.float32))
                manifest_rows.append({**row, "embedding": filename})
            done = min(begin + len(batch_rows), len(rows))
            if done % 1000 < len(batch_rows) or done == len(rows):
                print(f"{split}: processed {done}/{len(rows)}", flush=True)
        with (args.out / f"{split}.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["target", "audio_id", "attack", "label", "embedding"], delimiter="\t")
            writer.writeheader()
            writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
