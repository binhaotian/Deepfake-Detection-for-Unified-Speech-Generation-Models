#!/usr/bin/env python3
"""Benchmark AuK batch sizes on four prepared SE inputs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torchaudio

from auk.infer.infer_auk import AukInfer, save_audio


INSTRUCTION = "Remove only the background noise, preserve everything else, and output audio of the same length."


def messages(audio: Path) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": INSTRUCTION},
        {"type": "audio", "audio": str(audio)},
    ]}]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/matched_v0_snr0"))
    parser.add_argument("--out", type=Path, default=Path("data/batch_benchmark"))
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()

    pair_ids = [f"pair_{index:04d}" for index in range(1, 5)]
    inputs = [args.root / pair_id / "se_input.wav" for pair_id in pair_ids]
    requests = [messages(path) for path in inputs]
    args.out.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    engine = AukInfer("ckpts/AuK/config.yaml", "ckpts/AuK/auk_base.safetensors", dtype="bf16")
    sync()
    load_seconds = time.perf_counter() - load_start

    # Trigger lazy audio libraries and CUDA kernels outside measured runs.
    engine.generate(requests[0], audio=str(inputs[0]), nfe=4, seed=args.seed)
    sync()

    results = []
    for batch_size in (1, 2, 4):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        run_dir = args.out / f"batch_{batch_size}"
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        output_index = 0
        for begin in range(0, len(inputs), batch_size):
            chunk_inputs = inputs[begin : begin + batch_size]
            chunk_requests = requests[begin : begin + batch_size]
            if batch_size == 1:
                audio, sr = engine.generate(
                    chunk_requests[0], audio=str(chunk_inputs[0]),
                    nfe=args.nfe, cfg_strength=2.0, sway_sampling_coef=-1.0,
                    seed=args.seed,
                )
                outputs = [audio]
            else:
                outputs, sr = engine.generate_batch(
                    chunk_requests,
                    audios=[str(path) for path in chunk_inputs],
                    nfe=args.nfe, cfg_strength=2.0, sway_sampling_coef=-1.0,
                    seed=args.seed,
                )
            for audio in outputs:
                save_audio(audio, sr, str(run_dir / f"{pair_ids[output_index]}_auk_se.wav"))
                output_index += 1
        sync()
        elapsed = time.perf_counter() - started
        peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0

        valid = True
        output_seconds = 0.0
        for path in sorted(run_dir.glob("*.wav")):
            audio, sr = torchaudio.load(str(path))
            valid &= sr == 24000 and audio.shape[0] == 1 and bool(torch.isfinite(audio).all())
            output_seconds += audio.shape[-1] / sr
        results.append({
            "batch_size": batch_size,
            "wall_seconds": elapsed,
            "seconds_per_item": elapsed / len(inputs),
            "output_audio_seconds": output_seconds,
            "rtf": elapsed / output_seconds,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "valid": valid,
        })

    report = {"model_load_seconds": load_seconds, "nfe": args.nfe, "items": len(inputs), "results": results}
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
