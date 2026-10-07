#!/usr/bin/env python3
"""Run AuK content editing for accepted GPT-curated pair mappings.

The job is resumable: an existing ``edited.wav`` is skipped unless --overwrite
is supplied. Outputs are placed on /data by default because the root volume is
nearly full.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio

from auk.infer.infer_auk import AukInfer, save_audio


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BASE = ROOT / "data/matched_full_5369_snr0_batch4_generated"
TEXT_MANIFEST = ROOT / "data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv"
DEFAULT_OUT = Path("/data/AuK_content_edit_full_20261001")


def duration(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def make_messages(instruction: str, audio_path: Path) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "audio", "audio": str(audio_path)},
    ]}]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=TEXT_MANIFEST)
    parser.add_argument("--source-base", type=Path, default=SOURCE_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / "generation_progress.jsonl"
    report_path = args.out / "generation_report.json"
    config_path = args.out / "generation_config.json"

    with args.manifest.open(encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [r for r in all_rows if r.get("status") == "accepted" and r.get("auk_instruction")]
    if not rows:
        raise RuntimeError("No accepted rows with auk_instruction found")

    work = []
    skipped = 0
    for row in rows:
        pair_dir = args.out / row["pair_id"]
        output_path = pair_dir / "auk_content_edit.wav"
        source_path = args.source_base / row["source_pair_dir"] / "anchor.wav"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        work.append({
            "row": row,
            "source": source_path,
            "output": output_path,
            "seconds": duration(source_path),
            # One-word replacement: official content-scaled duration has a
            # one-word/one-word ratio, so preserving the anchor duration is
            # the deterministic equivalent.
            "seed": int(row["pair_id"].split("_")[-1]) + 310000,
        })
    work.sort(key=lambda x: x["seconds"])

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "source_base": str(args.source_base),
        "output_base": str(args.out),
        "total_manifest_rows": len(all_rows),
        "accepted_rows": len(rows),
        "already_existing_skipped": skipped,
        "pending_rows": len(work),
        "batch_size": args.batch_size,
        "nfe": args.nfe,
        "cfg": args.cfg,
        "sway": args.sway,
        "dtype": args.dtype,
        "duration_policy": "anchor_duration; single-word replacement has one-word content ratio",
        "instruction_policy": "Replace '{old_word}' with '{new_word}'.",
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not work:
        print(json.dumps({"status": "complete", **config}, ensure_ascii=False, indent=2))
        return

    engine = AukInfer("ckpts/AuK/config.yaml", "ckpts/AuK/auk_base.safetensors", dtype=args.dtype)
    sync_cuda()
    started = time.perf_counter()
    generated = 0
    failures = []
    total_batches = math.ceil(len(work) / args.batch_size)
    for batch_no, begin in enumerate(range(0, len(work), args.batch_size), 1):
        batch = work[begin : begin + args.batch_size]
        requests = [make_messages(item["row"]["auk_instruction"], item["source"]) for item in batch]
        try:
            outputs, sample_rate = engine.generate_batch(
                requests,
                audios=[str(item["source"]) for item in batch],
                gen_seconds=[item["seconds"] for item in batch],
                nfe=args.nfe,
                cfg_strength=args.cfg,
                sway_sampling_coef=args.sway,
                seed=[item["seed"] for item in batch],
            )
            for item, audio in zip(batch, outputs):
                item["output"].parent.mkdir(parents=True, exist_ok=True)
                save_audio(audio, sample_rate, str(item["output"]))
                (item["output"].parent / "instruction.txt").write_text(
                    item["row"]["auk_instruction"] + "\n", encoding="utf-8"
                )
                generated += 1
        except Exception as exc:
            failures.append({"batch": batch_no, "pair_ids": [x["row"]["pair_id"] for x in batch], "error": repr(exc)})
            raise
        sync_cuda()
        event = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "batch": batch_no,
            "batch_count": total_batches,
            "generated_this_batch": len(batch),
            "generated_total": generated,
            "pending_total": len(work),
            "pair_ids": [x["row"]["pair_id"] for x in batch],
        }
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"[content_edit] batch {batch_no}/{total_batches}: total={generated}/{len(work)}", flush=True)

    elapsed = time.perf_counter() - started
    report = {
        **config,
        "status": "complete" if not failures else "failed",
        "generated": generated,
        "failures": failures,
        "wall_seconds": elapsed,
        "seconds_per_output": elapsed / generated if generated else None,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
