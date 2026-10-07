#!/usr/bin/env python3
"""Build the deterministic v0 anchor manifest and synthesize SE/TSE inputs.

This script intentionally does not run AuK.  It prepares all real/reference/
mixture inputs and records every choice needed to reproduce them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import tarfile
from pathlib import Path

import torch
import torchaudio


SR = 24_000
SEED = 20260922


def read_meta(meta_tar: Path) -> dict[str, dict[str, str]]:
    with tarfile.open(meta_tar) as z:
        f = z.extractfile("ASVspoof2019_LA_VCTK_MetaInfo.tsv")
        assert f is not None
        return {
            r["ASVspoof_ID"]: r
            for r in csv.DictReader((line.decode() for line in f), delimiter="\t")
        }


def read_protocol(path: Path) -> dict[str, str]:
    return {line.split()[1]: line.split()[-1] for line in path.read_text().splitlines() if line.strip()}


def read_enrollment_protocol(path: Path) -> dict[str, list[str]]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        target, refs = line.split()
        out[target] = refs.split(",")
    return out


def speaker_info(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 3:
            out[fields[0]] = fields[2]
    return out


def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x * x) + 1e-12))


def load(path: Path) -> tuple[torch.Tensor, int]:
    x, sr = torchaudio.load(str(path))
    x = x.mean(0, keepdim=True).to(torch.float32)
    return x, sr


def resample(x: torch.Tensor, sr: int) -> torch.Tensor:
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    return x


def fit(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[-1] == 0:
        raise ValueError("empty audio")
    if x.shape[-1] < n:
        reps = math.ceil(n / x.shape[-1])
        x = x.repeat(1, reps)
    return x[..., :n]


def write(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), x, SR, encoding="PCM_S", bits_per_sample=16)


def choose_reference(enroll: dict[str, list[str]], target: str, anchor_id: str, meta: dict) -> str:
    candidates = [x for x in enroll[target] if x != anchor_id]
    if not candidates:
        raise RuntimeError(f"no independent enrollment for {target}")
    # Stable choice, independent of filesystem ordering.
    return sorted(candidates)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--la", type=Path, default=Path("/root/LA"))
    ap.add_argument("--vctk-text", type=Path, default=Path("/root/VCTK-text"))
    ap.add_argument("--meta", type=Path, default=Path("ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz"))
    ap.add_argument("--noise-dir", type=Path, default=Path("data/noise_esc50"))
    ap.add_argument("--out", type=Path, default=Path("data/matched_v0"))
    ap.add_argument("--snr-db", type=float, default=10.0)
    ap.add_argument("--sir-db", type=float, default=0.0)
    args = ap.parse_args()

    meta = read_meta(args.meta)
    labels = read_protocol(args.la / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    enroll = {}
    for p in sorted((args.la / "ASVspoof2019_LA_asv_protocols").glob("*.eval.*.trn.txt")):
        enroll.update(read_enrollment_protocol(p))
    genders = speaker_info(args.vctk_text / "speaker-info.txt")

    # The 100 text files were deliberately downloaded as the exact v0 candidate set.
    text_files = sorted((args.vctk_text / "txt").glob("p*/*.txt"))
    candidates = []
    for tf in text_files:
        vctk_id = tf.stem
        matches = [i for i, r in meta.items() if r["VCTK_ID"] == vctk_id and labels.get(i) == "bonafide"]
        if len(matches) != 1:
            raise RuntimeError(f"expected one bona fide mapping for {vctk_id}, got {matches}")
        candidates.append((vctk_id, matches[0], tf))
    if len(candidates) != 100:
        raise RuntimeError(f"expected 100 anchors, found {len(candidates)}")

    # Map ASV target IDs to VCTK speaker IDs using enrollment metadata.
    target_spk = {}
    for target, refs in enroll.items():
        spks = {meta[r]["VCTK_ID"].split("_")[0] for r in refs if r in meta and meta[r]["VCTK_ID"] != "-"}
        if len(spks) == 1:
            target_spk[target] = next(iter(spks))

    # Group enrollment target IDs by speaker; choose a different same-gender speaker
    # for TSE and rotate it deterministically by pair index.
    spk_to_target = {}
    for t, s in target_spk.items():
        spk_to_target.setdefault(s, t)
    speaker_ids = sorted(spk_to_target)
    noise_files = sorted(args.noise_dir.glob("*.wav"))
    if len(noise_files) < 4:
        raise RuntimeError(f"need at least 4 downloaded noise clips in {args.noise_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.tsv"
    fields = ["pair_id", "anchor_id", "anchor_vctk_id", "speaker", "gender", "text",
              "anchor_source", "tts_reference_source", "interferer_source", "noise_source",
              "snr_db", "sir_db", "measured_snr_db", "measured_sir_db", "seed", "pair_dir"]
    rng = random.Random(SEED)
    rows = []
    for idx, (vctk_id, anchor_id, tf) in enumerate(candidates, 1):
        row_seed = SEED + idx
        speaker = vctk_id.split("_")[0]
        target = next((t for t, s in target_spk.items() if s == speaker), None)
        if target is None:
            raise RuntimeError(f"no ASV target for speaker {speaker}")
        ref_id = choose_reference(enroll, target, anchor_id, meta)
        same_gender = [s for s in speaker_ids if s != speaker and genders.get(s) == genders.get(speaker)]
        pool = same_gender or [s for s in speaker_ids if s != speaker]
        int_spk = pool[(idx - 1) % len(pool)]
        int_target = spk_to_target[int_spk]
        int_id = sorted(enroll[int_target])[(idx - 1) % len(enroll[int_target])]
        noise = noise_files[(idx - 1) % len(noise_files)]

        pair = args.out / f"pair_{idx:04d}"
        anchor_path = args.la / "ASVspoof2019_LA_eval/flac" / f"{anchor_id}.flac"
        ref_path = args.la / "ASVspoof2019_LA_eval/flac" / f"{ref_id}.flac"
        int_path = args.la / "ASVspoof2019_LA_eval/flac" / f"{int_id}.flac"
        if not all(p.exists() for p in (anchor_path, ref_path, int_path)):
            raise FileNotFoundError(anchor_path, ref_path, int_path)

        anchor, asr = load(anchor_path); anchor = resample(anchor, asr)
        ref, rsr = load(ref_path); ref = resample(ref, rsr)
        inter, isr = load(int_path); inter = resample(inter, isr)
        noise_x, nsr = load(noise); noise_x = resample(noise_x, nsr)
        n = anchor.shape[-1]
        noise_x = fit(noise_x, n)
        inter = fit(inter, n)
        # Exact RMS-controlled construction.  Noise/interferer are aligned at t=0
        # and fully overlap the anchor; there is no hidden random crop.
        snr_db, sir_db = args.snr_db, args.sir_db
        noise_x = noise_x * (rms(anchor) / (10 ** (snr_db / 20)) / max(rms(noise_x), 1e-8))
        inter = inter * (rms(anchor) / (10 ** (sir_db / 20)) / max(rms(inter), 1e-8))
        se = anchor + noise_x
        tse = anchor + inter
        # Apply one common scale to every file in this pair.  This prevents
        # per-file clipping from changing the requested SNR/SIR relationship.
        pair_peak = max(float(x.abs().max()) for x in (anchor, ref, noise_x, inter, se, tse))
        if pair_peak > 0.98:
            scale = 0.98 / pair_peak
            anchor, ref, noise_x, inter, se, tse = [x * scale for x in (anchor, ref, noise_x, inter, se, tse)]
        write(pair / "anchor.wav", anchor)
        write(pair / "tts_reference.wav", ref)
        write(pair / "se_noise.wav", noise_x)
        write(pair / "se_input.wav", se)
        write(pair / "tse_interferer.wav", inter)
        write(pair / "tse_input.wav", tse)
        text = " ".join(tf.read_text().split())
        rows.append({"pair_id": f"pair_{idx:04d}", "anchor_id": anchor_id, "anchor_vctk_id": vctk_id,
                     "speaker": speaker, "gender": genders.get(speaker, ""), "text": text,
                     "anchor_source": str(anchor_path), "tts_reference_source": str(ref_path),
                     "interferer_source": str(int_path), "noise_source": str(noise), "snr_db": snr_db,
                     "sir_db": sir_db, "measured_snr_db": 20 * math.log10(rms(anchor) / rms(noise_x)),
                     "measured_sir_db": 20 * math.log10(rms(anchor) / rms(inter)),
                     "seed": row_seed, "pair_dir": str(pair)})
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} pairs to {args.out}")


if __name__ == "__main__":
    main()
