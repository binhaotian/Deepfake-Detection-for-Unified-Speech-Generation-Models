#!/usr/bin/env python3
"""Create loudness-controlled analysis copies for the four core classes."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch, torchaudio

FILES = {"real":"anchor.wav", "tts":"auk_tts.wav", "se":"auk_se.wav", "tse":"auk_tse.wav"}
EXTRA_FILES = ("tts_reference.wav", "se_noise.wav", "se_input.wav", "tse_interferer.wav", "tse_input.wav")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,default=Path('data/matched_full_5369_snr0_batch4_generated')); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--target-rms',type=float,default=0.07); args=ap.parse_args()
    rows=list(csv.DictReader((args.data_root/'manifest.tsv').open(),delimiter='\t')); args.out.mkdir(parents=True,exist_ok=True)
    stats=[]
    for i,row in enumerate(rows,1):
        pair=row['pair_id']; src=args.data_root/pair; dst=args.out/pair; dst.mkdir(exist_ok=True)
        for cls,name in FILES.items():
            x,sr=torchaudio.load(str(src/name)); x=x.mean(0,keepdim=True); rms=x.square().mean().sqrt().clamp_min(1e-8); y=x*(args.target_rms/rms); y=y.clamp(-0.999,0.999); torchaudio.save(str(dst/name),y,sr)
            stats.append({'pair_id':pair,'class':cls,'input_rms':float(rms),'output_rms':float(y.square().mean().sqrt()),'input_peak':float(x.abs().max()),'output_peak':float(y.abs().max())})
        for name in EXTRA_FILES:
            src_file = src / name
            if src_file.exists():
                torchaudio.save(str(dst / name), *torchaudio.load(str(src_file)))
        if i%500==0 or i==len(rows): print(f'processed {i}/{len(rows)} pairs',flush=True)
    with (args.out/'manifest.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(rows)
    (args.out/'normalization_config.json').write_text(json.dumps({'target_rms':args.target_rms,'files':FILES,'source_root':str(args.data_root.resolve())},indent=2)+'\n')
    with (args.out/'normalization_stats.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=stats[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(stats)
if __name__=='__main__': main()
