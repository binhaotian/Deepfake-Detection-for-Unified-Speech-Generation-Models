# G2 Restoration Pilot Report

Date: 2026-10-07
Experiment: `EXP-011`
Status: generated and automatically validated; initial human listening review
identified protocol issues that require a revision pilot.

## Objective

Extend the existing AuK matched set with three Restoration and Source Selection
tasks while holding the real anchor, speaker, and linguistic content fixed:

```text
RES-02  Dereverberation
RES-03  Full enhancement: noise + reverberation
RES-04  Narrowband channel restoration
```

This is a 50-anchor pilot, not an authorization for full 5,369-pair generation.

## Authoritative artifact

```text
/data/AuK_g2_restoration_pilot_50_20261007/
```

The pilot contains 50 pairs from 48 speakers. Selection is speaker-balanced
round robin over the completed 5,369-pair manifest.

Packaged deliverables:

```text
/data/AuK_g2_restoration_pilot_50_20261007.zip
/data/AuK_g2_restoration_listening_10_20261007.zip
/data/AuK_g2_restoration_packages_20261007.sha256
```

The complete ZIP is approximately 115 MB. The listening ZIP is approximately
19 MB and contains 10 RT60-diverse pairs with seven ordered WAV files per pair.
Both ZIP integrity tests and all internal SHA256 checks passed.

## Input construction

### Dereverberation

```text
RIR source: OpenSLR SLR28 RIRS_NOISES/simulated_rirs/mediumroom
RIR candidates scanned: 2,000 of 20,000 medium-room WAV files
eligible estimated RT60: 0.45--0.65 seconds
eligible candidates: 109
selected pool: 20 RIRs from different rooms
actual selected RT60 range: 0.452408--0.644199 seconds
construction: fftconvolve(anchor, aligned RIR), then crop to anchor samples
```

Instruction:

```text
Remove only the room reverberation, preserve everything else,
and output audio of the same length.
```

### Full enhancement

```text
full_input = (anchor * the same pair RIR) + the existing pair noise
requested SNR: 0 dB
measured SNR range after PCM write/read: approximately +/- 2.2e-7 dB
```

Instruction:

```text
Preserve all speakers, remove noise and reverberation,
and output clean speech of the same length.
```

### Channel restoration

```text
24-kHz anchor -> 8 kHz -> 24 kHz
resampler: scipy.signal.resample_poly
codec: none
operation label: narrowband_8khz
median attenuation above 4.5 kHz: 41.45 dB
```

Instruction:

```text
Repair the telephone effect and restore natural, clear speech.
```

The implemented corruption is bandwidth limitation only; do not describe this
pilot as a G.711 codec condition.

## AuK inference

```text
model: AuK Base
checkpoint: /root/AuK/ckpts/AuK/auk_base.safetensors
sample rate: 24 kHz
dtype: BF16
NFE: 32
CFG: 2.0
sway: -1.0
batch size: 4
resume: skip existing output
```

Generation totals:

| Task | Outputs | Wall time | Seconds/output |
|---|---:|---:|---:|
| Dereverberation | 50 | 83.87 s | 1.677 |
| Full enhancement | 50 | 42.46 s | 0.849 |
| Channel restoration | 50 | 41.29 s | 0.826 |
| Total generated | 150 | 227.51 s including model load/validation | — |

Model loading took 43.68 seconds. Peak reserved GPU memory was 18.81 GiB.

## Validation

```text
input preparation errors: 0
AuK generation validation errors: 0
AuK generation validation warnings: 0
WAV files decoded independently with FFmpeg: 470/470
maximum output/input duration difference: 0.019958 seconds
post-generation format/non-finite errors: 0
```

Some generated outputs reach a numerical peak of 1.0. The maximum fraction of
samples at or above 0.999 is 0.000588, below the predefined warning threshold
of 0.001. Human review must still listen for audible clipping or other defects.

## Directory structure

```text
pair_xxxxx/
├── anchor.wav
├── input_metadata.json
├── dereverb/
│   ├── dereverb_input.wav
│   └── auk_dereverb.wav
├── full_enhancement/
│   ├── full_enhance_speech.wav
│   ├── full_enhance_noise.wav
│   ├── full_enhance_input.wav
│   └── auk_full_enhance.wav
└── channel_restoration/
    ├── channel_input.wav
    └── auk_channel_restore.wav
```

Root-level records:

```text
manifest.tsv
input_config.json
input_preparation_report.json
input_preparation_console.log
rir_manifest.json
rir_pool/
generation_config.json
generation_progress.jsonl
generation_report.json
generation_console.log
post_generation_qc.json
```

## Exact commands

Input preparation:

```bash
/root/anaconda3/envs/auk/bin/python scripts/prepare_g2_restoration_pilot.py \
  --source /root/AuK/data/matched_full_5369_snr0_batch4_generated \
  --rir-root /data/resources/openslr28/extracted/RIRS_NOISES/simulated_rirs/mediumroom \
  --out /data/AuK_g2_restoration_pilot_50_20261007 \
  --count 50 --rir-pool-size 20 --rir-scan-limit 2000 \
  --rt60-low 0.45 --rt60-high 0.65 --seed 20261007
```

AuK generation:

```bash
PYTHONPATH=/root/AuK/src /root/anaconda3/envs/auk/bin/python \
  scripts/generate_g2_restoration_pilot.py \
  --root /data/AuK_g2_restoration_pilot_50_20261007 \
  --batch-size 4 --nfe 32 --cfg 2.0 --sway -1.0 --dtype bf16
```

## Interpretation boundary and next decision

This run proves that the three data paths can be constructed reproducibly and
that AuK can generate all requested outputs without structural failures. It
does not yet prove perceptual success or justify full-scale generation.

Next required step:

```text
human listening review of a diverse subset
-> mark content change, speaker drift, residual corruption, hallucination,
   boundary artifacts, and audible clipping
-> decide FULL / SUBSET / CONTROL / DEFER independently for each task
```

## Initial listening feedback and channel diagnosis

Initial human review on 2026-10-07 identified two issues:

1. The 0 dB noise in Full Enhancement perceptually dominates the moderate
   reverberation. This makes the condition behave mainly like difficult
   denoising rather than a balanced noise-plus-reverberation task.
2. The narrowband input and AuK channel-restoration output sound very similar.

Objective analysis of all 50 channel pairs supports the second observation:

```text
median high-band (>4.5 kHz) energy gain, output vs input: +3.11 dB
median recovered fraction of missing high-band energy: 0.0055%
median spectral centroid:
  anchor 565 Hz; input 516 Hz; output 551 Hz
median log-spectral distance:
  anchor vs input 34.15 dB
  anchor vs output 27.22 dB
```

Interpretation: AuK changed the spectrum, mainly within the retained band, but
did not materially reconstruct the missing high-frequency content. The current
channel condition must not be scaled as a successful bandwidth-restoration
task.

Proposed revision, not yet accepted:

```text
Full Enhancement:
  retain the current 0 dB output as a stress control;
  create the main balanced condition at 10 dB SNR using the same RIR/noise IDs.

Channel Restoration:
  first keep the same 8-kHz narrowband input and test an explicit
  speech-super-resolution/bandwidth-restoration instruction on five samples;
  only if that fails, redesign the degradation as a true telephone channel or
  a stronger explicit cutoff condition.
```

Machine-readable channel analysis:

```text
/data/AuK_g2_restoration_pilot_50_20261007/channel_restoration_analysis.json
```
