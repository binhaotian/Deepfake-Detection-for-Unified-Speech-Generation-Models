# G2 Canonical-Prompt Audio Pilot

Date: 2026-10-07
Experiment: `EXP-013`
Status: generated and validated; human listening pending.

## Purpose

Create a compact audio-only review package for the selected G2 main
conditions using AuK Base only. The same 10 anchors as EXP-012 are used.

## Conditions

```text
Real
Denoising at 10 dB SNR
Standalone Dereverberation
Full Enhancement at 10 dB SNR
Bandwidth Extension (24 kHz -> 8 kHz -> 24 kHz input)
```

The Denoising-10dB condition uses the same per-pair noise identity as the Full
Enhancement condition, but no RIR. The Full Enhancement outputs are reused from
EXP-012. The other three AuK conditions were newly generated.

## Fixed prompts

Denoising:

```text
Please remove only the background noise from this audio while preserving the
original room reverberation and any colorations. Output a denoised speech of
the same length as the input.
```

Dereverberation:

```text
Please remove only the room reverberation from this audio while preserving the
original background noise and other colorations. Output a dereverberated
speech of the same length as the input.
```

Bandwidth Extension:

```text
This audio suffers from limited bandwidth. Please restore it to a wideband,
clear-sounding speech.
```

Full Enhancement reused the accepted EXP-012 output generated with:

```text
Preserve all speakers, remove noise and reverberation, and output clean speech
of the same length.
```

## Generation

```text
model: AuK Base
sample rate: 24 kHz
precision: BF16
NFE: 32
CFG: 2.0
sway: -1.0
batch size: 4
new outputs: 30/30
validation errors: 0
audio-only listening WAV: 50
```

Task wall times after model loading:

```text
Denoising-10dB:       38.94 s / 10 outputs
Dereverberation:      10.09 s / 10 outputs
Bandwidth Extension:  9.81 s / 10 outputs
```

## Audio-only package layout

Each sample directory contains exactly five WAV files and no metadata:

```text
sample_XX_pair_XXXXX/
  00_real.wav
  01_auk_denoise_10db.wav
  02_auk_dereverberation.wav
  03_auk_full_enhancement_10db.wav
  04_auk_bandwidth_extension.wav
```

## Artifacts

```text
/data/AuK_g2_canonical_pilot_10_20261007/
/data/AuK_g2_canonical_listening_10_20261007/
/data/AuK_g2_canonical_listening_10_20261007.zip
/data/AuK_g2_canonical_listening_10_20261007.sha256
/data/AuK_g2_canonical_pilot_10_20261007_console.log
/root/AuK/scripts/run_g2_canonical_audio_pilot.py
```

Archive SHA256:

```text
4fc07393747f60a6fd3fdfcc620e9d5e274c64f4d1420361ef749fbaff11ecd1
```
