# Matched-set v0

`matched_v0/manifest.tsv` is the reproducible input set for the first AuK
experiment.  It contains 100 ASVspoof2019 LA-eval bona fide anchors, one per
pair, with the corresponding VCTK transcript.

For each pair:

- `anchor.wav`: original bona fide recording (the Real condition).
- `tts_reference.wav`: an independent ASV evaluation enrollment recording of
  the same VCTK speaker (not the anchor itself).
- `se_noise.wav` and `se_input.wav`: the exact noise and the anchor+noise
  mixture.  The mixture is RMS-controlled to 10 dB SNR.
- `tse_interferer.wav` and `tse_input.wav`: a different, preferentially same-
  gender enrollment speaker and the fully overlapping anchor+interferer
  mixture.  The mixture is RMS-controlled to 0 dB SIR.

All files are mono 24 kHz PCM WAV, matching AuK's native inference rate.  The
original 16 kHz FLAC paths, selected enrollment IDs, noise path, requested and
measured ratios, and deterministic seed are recorded in the manifest.  The
`auk_tts.wav`, `auk_se.wav`, and `auk_tse.wav` files are intentionally not
created by this preparation step: they must be generated later with one fixed
AuK checkpoint, instruction, and sampling configuration.

## Noise provenance

The v0 noise subset is 24 ESC-50 environmental recordings (rain,
thunderstorm, wind, sea waves, water drops, and engine; four clips per type),
downloaded by `scripts/download_noise_subset.py`.  The complete MUSAN archive
is approximately 11 GB and is not needed for this pilot.  Each selected clip
is deterministically cropped or looped to the anchor duration; this is noted
by the source path and can be replaced by a larger corpus in a later version.

## Reproduction

```bash
python scripts/download_noise_subset.py
python scripts/prepare_matched_set.py
```

The preparation script does not use AASIST scores to select or discard any
sample.  `se_input.wav` and `tse_input.wav` are inputs to AuK, not additional
Real examples.

## Current model/data inventory

For the current audit of AuK outputs and the three planned specialist models
(CosyVoice3, MossFormerGAN-SE, and DAE-TSE), see
[`MODEL_DATA_INVENTORY.md`](MODEL_DATA_INVENTORY.md).
