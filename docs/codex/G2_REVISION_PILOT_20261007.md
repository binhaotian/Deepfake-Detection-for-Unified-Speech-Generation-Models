# G2 Restoration Revision Pilot

Date: 2026-10-07
Experiment: `EXP-012`
Status: generated and automatically validated; human A/B decision recorded.

## Purpose

Revise two issues identified in the initial G2 listening review:

1. Full Enhancement at 0 dB SNR was noise-dominated and perceptually masked
   the reverberation.
2. The original channel-restoration prompt did not materially restore missing
   high-frequency content.

The revision uses the same 10 RT60-diverse pairs as the first listening subset.

## A/B conditions

### Full Enhancement

```text
A: existing 0 dB noise + reverberation input/output
B: same RIR and noise identity at 10 dB SNR, newly generated output
```

Measured SNR for the revised inputs is 10.000000 dB within PCM numerical error.

### Channel Restoration

The bandwidth-limited input is unchanged. Only the natural-language
instruction changes.

Old instruction:

```text
Repair the telephone effect and restore natural, clear speech.
```

New instruction:

```text
Perform speech super-resolution on this bandwidth-limited recording.
Restore the missing high-frequency speech content, preserve the speaker
identity and spoken words, and output audio of the same length.
```

Prompt provenance clarification (recorded after inspecting the full official
Prompt Enhancer configuration): the paper itself does not publish an exact
per-benchmark sentence, and the short public Cookbook uses the generic
`Repair the telephone effect...` quality-restoration template. However,
`/root/AuK/src/auk/infer/pe.config.yaml` contains a dedicated official
`improve_quality/bandwidth_extension` template that explicitly asks AuK to
recover missing high-frequency content and output wideband speech. The revised
instruction closely follows that official task route. The old instruction
instead described a telephone effect even though the pilot input was only
polyphase bandwidth limitation, without a full telephone-channel simulation.

## Generation

```text
AuK Base, 24 kHz, BF16, NFE=32, CFG=2.0, sway=-1.0, batch=4
Full Enhancement 10 dB outputs: 10/10
Explicit channel-prompt outputs: 10/10
Validation errors: 0
All WAV files independently decoded: 100/100
```

## Channel spectral diagnostic

For these 10 pairs:

```text
old prompt median high-band gain over input:       +2.30 dB
explicit prompt median high-band gain over input:  +50.60 dB

old prompt median recovered-missing-energy ratio:       0.000119
explicit prompt median recovered-missing-energy ratio:  8.137260
```

The explicit prompt clearly causes high-frequency generation, unlike the old
prompt. However, the median high-band amount is approximately 8.1 times the
amount missing relative to the anchor. This is not evidence of correct
restoration: it may indicate excessive high-frequency hallucination, hiss, or
other artifacts. Human listening is required before selecting the prompt.

## Artifacts

```text
/data/AuK_g2_revision_pilot_10_20261007/
/data/AuK_g2_revision_pilot_10_20261007/comparison_metrics.json
/data/AuK_g2_revision_pilot_10_20261007.zip
/data/AuK_g2_revision_listening_10_20261007.zip
/data/AuK_g2_revision_packages_20261007.sha256
```

Per pair:

```text
anchor.wav
full_enhancement_snr10/
  full_snr0_input.wav
  auk_full_snr0.wav
  full_snr10_speech.wav
  full_snr10_noise.wav
  full_snr10_input.wav
  auk_full_snr10.wav
channel_super_resolution/
  channel_input.wav
  auk_channel_old_prompt.wav
  auk_channel_explicit_prompt.wav
revision_metadata.json
```

## Listening decisions

For Full Enhancement, decide whether 10 dB preserves audible contributions
from both noise and reverberation while remaining a meaningful challenge.

For Channel Restoration, compare the old conservative output with the explicit
super-resolution output. Listen specifically for:

```text
natural high-frequency consonants versus synthetic hiss
speaker/timbre drift
word changes
metallic or vocoder artifacts
over-brightness
boundary artifacts
```

## Human review outcome

Recorded on 2026-10-07:

- The 10 dB Full Enhancement input (`03_full_snr10_input.wav`) is the more
  reasonable main experimental condition. The original 0 dB input is too
  noise-dominated and tends to obscure the reverberation component.
- The explicit speech-super-resolution output
  (`07_auk_channel_explicit_prompt.wav`) performs the requested operation much
  more clearly than the old Cookbook-style prompt output.
- The explicit channel output can contain strange noise and appears capable of
  over-restoring or hallucinating high-frequency energy. It is therefore a
  successful task-invocation test, but not yet an approved final production
  prompt.
