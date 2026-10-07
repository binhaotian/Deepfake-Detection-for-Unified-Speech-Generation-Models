# Experiment Ledger

Last curated: 2026-10-07. This file is append-only in spirit: correct factual errors, but do not erase superseded experiments. Use `superseded_by` and preserve the original result.

## Index

| ID | Status | Experiment | Primary result |
|---|---|---|---|
| EXP-001 | completed | AuK 5,369-pair matched dataset | 48,321 WAV; 16,107 AuK outputs |
| EXP-002 | completed | Initial frozen-encoder four-class probes | TTS easy; SE/TSE main confusion |
| EXP-003 | completed | TTS V2/V3 duration/boundary control | Wav2Vec2 changes; HuBERT stable |
| EXP-004 | completed | ASVspoof2019 LA encoder baseline | strong original benchmark separability |
| EXP-005 | completed | Specialist TTS/SE/TSE deployment and generation | 5,369 outputs per specialist |
| EXP-006 | completed | AuK versus specialist task/four-class probes | specialist boundary stronger |
| EXP-007 | completed | AuK Content Editing construction/generation | 3,734 accepted/generated pairs |
| EXP-008 | completed | AuK five-class CE probe | 75.55% W2V2; 92.97% HuBERT |
| EXP-009 | completed | CosyEdit specialist CE generation | 3,734/3,734 validated |
| EXP-010 | completed | Specialist five-class CE probe | 83.00% W2V2; 97.53% HuBERT |
| EXP-011 | completed, review pending | AuK G2 restoration 50-anchor pilot | 150/150 generated; automatic validation clean |
| EXP-012 | completed, review pending | G2 Full-SNR/channel-prompt revision pilot | 20/20 generated; explicit prompt strongly increases high band |

## EXP-001 — AuK anchor-centric matched dataset

```yaml
status: completed
completed: 2026-09-23
pair_count: 5369
speaker_count: 48
source: ASVspoof2019 LA eval bona fide target-speaker trials
tasks: [real, auk_tts, auk_se, auk_tse]
```

Purpose: construct controlled task variants around one real anchor.

Per pair:

```text
anchor.wav
tts_reference.wav
auk_tts.wav
se_noise.wav
se_input.wav
auk_se.wav
tse_interferer.wav
tse_input.wav
auk_tse.wav
```

Core conditions:

```text
TTS: independent same-speaker reference + anchor transcript
SE:  anchor + environmental noise, SNR = 0 dB
TSE: anchor + different-speaker overlap, SIR = 0 dB
AuK Base: 24 kHz, BF16, NFE=32, CFG=2.0, sway=-1.0, batch=4
```

Outputs:

```text
/root/AuK/data/matched_full_5369_snr0_batch4_generated/
/root/AuK/data/AuK_matched_full_5369_documentation/
```

Authoritative records:

```text
/root/AuK/docs/data_generation_logic.md
/root/AuK/data/matched_full_5369_snr0_batch4_generated/manifest.tsv
/root/AuK/data/matched_full_5369_snr0_batch4_generated/generation_config.json
```

Result: 5,369 complete pairs; 48,321 WAV; 16,107 AuK-generated task outputs.

## EXP-002 — Initial four-class frozen-encoder probes

```yaml
status: completed
date_range: 2026-09-24 through 2026-09-29
classes: [Real, TTS, SE, TSE]
split: speaker-disjoint, 34/7/7 speakers
pairs: 3637/944/788 train/validation/test
probe: StandardScaler + multinomial LogisticRegression
```

Encoders and embeddings:

```text
Wav2Vec2 Large: 1024-d
HuBERT Large:   1024-d
Whisper Large-v3: 1280-d, historical only
```

Initial V1 test accuracy:

| Encoder | Accuracy | Macro-F1 |
|---|---:|---:|
| Wav2Vec2 Large | 82.17% | 82.19% |
| HuBERT Large | 94.00% | 94.01% |
| Whisper Large-v3 | 91.05% | 91.12% |

Observation: errors concentrate in SE/TSE. TTS is nearly perfectly separable under HuBERT and Whisper.

Reports:

```text
/root/AuK/data/probe_results/wav2vec2-large-fourclass/report.json
/root/AuK/data/probe_results/hubert-large-fourclass/report.json
/root/AuK/data/probe_results/whisper-large-fourclass/report.json
```

Limitation: these are general-purpose speech representations, not dedicated anti-spoof detector outputs.

## EXP-003 — TTS duration and boundary shortcut audit

```yaml
status: completed
completed_by: 2026-09-29
variants: [V2_active_duration, V3_text_duration]
changed_component: AuK TTS duration/boundary construction only
```

V2:

```text
target duration = ceil_to_20ms(anchor active duration + 0.28 s)
generated active speech is aligned into the anchor active region
anchor leading/trailing boundary is restored
```

V3:

```text
target duration = English UTF-8 byte count * 0.0656 s
minimum/short-text correction and 20 ms quantization
same post-generation active-region alignment as V2
```

Test accuracy:

| Encoder | V1 | V2 | V3 |
|---|---:|---:|---:|
| Wav2Vec2 | 82.17% | 76.81% | 77.63% |
| HuBERT | 94.00% | 93.65% | 93.59% |

Conclusion: duration/boundary control affects Wav2Vec2 but does not explain HuBERT separability. V2 and V3 are very similar. SE/TSE remains the persistent confusion.

Authoritative documentation:

```text
/root/AuK/docs/experiment_log_v2_v3_and_asvspoof.md
/root/AuK/data/tts_full_v2_v3_5369/
```

## EXP-004 — ASVspoof2019 LA binary encoder baseline

```yaml
status: completed
completed_by: 2026-09-29
task: bona_fide versus spoof
split: official ASVspoof train/dev/eval
probe: class-balanced binary LogisticRegression
```

Eval performance:

| Encoder | Balanced accuracy | ROC-AUC | EER |
|---|---:|---:|---:|
| Wav2Vec2 | 94.91% | 0.9887 | 4.89% |
| HuBERT | 97.10% | 0.9963 | 2.73% |

Purpose: establish the relative strength of the same frozen representations on the original ASVspoof task. Do not use this result to claim direct transfer to AuK.

Reports:

```text
/root/AuK/data/probe_results/asvspoof_wav2vec2/report.json
/root/AuK/data/probe_results/asvspoof_hubert/report.json
```

## EXP-005 — Specialist TTS/SE/TSE baselines

```yaml
status: completed
completed: 2026-09-30
pair_count: 5369
models: [CosyVoice3, MossFormerGAN-SE-16K, DAE-TSE]
```

Input matching:

```text
CosyVoice3: same tts_reference.wav + target transcript
MossFormerGAN-SE: same se_input.wav
DAE-TSE: same tse_input.wav + normalized anchor text cue
```

Output counts:

```text
CosyVoice3-TTS: 5369/5369
MossFormer-SE:  5369/5369
DAE-TSE:        5369/5369
```

Paths:

```text
/data/specialist_models/
/data/venvs/
/data/specialist_outputs/
/data/specialist_experiment_documentation_20260930/
```

Important notes:

- CosyVoice3 uses official `inference_cross_lingual` because the independent enrollment reference lacks a reliable reference transcript.
- MossFormer outputs 16 kHz audio; 21 over-range outputs were uniformly attenuated to peak 0.98 and logged.
- DAE-TSE is content-cued and matches AuK's cue type; 605 non-fatal isolated-jump warnings were recorded, with no failed files.

## EXP-006 — AuK versus specialist task boundaries

```yaml
status: completed
completed: 2026-09-30
embedding_shape: [5369, 7, 1024]
active_encoders: [Wav2Vec2 Large, HuBERT Large]
```

Seven streams:

```text
[Real, AuK-TTS, CosyVoice3-TTS, AuK-SE, MossFormer-SE, AuK-TSE, DAE-TSE]
```

Specialist-only three-class accuracy:

| Encoder | TTS/SE/TSE accuracy |
|---|---:|
| Wav2Vec2 | 92.26% |
| HuBERT | 99.79% |

Four-class accuracy with Real:

| Encoder | AuK | Specialist | Specialist gap |
|---|---:|---:|---:|
| Wav2Vec2 | 82.14% | 86.96% | +4.82 pp |
| HuBERT | 94.13% | 98.48% | +4.35 pp |

Interpretation: the direction is consistent with weaker AuK task/real boundaries, but the specialist comparison confounds task and model identity.

Authoritative archive:

```text
/data/specialist_experiment_documentation_20260930/README.md
/data/probe_results/matched_fourclass_auk_wav2vec2_mi5000/report.json
/data/probe_results/matched_fourclass_specialist_wav2vec2_mi5000/report.json
/data/probe_results/matched_fourclass_auk_hubert_mi5000/report.json
/data/probe_results/matched_fourclass_specialist_hubert_mi5000/report.json
```

## EXP-007 — AuK single-word Content Editing dataset

```yaml
status: completed
date_range: 2026-10-01 through 2026-10-06
source_pairs: 5369
unique_source_texts: 3857
accepted_text_groups: 2708
accepted_pair_rows: 3734
```

Text pipeline:

```text
GPT candidate generation
-> independent GPT review
-> deterministic exact-one-word replacement validation
-> propagation from unique text group to pair rows
```

Statuses:

```text
accepted:     3734 pair rows
rejected:     1459 pair rows
needs_review: 176 pair rows
```

AuK generated `auk_content_edit.wav` for all 3,734 accepted pairs.

Paths:

```text
/root/AuK/data/content_edit_full_gpt_20261001/
/data/AuK_content_edit_full_20261001/
/data/content_edit_research_archive_20261006/
```

## EXP-008 — AuK five-class Content Editing probe

```yaml
status: completed
completed: 2026-10-06
classes: [Real, TTS, SE, TSE, Content Editing]
pair_count: 3734
split_pairs: 2533/635/566
split_speakers: 34/7/7
```

Results:

| Encoder | Accuracy | Macro-F1 |
|---|---:|---:|
| Wav2Vec2 | 75.55% | 75.47% |
| HuBERT | 92.97% | 92.96% |

Key errors:

```text
Wav2Vec2: Real -> CE 101; CE -> Real 98; SE/TSE remain confused
HuBERT: CE 541/566 correct; SE/TSE remain the primary confusion
```

Reports:

```text
/data/probe_results/auk_fiveclass_content_edit_wav2vec2/report.json
/data/probe_results/auk_fiveclass_content_edit_hubert/report.json
```

## EXP-009 — CosyEdit specialist Content Editing

```yaml
status: completed
completed: 2026-10-06
model: CosyEdit
official_repo_commit: a2aea98dade9ae6bc3d22c80dd5448b10a04a57a
checkpoint: CJY/CosyEdit
pair_count: 3734
output_sample_rate: 22050
```

Interface:

```text
inference_edit(target_text, original_text, original_speech)
```

Execution:

- Four independent model workers with non-overlapping shards.
- Resumable/skip-existing output behavior.
- Per-pair seed and result recorded in the manifest.

Validation:

```text
generated: 3734/3734
invalid/non-finite/empty: 0
duration range: 0.557–5.991 seconds
mean duration: 2.720 seconds
clipping fraction >1%: 0 files
```

Paths:

```text
/data/CosyEdit/content_edit_full_3734_20261006/
/data/CosyEdit/content_edit_full_3734_20261006.zip
```

## EXP-010 — Specialist five-class Content Editing probe

```yaml
status: completed
completed: 2026-10-06
classes: [Real, TTS, SE, TSE, Content Editing]
systems: [Real, CosyVoice3, MossFormerGAN-SE, DAE-TSE, CosyEdit]
pair_count: 3734
split_pairs: 2533/635/566
split_speakers: 34/7/7
```

Results:

| Encoder | Accuracy | Macro-F1 | AuK accuracy | Specialist gap |
|---|---:|---:|---:|---:|
| Wav2Vec2 | 83.00% | 83.05% | 75.55% | +7.46 pp |
| HuBERT | 97.53% | 97.53% | 92.97% | +4.56 pp |

Wav2Vec2 confusion matrix, order `[Real, TTS, SE, TSE, CE]`:

```text
[[440,  2, 66, 25, 33],
 [  3,546,  1,  3, 13],
 [ 48,  1,437, 56, 24],
 [ 22,  0, 62,461, 21],
 [ 32, 13, 33, 23,465]]
```

HuBERT confusion matrix:

```text
[[524,  1, 36,  3,  2],
 [  0,566,  0,  0,  0],
 [  4,  0,561,  1,  0],
 [  1,  0,  0,563,  2],
 [  1,  1, 16,  2,546]]
```

Reports:

```text
/data/probe_results/specialist_fiveclass_cosyedit_wav2vec2/report.json
/data/probe_results/specialist_fiveclass_cosyedit_hubert/report.json
/root/AuK/reports/specialist_content_edit_experiment_20261006_README.md
```

## EXP-011 — AuK G2 restoration pilot

```yaml
status: completed; human listening review pending
completed: 2026-10-07
pair_count: 50
speaker_count: 48
tasks: [dereverberation, full_enhancement, narrowband_channel_restoration]
outputs_per_task: 50
total_auk_outputs: 150
```

Construction:

```text
Dereverberation:
  OpenSLR SLR28 medium-room RIR, estimated RT60 0.452--0.644 s
Full enhancement:
  (anchor * same RIR) + existing pair noise at 0 dB SNR
Channel restoration:
  24 kHz -> 8 kHz -> 24 kHz polyphase bandwidth limitation
```

AuK configuration:

```text
AuK Base, 24 kHz, BF16, NFE=32, CFG=2.0, sway=-1.0, batch=4
```

Result:

```text
input construction validation errors: 0
generated: 150/150
generation validation errors/warnings: 0/0
independently decoded WAV: 470/470
total generation wall time: 227.51 s
peak reserved GPU memory: 18.81 GiB
```

Artifacts:

```text
/data/AuK_g2_restoration_pilot_50_20261007/
/data/AuK_g2_restoration_pilot_50_20261007.zip
/data/AuK_g2_restoration_listening_10_20261007.zip
/root/AuK/docs/codex/G2_RESTORATION_PILOT_20261007.md
```

## EXP-012 — G2 restoration revision A/B pilot

```yaml
status: completed; human A/B review recorded
completed: 2026-10-07
pair_count: 10
new_outputs: 20
conditions: [full_enhancement_snr10, channel_explicit_super_resolution_prompt]
```

Result:

```text
Full Enhancement measured SNR: 10.000000 dB
Old channel prompt median high-band gain: +2.30 dB
Explicit channel prompt median high-band gain: +50.60 dB
Explicit prompt median high-band recovery ratio: 8.14
Validation errors: 0
```

Human review on 2026-10-07:

```text
Full Enhancement: select the revised 10 dB condition as the more reasonable
main experimental setting; retain 0 dB only as a stress control.

Channel Restoration: the explicit speech-super-resolution prompt performs the
intended operation more clearly than the old prompt, but occasional strange
noise is audible. Together with the spectral overshoot, this prevents automatic
promotion of the current explicit prompt to full-scale production.
```

Artifacts:

```text
/data/AuK_g2_revision_pilot_10_20261007/
/root/AuK/docs/codex/G2_REVISION_PILOT_20261007.md
```

## EXP-013 — G2 canonical-prompt audio pilot

```yaml
status: completed; human listening pending
completed: 2026-10-07
pair_count: 10
new_outputs: 30
listening_wavs: 50
conditions: [real, denoise_10db, dereverberation, full_enhancement_10db, bandwidth_extension]
```

AuK configuration:

```text
AuK Base, 24 kHz, BF16, NFE=32, CFG=2.0, sway=-1.0, batch=4
```

Result:

```text
Denoising-10dB: 10/10 newly generated
Dereverberation with full canonical prompt: 10/10 newly generated
Bandwidth Extension with official soft template: 10/10 newly generated
Full Enhancement-10dB: 10 outputs reused from validated EXP-012
Real: 10 anchors
validation errors: 0
archive files: 50 WAV and no non-audio files
```

Artifacts:

```text
/data/AuK_g2_canonical_pilot_10_20261007/
/data/AuK_g2_canonical_listening_10_20261007.zip
/root/AuK/docs/codex/G2_CANONICAL_AUDIO_PILOT_20261007.md
```

## Adding a new entry

Copy `docs/codex/EXPERIMENT_ENTRY_TEMPLATE.md`, assign the next `EXP-xxx`, and include the machine-readable report/config/manifest paths. Failed and abandoned runs also belong in this ledger if they affect interpretation or leave artifacts.
