# Experiment Log: V2/V3 TTS Control and ASVspoof Encoder Probes

最后整理：2026-09-28

本文档汇总从 V2/V3 TTS 时长控制实验开始，到 Wav2Vec2/HuBERT frozen
representation probe 以及 ASVspoof2019 LA 对照实验的设置、产物和当前结论。

## 1. Experiment scope

目标是检查同一个 AuK unified generation model 的不同任务输出，在通用语音
encoder 的表征空间中是否具有可恢复的任务/真实性差异。当前使用的类别为：

```text
real = ASVspoof LA eval bona fide anchor
tts  = AuK-TTS
se   = AuK speech enhancement
tse  = AuK target speaker extraction
```

TTS V2/V3 只改变 TTS 目标时长控制方法；anchor、speaker、reference、text、
SE/TSE 输出和 probe 划分保持一致。

## 2. V2/V3 generation

### 2.1 Shared source and model settings

- Source: `data/matched_full_5369_snr0_batch4_generated`
- Pair count: 5,369
- Target speakers: 48
- TTS reference: fixed independent same-speaker ASV enrollment recording
- Transcript: anchor VCTK transcript
- AuK checkpoint: `ckpts/AuK/auk_base.safetensors`
- Model: AuK Base
- Sampling dtype: BF16
- NFE: 32
- CFG strength: 2.0
- Sway sampling coefficient: -1.0
- Generation batch size: 4
- Native generation sample rate: 24 kHz
- Prompt:

```text
Say the following with the same voice: '{anchor transcript}'
```

The raw output is saved first. Its detected active region is then placed at the
anchor's active-region position, preserving the anchor's leading/trailing boundary.
The raw output and aligned output are both retained.

### 2.2 V2: active duration plus boundary budget

- Variant: `v2_active_duration`
- Duration mode: `active_duration_plus_boundary`
- Active boundary threshold: -40 dB relative to utterance peak
- Boundary budget: 0.28 s
- Target duration:

```text
ceil_to_20ms(anchor_active_duration + 0.28 s)
```

The model is therefore given a duration based on the anchor's effective speech
duration plus an explicit boundary allowance.

### 2.3 V3: text-duration prior

- Variant: `v3_text_duration`
- Duration mode: `text_duration_plus_boundary` (historical name; no extra 0.28 s is
  added to the generation target)
- Text duration prior:

```text
English UTF-8 byte count * 0.0656 s
```

For text shorter than 10 bytes, the value is divided by 0.3; the final estimate is
clamped to at least 0.3 s and quantized to a 20 ms grid.

The generated boundary is removed during alignment and replaced using the anchor's
leading/trailing active-boundary positions. Thus V3 delegates the internal timing to
the text-duration prior while preserving the same external anchor boundary.

### 2.4 Generation records

The following files are authoritative:

- `data/tts_full_v2_v3_5369/v2_active_duration/generation_config.json`
- `data/tts_full_v2_v3_5369/v3_text_duration/generation_config.json`
- `data/tts_full_v2_v3_5369/v2_active_duration/manifest.tsv`
- `data/tts_full_v2_v3_5369/v3_text_duration/manifest.tsv`
- `data/tts_full_v2_v3_5369/pilot_summary.json`
- `data/AuK_matched_full_5369_documentation/metadata/generation_report.json`

Each V2/V3 manifest records pair identity, speaker, transcript, target/reference IDs,
seed, target duration, text-duration estimate, anchor active duration, raw TTS active
duration, and aligned duration. The generation report records file integrity, peak,
clipping fraction, SHA-256, and validation warnings.

## 3. Frozen encoder extraction

Encoders:

- `encoders/wav2vec2-large-960h-lv60-self`
- `encoders/hubert-large-ll60k`

Shared extraction settings:

- Audio converted to mono 16 kHz
- Encoder frozen and run in inference mode
- CUDA inference with FP16 autocast
- Last hidden state mean pooled over time
- Hidden dimension: 1,024
- Four classes: `real`, `tts`, `se`, `tse`
- 5,369 pairs per variant

Extraction configs:

- `data/embeddings/tts_v2_wav2vec2/config.json`
- `data/embeddings/tts_v3_wav2vec2/config.json`
- `data/embeddings/tts_v2_hubert/config.json`
- `data/embeddings/tts_v3_hubert/config.json`

## 4. AuK four-class linear probes

Classifier:

```text
StandardScaler + multinomial LogisticRegression
C = 1.0
max_iter = 1500
seed = 20260925
```

Split protocol:

- Speaker-disjoint
- 34 train speakers
- 7 validation speakers
- 7 test speakers
- Pair counts: 3,637 / 944 / 788

Reports:

- `data/probe_results/tts_v2_wav2vec2/report.json`
- `data/probe_results/tts_v3_wav2vec2/report.json`
- `data/probe_results/tts_v2_hubert/report.json`
- `data/probe_results/tts_v3_hubert/report.json`

Test-set accuracy and macro F1:

| Encoder | V2 accuracy | V2 macro F1 | V3 accuracy | V3 macro F1 |
|---|---:|---:|---:|---:|
| Wav2Vec2 | 0.7681 | 0.7683 | 0.7763 | 0.7754 |
| HuBERT | 0.9365 | 0.9367 | 0.9359 | 0.9361 |

Test-set confusion matrices use row/column order `[real, tts, se, tse]`.

Wav2Vec2 V2:

```text
[[679, 23,  37,  49],
 [ 11, 683, 41,  53],
 [ 26, 38, 569, 155],
 [ 37, 60, 201, 490]]
```

Wav2Vec2 V3:

```text
[[691, 17,  36,  44],
 [ 13, 699, 34,  42],
 [ 27, 36, 576, 149],
 [ 38, 66, 203, 481]]
```

HuBERT V2:

```text
[[780,  1,  4,  3],
 [  0, 781, 3,  4],
 [  0,  2, 706, 80],
 [  1,  4, 98, 685]]
```

HuBERT V3:

```text
[[778,  0,  4,  6],
 [  0, 779, 1,  8],
 [  0,  2, 705, 81],
 [  1,  2, 97, 688]]
```

Current conclusion: V2 and V3 are nearly indistinguishable at the probe level. The
largest persistent confusion is `SE <-> TSE`; HuBERT separates `real` and `TTS`
particularly well.

## 5. V1 comparison

The original TTS generation corresponds to the earlier generated matched-set outputs.
Its four-class reports were rerun with the same `max_iter=1500` probe setting:

- `data/probe_results/wav2vec2-large-fourclass/report.json`
- `data/probe_results/hubert-large-fourclass/report.json`

Test-set accuracy:

| Encoder | V1 | V2 | V3 |
|---|---:|---:|---:|
| Wav2Vec2 | 0.8217 | 0.7681 | 0.7763 |
| HuBERT | 0.9400 | 0.9365 | 0.9359 |

Interpretation: V1 to V2/V3 changes Wav2Vec2 separability by several percentage
points, while HuBERT changes by less than half a percentage point. V2 and V3 remain
close to each other.

## 6. ASVspoof2019 LA baseline

Purpose: establish how the same frozen encoders perform on the original benchmark's
binary task before interpreting AuK task identity.

Protocol:

- Train: `ASVspoof2019.LA.cm.train.trn.txt`
- Dev: `ASVspoof2019.LA.cm.dev.trl.txt`
- Eval: `ASVspoof2019.LA.cm.eval.trl.txt`
- Labels: `bonafide` vs `spoof`
- Classifier: StandardScaler + class-balanced binary LogisticRegression
- `max_iter=1500`, seed 20260925
- Batch size 8 for embedding extraction

Reports:

- `data/probe_results/asvspoof_wav2vec2/report.json`
- `data/probe_results/asvspoof_hubert/report.json`

Eval confusion matrices, row/column order `[bonafide, spoof]`:

Wav2Vec2:

```text
[[ 6885,   470],
 [ 2419, 61463]]
```

HuBERT:

```text
[[ 7258,    97],
 [ 2867, 61015]]
```

Eval balanced accuracy / ROC-AUC / EER:

| Encoder | Balanced accuracy | ROC-AUC | EER |
|---|---:|---:|---:|
| Wav2Vec2 | 0.9491 | 0.9887 | 0.0489 |
| HuBERT | 0.9710 | 0.9963 | 0.0273 |

## 7. Important comparability note

The AuK four-class extraction uses an unmasked mean over the encoder last hidden state.
The ASVspoof baseline uses an attention-mask-aware mean because it batches variable-length
trials with padding. The two evaluations are therefore internally consistent within each
experiment, but their absolute scores should not be treated as a perfectly controlled
cross-experiment comparison until AuK extraction is rerun with the same mask-aware
pooling implementation.

## 8. What is recorded and what is not

Recorded in machine-readable form:

- Dataset construction and selection logic
- Per-pair metadata and random seeds
- Generation model/checkpoint/sampling settings
- V2/V3 duration rules and active-boundary measurements
- Output validation, clipping warnings and SHA-256 hashes
- Encoder model, sample rate, pooling, dtype and device
- Probe split, seed, class labels, confusion matrices and metrics

Not yet centralized or fully recorded:

- One single chronological experiment journal linking all runs
- Exact shell command lines used for every historical run
- Python/package versions and GPU driver/CUDA snapshot
- Saved per-sample linear-probe logits or predicted labels
- A unified analysis report with all matrices and interpretation

The machine-readable files are sufficient to reproduce the current metrics, but the
missing items above should be added before treating this as a final paper-quality audit
trail.
