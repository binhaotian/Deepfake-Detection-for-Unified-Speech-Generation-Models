# Research Brief: Unified Speech Models and Deepfake Task Boundaries

Last curated: 2026-10-07.

## 1. Motivation

Unified speech generation/editing models such as AuK use one model and heavily shared components to perform tasks that were traditionally handled by separate systems. AuK supports TTS, speech content editing, acoustic/paralinguistic editing, enhancement, separation, and target speaker extraction through a shared instruction interface.

This creates a forensic question. TTS, restoration, extraction, and editing have different semantic meanings, but their outputs may share a representation space, generation backbone, latent model, codec/VAE, and decoder. The project asks whether those shared components weaken task boundaries in representations relevant to deepfake detection or attribution.

## 2. Current primary question

The current defensible formulation is:

> Compared with a pipeline in which separate specialist models perform TTS, SE, TSE, and CE, do outputs from one unified model (AuK) have weaker task/real boundaries under the same frozen representation and probe protocol?

"Weaker boundary" currently means lower held-out speaker-disjoint task-classification performance, together with structured confusion between particular classes. It does not mean chance-level performance is required.

## 3. Deepfake-oriented questions that remain distinct

Do not collapse these into one claim:

### Q-A: Task attribution

Given an output, can a representation/classifier recover whether it came from TTS, SE, TSE, or CE?

### Q-B: Source authenticity

Can a detector distinguish human-origin speech from generated or identity/content-manipulated speech?

### Q-C: Processing status

Can a detector recognize that audio was processed by a neural system, even when speaker identity and linguistic content remain authentic?

### Q-D: Generator attribution/shared fingerprint

Do different AuK tasks share a model-specific fingerprint, and is that fingerprint stronger than task-specific differences?

The completed frozen-encoder linear probes primarily address Q-A. They provide indirect evidence relevant to Q-D, but they do not directly answer Q-B or establish dedicated deepfake-detector behavior.

## 4. Current task taxonomy

### TTS

- Input: independent same-speaker reference plus anchor transcript.
- Output: speech newly synthesized by AuK or CosyVoice3.
- Speaker/content are intended to match the anchor, but there is no waveform-level real counterpart.
- This is the strongest conventional spoof-like condition.

### Speech Enhancement (SE)

- Input: real anchor plus environmental noise at controlled SNR.
- Output: restored speech from AuK or MossFormerGAN-SE.
- Intended speaker identity and linguistic content are unchanged.
- Best interpreted as benign neural processing unless the research protocol explicitly defines all generated waveforms as spoof.

### Target Speaker Extraction (TSE)

- Input: real anchor fully overlapped with another speaker at controlled SIR.
- Output: extracted target speech from AuK or DAE-TSE.
- Target identity/content are intended to remain unchanged.
- AuK and DAE-TSE are both content-cued in the current comparison.

### Content Editing (CE)

- Input: real anchor plus an instruction replacing exactly one word.
- Output: mostly preserved speech with one content substitution.
- This changes linguistic authenticity while attempting to preserve speaker and surrounding acoustics.
- AuK and CosyEdit are compared on the same 3,734 accepted text/pair conditions.

### Not currently in the main experiment

- Voice/speaker conversion.
- Timbre editing.
- Emotion/style conversion.
- Accent removal.
- Whisper conversion.
- Acoustic controls such as pitch, speed, or volume.

These are scientifically relevant but must not be added until the authenticity semantics and specialist controls are defined.

## 5. Matched-data design

The main unit is one ASVspoof2019 LA eval bona fide anchor:

```text
same anchor
├── Real
├── TTS: same-speaker reference + anchor transcript
├── SE: anchor + controlled environmental noise -> enhancement
├── TSE: anchor + controlled interferer -> extraction
└── CE: anchor + one-word substitution instruction -> edited speech
```

Controlled factors where possible:

- target speaker;
- anchor transcript/content target;
- source corpus/domain;
- pair identity;
- noise/interferer construction;
- SNR/SIR;
- speaker-disjoint evaluation.

Important asymmetry:

- SE, TSE, and CE directly transform the anchor waveform.
- TTS resynthesizes the transcript from a separate voice reference and has no strict waveform-level anchor counterpart.

Do not describe all tasks as perfectly symmetric paired transformations.

## 6. Current representation protocol

Active encoders:

- Wav2Vec2 Large (`wav2vec2-large-960h-lv60-self`), 1024 dimensions.
- HuBERT Large (`hubert-large-ll60k`), 1024 dimensions.

Current extraction:

```text
mono -> 16 kHz -> frozen encoder -> last hidden state
-> attention-mask-aware temporal mean pooling
```

Probe:

```text
StandardScaler + multinomial LogisticRegression
```

Split:

```text
speaker-disjoint
34 train speakers / 7 validation speakers / 7 test speakers
```

Whisper Large-v3 has historical four-class results but is paused for new experiments.

## 7. Unified versus specialist comparison

Unified group:

```text
Real
AuK-TTS
AuK-SE
AuK-TSE
AuK-CE
```

Specialist group:

```text
Real
CosyVoice3-TTS
MossFormerGAN-SE
DAE-TSE
CosyEdit-CE
```

The same encoder, pooling, pair subset, speaker split, and probe are used within each comparison.

Known confound:

```text
specialist class identity = task identity + model identity + model-specific output pipeline
```

Therefore higher specialist separability can be consistent with more distinct model fingerprints, more distinct task transformations, or both.

## 8. Claim boundaries

### Supported by current results

- AuK task/real boundaries are weaker than the current specialist pipeline under both active encoders.
- AuK SE and TSE form the most stable confusion pair.
- AuK TTS is highly separable under HuBERT.
- AuK CE is substantially confused with Real under Wav2Vec2 but not under HuBERT.

### Not yet supported

- Dedicated deepfake detectors cannot distinguish AuK task outputs.
- Shared AuK parameters are the sole cause of the observed effect.
- AuK has one task-invariant forensic fingerprint.
- Generic encoder linear-probe accuracy is equivalent to real/fake detector EER.
- SE and TSE should semantically be labelled fake.

## 9. Relevant local sources

```text
/root/AuK/AUK.pdf
/root/AuK/What Counts as Real Speech Restoration and Voice Quality Conversion Pose New Challenges to Deepfake Detection.pdf
/root/AuK/README.md
/root/AuK/docs/COOKBOOK.md
```

Historical design documentation:

```text
/root/AuK/docs/data_generation_logic.md
/root/AuK/docs/experiment_log_v2_v3_and_asvspoof.md
/data/specialist_experiment_documentation_20260930/README.md
/data/content_edit_research_archive_20261006/REPORT_CN.md
```

Complete AuK capability inventory and expansion relevance:

```text
/root/AuK/docs/codex/AUK_TASK_INVENTORY.md
```
