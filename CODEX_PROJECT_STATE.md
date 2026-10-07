# Codex Project State

```yaml
last_curated: 2026-10-07
last_machine_audit: see docs/codex/AUTO_INVENTORY.md
repository: /root/AuK
external_artifacts: /data
current_phase: G2 canonical-prompt pilot human listening review
active_design_group: G2 restoration and source selection
next_protocol_item: review the 10-pair five-condition audio-only package before any G2 full-scale generation
active_run: none
git_snapshot_preparation: origin configured and snapshot branch created; not committed or pushed
paused:
  - new Whisper embedding/probe work
  - AuK-Flash task generation for the main comparison
completed_experiment_range: EXP-001 through EXP-013
```

## Current objective

Study whether outputs from different tasks completed by one unified speech model have weaker task boundaries in speech/deepfake representation space than the same tasks completed by separate specialist models.

As of 2026-10-07, the user and advisor decided to broaden the study beyond the existing four AuK task conditions. The current work is to define a hierarchical task taxonomy and generation protocol before launching any new full-scale inference. Proposed roadmap: `docs/codex/AUK_EXPANSION_ROADMAP.md`.

The working taxonomy and discussion order were approved on 2026-10-07. Detailed group protocol planning is tracked in `docs/codex/TASK_GROUP_PROTOCOL_PLAN.md`. The active group is G2 Restoration and Source Selection; the next item is RES-02 Dereverberation.

On 2026-10-07, a rapid first-batch recipe was researched and recorded for
RES-02 dereverberation, RES-03 full enhancement, and RES-04 narrowband channel
restoration. It proposes one common 50-anchor pilot, OpenSLR SLR28 medium-room
RIRs, matched existing 0 dB noise assignments for full enhancement, and a
fixed 24 kHz -> 8 kHz -> 24 kHz bandwidth-limitation condition. The user
accepted these settings on 2026-10-07. The 50-anchor pilot completed on the
same date: 150/150 AuK outputs, zero automatic validation errors/warnings.
Artifact: `/data/AuK_g2_restoration_pilot_50_20261007/`. Human listening review
found that 0 dB Full Enhancement is noise-dominated and that current channel
outputs do not materially restore missing high-frequency content. No G2 task
has been approved for full-scale generation yet.

EXP-012 completed on 2026-10-07 using the same 10 listening pairs. It generated
10 Full Enhancement outputs at 10 dB SNR and 10 channel outputs with an explicit
speech-super-resolution prompt. Automatic validation passed. The explicit
prompt strongly increases high-band energy. Human A/B review selected the 10 dB
Full Enhancement condition as the more reasonable experimental setting and
confirmed that the explicit channel prompt invokes the intended operation much
more clearly than the old Cookbook template. However, the new channel outputs
contain occasional strange noise and may over-generate high-frequency energy,
so the channel prompt is not yet approved for full-scale generation. Listening package:
`/data/AuK_g2_revision_listening_10_20261007.zip`.

Follow-up source inspection found that the official Prompt Enhancer config has
dedicated restoration routes that are more specific than the short Cookbook:
`enhance_speech/cleanup_mode=dereverb`,
`enhance_speech/cleanup_mode=denoise_dereverb`, and
`improve_quality/bandwidth_extension`. The successful explicit channel prompt
closely matches the official bandwidth-extension route; the failed old prompt
described a telephone effect although the constructed input was bandwidth-only.
Standalone dereverberation remains necessary because combined enhancement can
be perceptually dominated by noise removal.

The main task-boundary dataset will use AuK Base only. AuK-Flash is excluded
from the main task classes because changing the model variant would add a
generator/checkpoint confound.

For the new G2 factorial restoration comparison, Denoising will be regenerated
at 10 dB SNR using the same per-pair noise identity as Full Enhancement at
10 dB. The completed 0 dB AuK-SE outputs remain preserved as a stress/historical
condition and are not overwritten.

EXP-013 completed on 2026-10-07. The accepted canonical settings were applied
to the same 10 anchors: Denoising-10dB, standalone Dereverberation, accepted
Full Enhancement-10dB, and the official soft Bandwidth Extension prompt. Thirty
new AuK Base outputs were generated with zero validation errors. The audio-only
review archive contains 10 folders x 5 WAV (`Real` plus four task outputs) and
no metadata files:
`/data/AuK_g2_canonical_listening_10_20261007.zip`.

Current AuK tasks:

```text
TTS
Speech Enhancement (SE)
Target Speaker Extraction (TSE)
Content Editing (CE; one-word substitution)
```

Current specialist controls:

```text
TTS -> CosyVoice3
SE  -> MossFormerGAN-SE-16K
TSE -> DAE-TSE (content-cued)
CE  -> CosyEdit
```

## What is complete

### Dataset and generation

- AuK matched set: 5,369 pairs, 48 target speakers, 9 WAV per pair, 48,321 WAV total.
- AuK outputs: 5,369 each for TTS, SE, and TSE; 16,107 generated outputs.
- Conditions: SE SNR = 0 dB; TSE SIR = 0 dB.
- Specialist outputs: 5,369 each for CosyVoice3-TTS, MossFormerGAN-SE, and DAE-TSE.
- Content Editing text selection: 3,734 accepted pair rows from 2,708 accepted unique-text groups.
- AuK CE: 3,734/3,734 generated.
- CosyEdit CE: 3,734/3,734 generated and validated.

### Representation and probe work

- Historical encoders: Wav2Vec2 Large, HuBERT Large, Whisper Large-v3.
- Active encoders: Wav2Vec2 Large and HuBERT Large only.
- Probe: frozen encoder -> utterance embedding -> `StandardScaler + LogisticRegression`.
- Split: speaker-disjoint, normally 34/7/7 speakers.
- Four-class comparisons completed on all 5,369 pairs.
- Five-class comparisons completed on the 3,734 accepted CE pairs.

### Repository preservation

- GitHub snapshot scope was curated on 2026-10-07 for the user-approved public research repository.
- `RESEARCH_SNAPSHOT.md` records the intended upload and exclusion policy.
- Large datasets, audio, embeddings, checkpoints, archives, and local paper copies are excluded by `.gitignore`.
- The research repository is configured as `origin`; Tencent AuK is retained as fetch-only `upstream`.
- The active branch is `research-snapshot-2026-10-07`; no commit or push has been performed.

## Latest comparable results

### Four classes: `[Real, TTS, SE, TSE]`

| Encoder | AuK accuracy | Specialist accuracy | Gap |
|---|---:|---:|---:|
| Wav2Vec2 Large | 82.14% | 86.96% | +4.82 pp specialist |
| HuBERT Large | 94.13% | 98.48% | +4.35 pp specialist |

### Five classes: `[Real, TTS, SE, TSE, Content Editing]`

| Encoder | AuK accuracy | Specialist accuracy | Gap |
|---|---:|---:|---:|
| Wav2Vec2 Large | 75.55% | 83.00% | +7.46 pp specialist |
| HuBERT Large | 92.97% | 97.53% | +4.56 pp specialist |

Authoritative reports:

```text
/data/probe_results/auk_fiveclass_content_edit_wav2vec2/report.json
/data/probe_results/auk_fiveclass_content_edit_hubert/report.json
/data/probe_results/specialist_fiveclass_cosyedit_wav2vec2/report.json
/data/probe_results/specialist_fiveclass_cosyedit_hubert/report.json
```

## Current interpretation

Supported observation:

> Under the same frozen encoder, speaker-disjoint split, and linear-probe protocol, the task/real boundaries are consistently weaker for the AuK unified-model outputs than for the current specialist-model pipeline.

Not supported:

- AuK tasks are impossible to distinguish.
- Existing deepfake detectors necessarily fail on AuK.
- The observed gap is caused only by shared AuK parameters.
- Specialist performance represents a pure task effect; task identity and specialist model identity are confounded.

Stable error patterns:

- AuK SE and TSE are the most consistently confused pair.
- TTS is usually highly separable, especially under HuBERT.
- AuK CE and Real have substantial two-way confusion under Wav2Vec2, but CE remains highly separable under HuBERT.

## Current blockers and risks

1. The work has not yet used a dedicated anti-spoof/deepfake detector as the main representation source.
2. Unified-versus-specialist comparisons confound task and model identity.
3. The final operational meaning of "deepfake tasks become blurred" is not yet fixed.
4. Task taxonomy remains open, especially VC, emotion/style conversion, and benign restoration versus authenticity-changing editing.
5. `Real` is an ASVspoof bona fide anchor, not a newly recorded parallel real utterance for every generated condition.
6. The Git snapshot is not yet committed or pushed.

## Recommended next discussion

Before launching more generation, decide which deepfake-oriented question is primary:

1. **Task attribution:** can a forensic representation recover TTS/SE/TSE/CE identity?
2. **Authenticity semantics:** does a detector treat benign processing as spoofing, and does this differ between AuK and specialists?
3. **Shared fingerprint:** are AuK task outputs closer to each other than outputs from task-specific models after controlling nuisance factors?

Then select dedicated anti-spoof representations/detectors and define metrics before running them.

Before that detector stage, determine the expanded task protocols one by one. The recommended first block is restoration/source selection: dereverberation, telephone/bandwidth restoration, full enhancement, and order-cued separation.

## Immediate operating instructions

- No AuK GPU job is active after completion of EXP-011 on 2026-10-07.
- Do not scale the G2 pilot until human listening review assigns each task
  `FULL`, `SUBSET`, `CONTROL`, or `DEFER`.
- Do not continue the partial CosyEdit Whisper embedding. Whisper is paused.
- Run `python scripts/audit_project_state.py` before assuming disk, GPU, process, or artifact status.
- Read `docs/codex/DECISIONS_AND_OPEN_QUESTIONS.md` before proposing a changed protocol.
- Push research work only to `origin`; the Tencent repository is retained as fetch-only `upstream`.
