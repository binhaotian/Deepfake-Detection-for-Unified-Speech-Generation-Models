# AuK Task Expansion Roadmap

Status: approved working taxonomy and planning roadmap. Last curated: 2026-10-07. This document does not authorize generation; each task still requires an explicit protocol and pilot decision.

## 1. New project direction

The project will no longer be limited to TTS, denoising SE, content-cued TSE, and one-word Content Editing. The intended scope is a broad, structured study of AuK's speech-generation and speech-editing capabilities, organized by what changes relative to a real anchor and what that change means for authenticity/deepfake analysis.

AuK's official five training families remain the capability source:

```text
Speech Generation
Content Editing
Acoustic Editing
Paralinguistic Editing
Enhancement and Separation
```

For this research, the final labels should use a forensic taxonomy that cuts across the official training taxonomy.

## 2. Proposed hierarchical taxonomy

### Level 0 — Real

```text
REAL
└── original ASVspoof/VCTK-mapped bona fide anchor
```

### Level 1 — Synthesis

The target waveform is newly generated rather than locally transformed from the anchor.

```text
SYNTHESIS
├── Zero-shot TTS
└── Instruct TTS
```

### Level 1 — Restoration and source selection

The target speaker/content should remain authentic; the model removes degradation or unwanted sources.

```text
RESTORATION_SELECTION
├── Denoising
├── Dereverberation
├── Full enhancement: noise + reverberation
├── Channel/quality restoration
│   ├── telephone/bandwidth restoration
│   ├── muffling restoration
│   ├── clipping restoration
│   └── dropout restoration
├── Target speaker extraction: content cue
└── Speech separation: order/loudness/timestamp cue
```

TSE and enhancement belong together at Level 1 because the intended target content and identity are preserved and the operation removes unwanted components. They remain separate Level-2 tasks.

### Level 1 — Authenticity-changing manipulation

This broad family covers deliberate changes to what was said or whose/what kind of voice is represented.

```text
AUTHENTICITY_MANIPULATION
├── Linguistic-content manipulation
│   ├── substitution
│   ├── insertion
│   └── deletion
└── Voice/identity-characteristic manipulation
    └── timbre editing
```

Content Editing and timbre editing can share this Level-1 family because both alter an authenticity-bearing attribute. They should not be collapsed into one flat task label: content and speaker/voice identity are different axes and must remain distinct at Level 2.

AuK does not expose a separately named reference-speaker Voice Conversion task. Timbre editing is the closest official capability, but description-driven timbre editing must not be silently relabelled as conventional target-speaker VC.

### Level 1 — Paralinguistic and delivery manipulation

The words and intended speaker are preserved, while delivery style or vocal behavior changes.

```text
PARALINGUISTIC
├── Emotion editing
├── De-accent
├── Whisper conversion
└── Nonverbal editing
    ├── nonverbal addition
    └── nonverbal removal
```

### Level 1 — Low-level acoustic control

The operation directly modifies a measurable acoustic property.

```text
ACOUSTIC_CONTROL
├── Pitch editing
├── Speed editing
└── Volume editing
```

These tasks are important controls: if they are easy to classify, that may reflect explicit pitch/duration/RMS differences rather than a model fingerprint.

### Separate domain — Singing and music

```text
MUSIC_DOMAIN
├── Lyric editing
└── Music/vocal separation
```

These are official AuK capabilities but require singing/music data and should not be mixed into the first speech-only benchmark.

## 3. Label hierarchy for future experiments

Every generated sample should have at least three labels:

```text
family_label     broad forensic family
task_label       concrete task
operation_label  target/parameter or edit operation
```

Examples:

| Audio | family_label | task_label | operation_label |
|---|---|---|---|
| AuK denoising | restoration_selection | speech_enhancement | denoise |
| AuK dereverb | restoration_selection | speech_enhancement | dereverberate |
| AuK TSE | restoration_selection | target_speaker_extraction | content_cue |
| AuK CE substitution | authenticity_manipulation | content_editing | substitution |
| AuK timbre | authenticity_manipulation | timbre_editing | target_description_id |
| AuK happy emotion | paralinguistic | emotion_editing | happy |
| AuK +2 semitones | acoustic_control | pitch_editing | plus_2_semitones |
| AuK zero-shot TTS | synthesis | zero_shot_tts | same_speaker_reference |

This enables:

1. family-level classification;
2. task-level classification;
3. within-family operation classification;
4. Real versus each broad family;
5. unified-model versus specialist comparisons at compatible levels.

## 4. Proposed task inventory

### 4.1 Existing completed conditions

| ID | Family | Task | Status |
|---|---|---|---|
| REAL-01 | Real | bona fide anchor | complete, 5,369 |
| SYN-01 | Synthesis | zero-shot TTS | complete, 5,369 |
| RES-01 | Restoration/selection | denoising | complete, 5,369 |
| RES-05 | Restoration/selection | content-cued TSE | complete, 5,369 |
| MAN-01 | Authenticity manipulation | one-word substitution | complete, 3,734 accepted pairs |

### 4.2 Core speech expansion candidates

These should be discussed and piloted one by one.

| Proposed ID | Family | Task/operation | Anchor compatibility | Preliminary priority |
|---|---|---|---|---|
| RES-02 | Restoration/selection | dereverberation | high after artificial RIR mixing | high |
| RES-03 | Restoration/selection | full noise + reverb enhancement | high | medium |
| RES-04 | Restoration/selection | telephone/bandwidth restoration | high after deterministic degradation | high |
| RES-06 | Restoration/selection | order-cued speech separation | high using existing/new mixtures | medium |
| MAN-02 | Authenticity manipulation | content insertion | high after text design | high |
| MAN-03 | Authenticity manipulation | content deletion | high after text design | high |
| MAN-04 | Authenticity manipulation | timbre editing | high waveform compatibility; target-label design unresolved | high, after definition |
| PARA-01 | Paralinguistic | emotion editing | high; target emotion design needed | high |
| PARA-02 | Paralinguistic | whisper conversion | high for normal-to-whisper | high |
| PARA-03 | Paralinguistic | de-accent | conditional on anchor accent suitability | conditional |
| PARA-04A | Paralinguistic | nonverbal addition | high after location/event design | medium |
| PARA-04R | Paralinguistic | nonverbal removal | only anchors containing target events | conditional |
| ACO-01 | Acoustic control | pitch editing | high | high as control |
| ACO-02 | Acoustic control | speed editing | high; duration changes | high as control |
| ACO-03 | Acoustic control | volume editing | high | high as control |

### 4.3 Secondary/non-matched candidates

| Proposed ID | Task | Reason not in first common matched set |
|---|---|---|
| SYN-02 | Instruct TTS | no same-speaker reference; voice is designed rather than matched |
| MUSIC-01 | Lyric editing | requires isolated singing vocals |
| MUSIC-02 | music/vocal separation | music/singing domain rather than read speech |

## 5. Transformation metadata matrix

These fields should be explicit in every future manifest.

| Task | Content changes | Identity target changes | Delivery/style changes | Scene/degradation changes | Duration may change |
|---|---:|---:|---:|---:|---:|
| Zero-shot TTS | generated from text | imitated from reference | may drift | entire waveform regenerated | yes |
| Denoise | no | no | should not | noise removed | no |
| Dereverb | no | no | should not | reverb removed | no |
| Channel restoration | no | no | should not | channel degradation repaired | no |
| TSE/separation | target content no | target identity no | should not | other source(s) removed | no |
| CE substitution | yes | no | should not | local region regenerated | possibly |
| CE insertion/deletion | yes | no | should not | local region regenerated | yes |
| Timbre editing | no | yes/voice characteristic altered | often | waveform broadly regenerated | usually no target change |
| Emotion editing | no | intended no | yes | waveform broadly edited | usually no |
| De-accent | pronunciation/prosody only | intended no | yes | waveform broadly edited | usually no |
| Whisper conversion | no | intended no | yes | phonation changes | usually no |
| Nonverbal add/remove | nonlexical event changes | no | yes | local event changes | often |
| Pitch editing | no | intended no | low-level | pitch changes | no |
| Speed editing | no | intended no | low-level | time scale changes | yes |
| Volume editing | no | no | low-level | gain changes | no |

## 6. Avoiding combinatorial explosion

Do not immediately generate all supported parameter values for all 5,369 anchors. Use three stages.

### Stage A — Protocol design

For each candidate, decide:

```text
task definition
family/task/operation labels
eligible anchors
exact instruction template
target parameter policy
duration policy
quality-control policy
specialist baseline availability
whether the output should be considered spoof, processed-real, or a separate axis
```

### Stage B — Common 50-pair pilot

Use the same 50 anchors across as many compatible speech tasks as possible. Save:

```text
anchor
task input/intermediate degradation where applicable
AuK raw output
final analysis output if alignment is applied
instruction
all parameter metadata
```

Perform human listening and automatic checks without filtering by encoder/detector score.

### Stage C — Scale decision

After the pilot, assign each task one of:

```text
FULL:        generate for all eligible 5,369 anchors
SUBSET:      generate a balanced eligible subset
CONTROL:     keep only as diagnostic control
DEFER:       protocol/quality not adequate
SEPARATE:    different corpus/domain study
```

## 7. Proposed decision order

We should define tasks in this order, one at a time:

### Block 1 — Restoration and source selection

```text
1. Dereverberation
2. Telephone/bandwidth restoration
3. Full enhancement
4. Order-cued separation
```

Reason: the data-generation logic can reuse the existing anchors and deterministic degradation/mixing pipeline. This block also gives a clearer “benign processing” family around existing SE/TSE.

### Block 2 — Acoustic controls

```text
5. Pitch
6. Speed
7. Volume
```

Reason: easiest to define and generate; useful as low-level controls. Parameter balance and shortcut interpretation must be fixed before generation.

### Block 3 — Paralinguistic manipulation

```text
8. Emotion
9. Whisper conversion
10. De-accent
11. Nonverbal addition/removal
```

Reason: scientifically central but requires more quality/eligibility decisions than acoustic controls.

### Block 4 — Authenticity-changing manipulation

```text
12. CE insertion
13. CE deletion
14. Timbre editing
```

Reason: content and identity authenticity are central to the deepfake framing, but text/target-label design needs deliberate review.

### Block 5 — Non-matched and separate-domain tasks

```text
15. Instruct TTS
16. Lyric editing
17. Music separation
```

Reason: these should not constrain or delay the common speech matched benchmark.

The user approved this group-by-group planning order on 2026-10-07. Detailed protocol work begins with `RESTORATION_SELECTION`; see `docs/codex/TASK_GROUP_PROTOCOL_PLAN.md`.

## 8. Planned experiments after data generation

The expanded dataset should support multiple label granularities rather than one very large flat classifier only.

```text
Probe 1: Real vs all AuK-processed/generated
Probe 2: family-level classification
Probe 3: task-level classification
Probe 4: within-family operation classification
Probe 5: AuK versus specialist at family/task level
Probe 6: dedicated anti-spoof detector scores/representations
Probe 7: nuisance controls for duration, RMS, pitch, and transformation magnitude
```

The final deepfake claim should be based on dedicated detector experiments plus the hierarchical representation analysis, not only generic encoder task probes.
