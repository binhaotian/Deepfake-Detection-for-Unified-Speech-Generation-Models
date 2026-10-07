# Task Group Protocol Plan

Status: active protocol-design document. Last updated: 2026-10-07.

No new full-scale generation is authorized by this document. Each task advances through definition, pilot, review, and scale decision.

## 1. Approved working groups

```text
G0  Real
G1  Synthesis
G2  Restoration and Source Selection
G3  Authenticity-changing Manipulation
G4  Paralinguistic and Delivery Manipulation
G5  Low-level Acoustic Control
G6  Singing/Music Domain (separate study)
```

Current discussion order:

```text
1. G2 Restoration and Source Selection
2. G5 Low-level Acoustic Control
3. G4 Paralinguistic and Delivery Manipulation
4. G3 Authenticity-changing Manipulation
5. G1 Synthesis expansion
6. G6 Singing/Music Domain
```

G0 Real is the anchor/control rather than a generated task group.

## 2. Required protocol fields for every task

Before a task can enter a pilot, record:

```yaml
task_id:
family_label:
task_label:
operation_label:
scientific_question:
authenticity_semantics:
source_anchor_pool:
eligibility_rule:
input_construction:
instruction_template:
target_parameter_policy:
duration_policy:
boundary_policy:
random_seed_policy:
output_sample_rate:
required_intermediate_files:
manifest_fields:
automatic_quality_checks:
human_review_protocol:
specialist_baseline:
pilot_size:
scale_decision_rule:
```

Required scale decision:

```text
FULL     all eligible anchors
SUBSET   balanced eligible subset
CONTROL  diagnostic condition only
DEFER    protocol or quality inadequate
SEPARATE different corpus/domain study
```

## 3. Group G2 — Restoration and Source Selection

### 3.1 Group definition

The model should preserve the selected target speaker and linguistic content while removing degradation or unwanted sources.

Group-level authenticity interpretation:

```text
human-origin target speech
+ neural processing
but no intended target-speaker impersonation or lexical-content fabrication
```

This is the primary `processed-real / benign-processing` family. Individual outputs may contain artifacts, but the requested operation does not intentionally falsify target identity or words.

### 3.2 Existing completed tasks

#### RES-01 — Denoising

```text
anchor + environmental noise at 0 dB SNR
-> AuK removes background noise
```

Status: full 5,369 complete.

#### RES-05 — Content-cued target speaker extraction

```text
anchor + different-speaker interferer at 0 dB SIR
-> keep the speaker who says the anchor transcript
```

Status: full 5,369 complete.

### 3.3 Proposed additions

#### RES-02 — Dereverberation

Input concept:

```text
anchor convolved with a controlled room impulse response
-> AuK removes only room reverberation
```

Official instruction basis:

```text
Remove only the room reverberation, preserve everything else,
and output audio of the same length.
```

Items to decide:

1. measured RIR corpus, simulated RIRs, or a balanced mixture;
2. target RT60 range and room-size distribution;
3. whether to preserve direct-path delay or align the reverberant input to anchor start;
4. peak/RMS scaling and clipping protection;
5. whether all 5,369 anchors are eligible;
6. specialist dereverberation baseline;
7. success criteria for a 50-pair pilot.

Expected files:

```text
rir.wav or rir metadata
dereverb_input.wav
auk_dereverb.wav
```

#### RES-03 — Full enhancement

Input concept:

```text
anchor + environmental noise + room reverberation
-> AuK removes both noise and reverberation
```

Official instruction basis:

```text
Preserve all speakers, remove noise and reverberation,
and output clean speech of the same length.
```

Items to decide:

1. apply noise before or after RIR convolution;
2. SNR and RT60 policy;
3. whether to reuse the same noise/RIR assignments as RES-01/RES-02;
4. how to compare selective versus combined restoration;
5. specialist full-enhancement baseline.

Expected files:

```text
full_enhance_noise.wav
full_enhance_rir metadata
full_enhance_input.wav
auk_full_enhance.wav
```

#### RES-04 — Channel/bandwidth restoration

First recommended operation:

```text
telephone/band-limited speech restoration
```

Input concept:

```text
anchor
-> deterministic telephone/bandwidth degradation
-> AuK restores natural clear speech
```

Official instruction basis:

```text
Repair the telephone effect and restore natural, clear speech.
```

Potential later operations:

```text
muffling
clipping
dropout
megaphone coloration
underwater-like filtering
DC offset
```

Do not treat every degradation as a top-level task initially. Use `task_label=channel_restoration` and separate `operation_label` values.

Items to decide:

1. exact telephone/bandwidth filter and sample-rate simulation;
2. whether the degradation is deterministic and identical across systems;
3. one canonical operation versus multiple balanced operations;
4. whether the model output should be compared with clean anchor using objective restoration metrics;
5. specialist bandwidth-extension/restoration baseline.

Expected files:

```text
channel_input.wav
channel_degradation.json
auk_channel_restore.wav
```

#### RES-06 — Order-cued speech separation

Input concept:

```text
two-speaker conversational/overlap mixture
-> keep first- or second-starting speaker
```

Official instruction basis:

```text
Keep only the first/second speaker to start talking
and remove all other speakers.
```

This must not reuse a fully simultaneous `t=0` mixture without modification: if both speakers start at exactly the same time, speaking-order cues are undefined. A separate mixture protocol with controlled onset offsets is required.

Items to decide:

1. onset offset distribution;
2. overlap ratio and SIR;
3. target-order balance;
4. whether the same target/interferer pairs as existing TSE are reused;
5. how to prevent trivial duration/onset shortcuts;
6. specialist speech-separation baseline;
7. whether content-cued TSE and order-cued separation are separate tasks or operations under one source-selection task.

Expected files:

```text
order_interferer.wav
order_mix.wav
order_mix_metadata.json
auk_order_separation.wav
```

### 3.4 Recommended within-group analysis

Family-level:

```text
Real vs Restoration/Selection vs other families
```

Task-level:

```text
Denoise
Dereverberate
Full enhancement
Channel restoration
Content-cued TSE
Order-cued separation
```

Within-task operations:

```text
channel restoration: telephone vs muffling vs clipping vs dropout
source selection: content cue vs order cue
```

Matched comparisons:

```text
same anchor:
Real
Denoise
Dereverb
Full enhancement
Channel restoration
TSE
Order separation
```

### 3.5 Recommended design sequence for G2

```text
G2-1  Finalize RES-02 Dereverberation
G2-2  Finalize RES-04 Telephone/bandwidth restoration
G2-3  Define RES-03 Full enhancement using G2-1 inputs
G2-4  Redesign mixtures for RES-06 Order-cued separation
G2-5  Produce one common 50-anchor pilot
G2-6  Review quality and decide FULL/SUBSET/CONTROL/DEFER
```

The next unresolved item is `G2-1 RES-02 Dereverberation`.

### 3.6 Proposed rapid first-batch recipe for RES-02/03/04

Status: accepted and generated on 2026-10-07. Automatic validation passed;
human listening review and scale decisions remain pending. The purpose is to
create one controlled, medium-severity condition per task before considering
severity sweeps.

Common pilot scope:

```text
same 50 anchors across RES-02, RES-03, and RES-04
one output per task per anchor
50 anchors -> 150 AuK outputs
```

"One severity" does not mean using one identical corruption file for every
anchor. RIR identities and existing noise identities should rotate, while the
severity policy remains fixed.

#### RES-02 proposed recipe: medium-room dereverberation

Use the Apache-2.0 OpenSLR SLR28 `RIRS_NOISES` collection and initially select
RIRs from `simulated_rirs/mediumroom`. Estimate RT60 from each candidate RIR and
prefer a single moderate band around 0.45--0.65 s; widen the band only if the
downloaded subset does not contain enough diverse RIRs. This is a reproducible
RIR augmentation source and avoids the need to design a new room simulator
before the first pilot.

```text
RIR at 16 kHz
-> high-quality resampling to 24 kHz
-> remove pre-direct-path delay / align first arrival
-> convolve with 24-kHz anchor
-> truncate to the original anchor sample count
-> active/whole-signal RMS matching according to one fixed documented rule
-> shared peak protection only when required
```

Use multiple medium-room RIR files in balanced deterministic rotation. The
first pilot should not mix small-, medium-, and large-room subsets.

Files:

```text
anchor.wav
rir reference or copied RIR
dereverb_input.wav
auk_dereverb.wav
dereverb_metadata.json
```

Instruction:

```text
Remove only the room reverberation, preserve everything else,
and output audio of the same length.
```

#### RES-03 proposed recipe: matched noise plus reverb

Use the exact RIR assignment from RES-02 and the exact noise identity/crop from
the completed RES-01 condition. Follow the common microphone-noise simulation
order:

```text
reverberant_speech = convolve(anchor, rir)
full_enhance_input = reverberant_speech + scaled_noise
```

Add noise after reverberating the speech rather than convolving the already
mixed noisy waveform. Preserve the existing SNR = 0 dB policy so that RES-03
remains directly matched to the completed denoising condition. Recompute and
record the achieved SNR against the reverberant speech because convolution can
change speech RMS.

Files:

```text
anchor.wav
same RIR assignment as RES-02
same noise identity/crop as RES-01
full_enhance_noise.wav
full_enhance_input.wav
auk_full_enhance.wav
full_enhance_metadata.json
```

Instruction:

```text
Preserve all speakers, remove noise and reverberation,
and output clean speech of the same length.
```

#### RES-04 proposed recipe: one narrowband condition

Use the mature bandwidth-limitation construction used by the URGENT challenge:
downsample and then resample back to the model's original rate. For the first
batch, fix the effective rate and resampler rather than randomizing either.

```text
24-kHz anchor
-> high-quality downsample to 8 kHz
-> high-quality resample back to 24 kHz
-> exact original sample count
```

This produces an effective approximately 4-kHz bandwidth and is labelled
`operation_label=narrowband_8khz`. Do not add G.711, clipping, packet loss, or
megaphone coloration in the first batch. Those are distinct later operations,
not hidden additions to the same condition.

Files:

```text
anchor.wav
channel_input.wav
auk_channel_restore.wav
channel_degradation.json
```

Suggested first instruction:

```text
Repair the telephone effect and restore natural, clear speech.
```

The manifest must state that the implemented corruption is bandwidth
limitation only, despite the natural-language instruction using "telephone
effect". A later pilot may compare a true G.711 telephone channel as a separate
operation.

#### First-batch review gate

Before full generation, review the same 50 anchors across all three tasks and
check:

```text
input degradation is clearly audible but speech remains intelligible
AuK output preserves transcript and speaker
output length/boundaries are not task-specific shortcuts
no systematic clipping or silence
dereverb and full-enhancement outputs actually reduce the requested corruption
channel-restoration output restores bandwidth without hallucinating words
```

After review, choose `FULL`, `SUBSET`, `CONTROL`, or `DEFER` separately for each
task. Do not infer one task's scale decision from the other two.

Completed pilot artifact and report:

```text
/data/AuK_g2_restoration_pilot_50_20261007/
/root/AuK/docs/codex/G2_RESTORATION_PILOT_20261007.md
```

## 4. Remaining groups

Only their scope is recorded here; detailed protocols will be added after G2.

### G5 — Low-level Acoustic Control

```text
Pitch
Speed
Volume
```

Main decisions: parameter balancing, duration/RMS shortcuts, and whether these remain diagnostic controls.

### G4 — Paralinguistic and Delivery Manipulation

```text
Emotion
Whisper conversion
De-accent
Nonverbal addition/removal
```

Main decisions: target selection, anchor eligibility, success validation, and authenticity semantics.

### G3 — Authenticity-changing Manipulation

```text
Content substitution/insertion/deletion
Timbre editing
```

Main decisions: text/target design, duration changes, identity labels, and specialist controls.

### G1 — Synthesis expansion

```text
Zero-shot TTS already complete
Instruct TTS optional/non-matched
```

Main decision: how to include voice-designed synthesis without pretending it is same-speaker matched.

### G6 — Singing/Music Domain

```text
Lyric editing
Music/vocal separation
```

Requires a separate corpus and should not block the speech benchmark.
