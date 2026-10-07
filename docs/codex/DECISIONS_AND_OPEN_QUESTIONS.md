# Decisions and Open Questions

Last curated: 2026-10-07.

## Status vocabulary

```text
accepted   = active project decision; do not silently change
historical = used in completed work but not necessarily preferred now
paused     = preserve artifacts; do not continue without user instruction
open       = unresolved scientific or protocol question
```

## Accepted decisions

### D-001 — Anchor source

Status: accepted.

Use ASVspoof2019 LA eval bona fide recordings from the 48 target speakers as Real anchors. VCTK provides transcripts and speaker metadata; VCTK waveforms do not replace ASVspoof anchors.

### D-002 — Main pair identity

Status: accepted.

The unit of analysis is `pair_id`. All task/system variants of one pair remain linked. Evaluation splits are speaker-disjoint.

### D-003 — TTS reference

Status: accepted.

Use an independent same-speaker ASV enrollment recording. Do not use the anchor as its own TTS reference.

### D-004 — SE input

Status: accepted.

Use anchor plus deterministic environmental noise at SNR = 0 dB. Preserve exact noise identity and measured SNR in the manifest.

### D-005 — TSE input

Status: accepted.

Use anchor plus a different-speaker interferer with full overlap at SIR = 0 dB. Prefer same-gender interferers and rotate among fixed candidates.

### D-006 — Main AuK generation settings

Status: historical/accepted for the completed main set.

```text
AuK Base
24 kHz
BF16
NFE = 32
CFG = 2.0
sway = -1.0
batch size = 4
```

Exact configuration: `/root/AuK/data/matched_full_5369_snr0_batch4_generated/generation_config.json`.

### D-007 — Content Editing definition

Status: accepted.

Current CE uses one-word substitution only. Candidate generation, independent GPT review, and deterministic exact-one-word validation precede audio generation.

### D-008 — Specialist models

Status: accepted for the current baseline.

```text
TTS: CosyVoice3
SE:  MossFormerGAN-SE-16K
TSE: DAE-TSE, content-cued
CE:  CosyEdit
```

Do not call every selected model an uncontested absolute SOTA. They are strong, public, reproducible, task-compatible specialist baselines.

### D-009 — Probe split

Status: accepted.

Use speaker-disjoint 34/7/7 speaker splits with random seed 20260925 unless a new experiment explicitly defines another protocol. Report pair/sample counts because CE uses a 3,734-pair subset.

### D-010 — Active encoders

Status: accepted.

Continue current comparisons with Wav2Vec2 Large and HuBERT Large. New Whisper work is paused as of 2026-10-06/07. Historical Whisper results remain valid historical artifacts.

### D-011 — No score-based filtering

Status: accepted.

Do not remove samples based on detector score, embedding position, or whether a result supports the hypothesis. Quality-control exclusions must be defined independently and logged.

### D-012 — Intermediate inputs are not Real

Status: accepted.

`se_input.wav` and `tse_input.wav` are controlled model inputs, not additional bona fide/Real examples.

### D-013 — Broaden beyond the original four tasks

Status: accepted at the research-scope level on 2026-10-07.

The study will expand beyond TTS, denoising SE, content-cued TSE, and CE substitution. AuK's full speech capability set will be organized hierarchically by the authenticity dimension changed. This decision approves taxonomy/protocol design, not automatic full-scale generation of every task. Each task still requires an explicit protocol and pilot decision.

Planning document: `/root/AuK/docs/codex/AUK_EXPANSION_ROADMAP.md`.

### D-014 — Adopt the working hierarchical task groups

Status: accepted as the protocol-planning framework on 2026-10-07.

```text
Real
Synthesis
Restoration and Source Selection
Authenticity-changing Manipulation
Paralinguistic and Delivery Manipulation
Low-level Acoustic Control
Singing/Music Domain as a separate study
```

Each sample will eventually receive family, task, and operation labels. Exact paper terminology may still be refined, but new task design should use this hierarchy unless the user/advisor revises it.

The group-by-group discussion begins with Restoration and Source Selection. Detailed checklist: `/root/AuK/docs/codex/TASK_GROUP_PROTOCOL_PLAN.md`.

### D-015 — G2 three-task pilot construction

Status: accepted on 2026-10-07.

Prepare one common 50-anchor pilot for three Restoration/Selection tasks:

```text
RES-02 dereverberation
RES-03 full enhancement (noise + reverberation)
RES-04 narrowband channel restoration
```

Accepted first-pilot settings:

```text
RIR source: OpenSLR SLR28 RIRS_NOISES simulated medium-room subset
RIR severity: estimated RT60 0.45--0.65 s, with documented fallback 0.35--0.75 s
RIR diversity: multiple balanced RIR identities, not one shared RIR
full enhancement: (anchor * RIR) + the existing pair noise at measured 0 dB SNR
channel restoration: fixed 24 kHz -> 8 kHz -> 24 kHz polyphase bandwidth limitation
pilot size: the same 50 speaker-balanced anchors for all three tasks
AuK settings: Base, BF16, NFE=32, CFG=2.0, sway=-1.0, batch size 4
```

The pilot is isolated under `/data`; the completed 5,369-pair dataset must not
be modified. Generation is resumable and must save input construction metadata,
progress logs, and output validation reports.

### D-016 — G2 revision listening decision

Status: partially accepted on 2026-10-07.

For `RES-03 Full Enhancement`, use the revised input condition:

```text
(anchor * RIR) + matched environmental noise at measured SNR = 10 dB
```

The user judged this condition more reasonable than the original 0 dB pilot,
where noise dominated perception and masked the reverberation component. The
0 dB condition remains a stress-control artifact and is not the preferred main
condition.

For `RES-04 Channel Restoration`, the explicit speech-super-resolution
instruction invokes the intended bandwidth-restoration behavior much more
clearly than the public Cookbook-style telephone-repair instruction. It is not
yet accepted as the final production prompt: listening found occasional
strange noise, consistent with the measured high-band overshoot. Preserve both
pilot variants and either soften the explicit instruction or document the
strong version as a deliberately aggressive condition before full scaling.

Prompt routing note: the official Prompt Enhancer configuration contains a
dedicated `improve_quality/bandwidth_extension` template that explicitly asks
for missing high-frequency recovery. Future bandwidth-only inputs must use
this route rather than the generic `telephone effect` route. A telephone prompt
is appropriate only when the input construction actually simulates telephone
coloration/channel effects.

`RES-02 Dereverberation` remains a required standalone condition. Full
Enhancement does not replace it, because the combined task can be dominated by
denoising and conceal weak dereverberation behavior. Before scaling, compare
the short Cookbook instruction with the richer official Prompt Enhancer
`enhance_speech/cleanup_mode=dereverb` templates on the same pilot inputs.

### D-017 — One AuK variant in the main task-boundary study

Status: accepted on 2026-10-07.

Use AuK Base for all main unified-model task conditions. Do not switch selected
restoration/separation tasks to AuK-Flash merely because Flash reports stronger
perceptual metrics on some signal-processing benchmarks. Mixing Base and Flash
would introduce a model-variant fingerprint into the task comparison and weaken
the interpretation that one unified model produced every class.

Current fixed settings remain:

```text
AuK Base, 24 kHz, BF16, NFE=32, CFG=2.0, sway=-1.0
```

### D-018 — Matched 10 dB denoising control for G2

Status: accepted on 2026-10-07.

Add a 10 dB denoising-only condition for the new G2 restoration comparison.
Use the same per-pair noise identity and scaling convention as the 10 dB Full
Enhancement condition, but do not apply an RIR:

```text
Denoising-10dB input = anchor + noise at measured SNR 10 dB
Full-Enhancement-10dB input = (anchor * matched RIR) + same noise at measured SNR 10 dB
```

This makes Denoising versus Full Enhancement a controlled comparison instead
of confounding task identity with 0-versus-10 dB noise severity. Preserve the
completed 0 dB AuK-SE dataset as a historical/stress condition; do not replace,
overwrite, or relabel it.

## Open scientific questions

### Q-001 — Final task taxonomy

Status: open.

Current proposed hierarchy:

```text
Real
Synthesis
Restoration and source selection
Authenticity-changing manipulation
Paralinguistic/delivery manipulation
Low-level acoustic control
Separate singing/music domain
```

The Level-1 working hierarchy was approved on 2026-10-07. This question remains open only for final paper naming, Level-2 boundaries, and ambiguous cases such as timbre editing, de-accent, and source-selection cue variants. The official AuK five-family taxonomy is retained as capability provenance, while the forensic hierarchy describes what changes relative to the anchor.

### Q-002 — Operational definition of "cannot distinguish"

Status: open.

Candidate definitions include:

- real/fake score collapse;
- task-attribution accuracy/F1;
- pairwise class confusion;
- neighborhood mixing or class-centroid distance;
- shared model attribution despite task changes.

The final paper may require more than one, but one must be declared primary.

### Q-003 — Dedicated deepfake detector

Status: open/high priority.

The current encoder probes are representation studies. Select dedicated anti-spoof models or representations before making a detector-level claim. Candidate families previously discussed include AASIST, RawNet2/LCNN, and SSL anti-spoof systems, but no final detector set is approved here.

### Q-004 — Task versus model fingerprint confound

Status: open/high priority.

The specialist comparison binds each task to one model. Determine whether additional within-task multi-model controls, source-attribution probes, or hierarchical labels can separate task effects from generator identity.

### Q-005 — Authenticity semantics

Status: open.

Decide whether neural enhancement/extraction are labelled spoof, processed-real, or a separate axis. The answer affects the deepfake detector experiment and paper framing.

### Q-006 — Pair-relative analysis

Status: open.

Anchor-relative residuals are natural for SE/TSE/CE but structurally less clean for TTS, which lacks a waveform-level real counterpart. Do not force all tasks into a residual design without explicitly addressing this asymmetry.

### Q-007 — Additional unified models

Status: open/later.

AuK is currently the only unified model in the generated dataset. UniAudio, Metis/Metis-Omni, SpeechX, and Voicebox are related-work candidates, not completed experimental systems.

### Q-008 — Final statistical protocol

Status: open.

Before paper claims, decide whether to add multiple split seeds, confidence intervals, paired significance tests, per-speaker aggregation, and nuisance-controlled analysis.

## Operational questions

### O-001 — Git preservation

Status: upload scope and remotes prepared on 2026-10-07; commit and push still require user approval.

The batch inference changes in `src/auk/infer/infer_auk.py` and `src/auk/model/cfm_edit.py` must be preserved in the research snapshot. The intended repository contents and exclusions are recorded in `/root/AuK/RESEARCH_SNAPSHOT.md`.

The user approved a public GitHub repository for this curated snapshot on 2026-10-07. The research repository is configured as `origin`. The Tencent repository has been renamed to `upstream`, retained for fetches, and given a disabled push URL. The active local branch is `research-snapshot-2026-10-07`.

### O-002 — Root disk pressure

`/root` was at approximately 90% use on 2026-10-07. Prefer `/data` for new large models, environments, embeddings, audio, caches, and temporary files.

### O-003 — Partial Whisper artifact

`/data/embeddings/cosyedit_content_full_20261006/whisper/features.npy` is a partial/non-authoritative artifact from an interrupted/deferred attempt. Do not use it in reports. Do not delete it without permission.
