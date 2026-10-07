# AuK Task Inventory

Last curated: 2026-10-07.

Primary sources:

```text
/root/AuK/AUK.pdf
/root/AuK/README.md
/root/AuK/docs/COOKBOOK.md
```

## 1. Unified interface

The AuK technical report groups pre-training into five task families:

```text
1. Speech Generation
2. Acoustic Editing
3. Paralinguistic Editing
4. Content Editing
5. Enhancement and Separation
```

All families use the same high-level interface:

```text
natural-language instruction
+ optional input/reference audio
-> target waveform
```

The public Cookbook exposes 16 named task entries. Some entries contain multiple operations or degradation types, so the practical capability count is larger than 16.

## 2. Complete public task list

| Family | Public task | Input/condition | Intended change | Current project status |
|---|---|---|---|---|
| Speech Generation | Zero-shot TTS | voice reference + target text | generate new speech in reference voice | completed as AuK-TTS |
| Speech Generation | Instruct TTS | voice/style description + target text; no reference audio | design and generate a described voice | not generated |
| Content Editing | Speech Content Editing | source speech + edit instruction | insert, delete, or replace spoken content | substitution completed; insertion/deletion not done |
| Content Editing | Lyric Editing | isolated singing/vocal recording + lyric edit | edit lyrics while preserving melody and singer | not generated; different domain |
| Acoustic Editing | Pitch Editing | source speech + semitone instruction | change pitch, preserve duration/content/speaker | not generated |
| Acoustic Editing | Speed Editing | source speech + speed multiplier | change speaking rate; duration changes | not generated |
| Acoustic Editing | Volume Editing | source speech + dB instruction | change loudness, preserve other attributes | not generated |
| Paralinguistic Editing | Emotion Editing | source speech + target emotion | change emotion, preserve content and voice | not generated |
| Paralinguistic Editing | Timbre Editing | source speech + timbre description | change timbre, preserve linguistic content | not generated; closest public AuK task to description-driven VC |
| Paralinguistic Editing | De-accent | accented source speech | reduce accent, preserve voice/content | not generated |
| Paralinguistic Editing | Nonverbal Editing | source speech + add/remove instruction | add/remove breaths, laughs, coughs, humming, etc. | not generated |
| Paralinguistic Editing | Whisper Conversion | normal or whispered source speech | normal-to-whisper or whisper-to-normal | not generated |
| Enhancement & Separation | Speech Enhancement | degraded speech + selective restoration instruction | denoise, dereverberate, restore channel/quality | denoising completed; other subtypes not done |
| Enhancement & Separation | Speech Separation | multi-speaker mixture + order cue | retain speaker(s) by speaking order | not generated as a separate class |
| Enhancement & Separation | Music Separation | music/speech/singing mixture + source request | isolate singing or retain human voices | not generated; different domain |
| Enhancement & Separation | Target Speaker Extraction | multi-speaker mixture + spoken-content cue | retain the speaker who says specified content | completed as AuK-TSE |

## 3. Operation-level details

### 3.1 Speech Generation

#### Zero-shot TTS

```text
input: reference speech
instruction: Say the following with the same voice: "..."
output: newly synthesized target text
```

The reference transcript is not required. The current project already uses this mode with an independent same-speaker ASV enrollment recording.

#### Instruct TTS

```text
input audio: none
condition: free-form voice description + target text
```

The paper's voice descriptions can cover gender, age, speaking rate, clarity, fluency, vocal state, intonation, loudness, timbre, pitch, accent, emotion, and personality.

This task does not fit the current same-speaker matched design because it has no speaker reference and is intended to design a voice rather than reproduce the anchor speaker.

### 3.2 Content Editing

#### Speech Content Editing

Official operations:

```text
substitution: replace A with B
insertion:    add B before/after anchor A
deletion:     remove A or remove content near anchor A
```

The current project covers only one-word substitution. Insertion and deletion are distinct supported operations and were evaluated separately in the paper.

#### Lyric Editing

Requires isolated/dry vocals rather than ordinary VCTK speech. It edits local lyrics while attempting to preserve singer identity, melody, rhythm, expression, and surrounding audio.

### 3.3 Acoustic Editing

Training targets described in the paper:

```text
speed:  0.5x, 0.75x, 1.25x, 1.5x, 2.0x
pitch:  ±1, ±2, ±3 semitones
volume: ±5, ±10, ±15 dB
```

These operations preserve linguistic content and nominal speaker identity. Speed changes duration; pitch and volume are intended to preserve duration.

Because their target changes are low-level and quantitatively specified, they may create simple acoustic shortcuts. They are useful negative controls but are weaker candidates for the main deepfake story.

### 3.4 Paralinguistic Editing

#### Emotion Editing

Eight target categories in the report:

```text
angry, happy, sad, fearful, surprised, disgusted, calm, excited
```

Content and voice are intended to remain stable while delivery changes.

#### Timbre Editing

Changes timbre based on a natural-language description while preserving content. The report does not expose a separately named conventional speaker-to-speaker "Voice Conversion" entry; timbre editing is its closest public capability, but a descriptive timbre target is not identical to reference-speaker VC.

#### De-accent

Removes or reduces regional accent while preserving speaker voice and content. Training supervision described in the paper is based primarily on 13 Chinese dialect/regional-accent categories. The paper reports qualitative English generalization, so English use should not be treated as equally established without validation.

#### Nonverbal Editing

Supports both removal and insertion. The training corpus normalizes annotations into 39 nonverbal event types spanning physiological sounds, affective expressions, and discourse vocalizations.

Examples include breaths, laughs, coughs, and humming. Addition may change output duration; removal may require explicit target duration.

#### Whisper Conversion

Supports both directions:

```text
normal speech -> whisper
whisper -> normal speech
```

Speaker and linguistic content are intended to remain unchanged.

### 3.5 Enhancement and Separation

#### Speech Enhancement subtypes

Public Cookbook operations:

```text
denoising
dereverberation
full enhancement: remove noise and reverberation
quality/channel restoration
```

Quality restoration examples include:

```text
telephone effect
muffling
clipping
signal dropout
```

The technical report additionally describes bandwidth limitation, megaphone coloration, underwater-like filtering, and DC offset. VCTK-SR evaluation covers bandwidth/channel recovery and is described as speech super-resolution.

Enhancement is instruction-selective: a denoising request may intentionally retain reverberation, while a dereverberation request may retain environmental sound.

#### Multi-speaker separation

The training design is broader than the single public order-based example. Target speakers can be identified using:

```text
spoken content
speaking order
relative loudness
exclusive timestamp
```

Instructions may retain or remove one speaker or a subset of speakers, while preserving non-target noise/reverberation/channel effects when requested.

Current AuK-TSE uses spoken content as the cue. A separate order-cued speech-separation class would partly change the cueing method rather than introduce a wholly unrelated signal-processing task.

#### Music enhancement/separation

The paper describes mixtures containing speech, singing, and background music and requests such as:

```text
retain singing voice
retain all human voices
retain a singer by order/timestamp
remove non-vocal accompaniment
```

This requires a singing/music dataset and should not be mixed directly into the current VCTK/ASVspoof speech-only matched set.

## 4. Relevance to the current deepfake study

### Highest-value next candidates

#### A. Emotion Editing

Why useful:

- same input anchor;
- same linguistic content;
- intended same speaker;
- changes a semantically meaningful delivery attribute;
- lies between benign processing and identity/content manipulation.

Primary risk: emotion success varies by source utterance and target emotion, so task quality needs independent validation.

#### B. Whisper Conversion

Why useful:

- same anchor/content/speaker target;
- much stronger acoustic transformation than SE/TSE;
- does not require inventing new text;
- could test whether task boundaries follow acoustic change magnitude rather than authenticity semantics.

Primary risk: normal-to-whisper output may differ greatly in pitch/voicing and become trivially separable.

#### C. De-accent

Why useful:

- preserves text and intended identity;
- changes pronunciation/prosody rather than content;
- conceptually close to the “processed but not necessarily fake” question.

Primary risk: current English VCTK anchors may not contain strong removable accents, and AuK's de-accent training is mainly Chinese.

#### D. Timbre Editing

Why useful:

- closest AuK task to voice conversion;
- authenticity/identity implications are stronger than SE/TSE;
- source content is preserved.

Primary risk: target identity is not naturally matched without a reference speaker, and the semantic label “fake” becomes much less ambiguous only if the intended identity change is clearly specified.

#### E. Content insertion/deletion

Why useful:

- expands the already constructed CE family;
- paper evaluates insertion, deletion, and substitution separately;
- lets the project test whether manipulation operation matters within one task family/model.

Primary risk: insertion/deletion systematically change duration and alignment, adding an operation shortcut. They should initially be analyzed as CE subtypes, not automatically as new top-level classes.

### Useful controls, but weaker main tasks

```text
Pitch editing
Speed editing
Volume editing
Nonverbal addition/removal
Dereverberation
Channel restoration / speech super-resolution
Order-cued speech separation
```

These are useful for disentangling transformation magnitude, duration, or low-level acoustic effects. They may be better treated as controls or within-family subtypes than as equal top-level deepfake classes.

### Better kept as separate studies

```text
Instruct TTS: no same-speaker reference; breaks the current matching logic
Lyric Editing: singing domain
Music Separation: music/singing domain
```

## 5. Recommended expansion order

If expanding the current matched speech set, the cleanest order is:

```text
1. Emotion Editing
2. Whisper Conversion
3. Content Editing insertion/deletion as CE subtypes
4. Timbre Editing, after defining identity/authenticity labels
5. De-accent, only after checking anchor suitability
6. Acoustic controls and restoration subtypes as diagnostic controls
```

This order is a research recommendation, not an accepted protocol decision. Any new full generation requires an explicit decision in `DECISIONS_AND_OPEN_QUESTIONS.md` and a new `EXP-xxx` entry.
