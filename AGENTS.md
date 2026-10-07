# Codex instructions for the AuK deepfake study

This repository contains both the upstream AuK codebase and a local research project on task boundaries in unified speech generation models. Do not treat it as a clean upstream checkout.

## Required reading order

Before planning or running research work, read:

1. `CODEX_PROJECT_STATE.md` — current phase, completed work, paused items, and next questions.
2. `docs/codex/RESEARCH_BRIEF.md` — scientific question, definitions, and claim boundaries.
3. `docs/codex/DECISIONS_AND_OPEN_QUESTIONS.md` — decisions that should not be silently changed.
4. `docs/codex/ARTIFACT_INDEX.md` — authoritative paths under `/root/AuK` and `/data`.
5. Relevant entries in `docs/codex/EXPERIMENT_LEDGER.md` before reusing or extending an experiment.

Use `docs/codex/ENVIRONMENT_AND_TOOLS.md` before reading local PDFs, inspecting audio, choosing a Python environment, or launching GPU work.

For copy-paste prompts when opening another Codex window, see `docs/codex/NEW_SESSION_PROMPT.md`.

## Current research scope

The active comparison is between:

- one unified model, AuK, completing TTS, speech enhancement (SE), target speaker extraction (TSE), and content editing (CE);
- separate specialist models: CosyVoice3, MossFormerGAN-SE-16K, DAE-TSE, and CosyEdit.

The current evidence comes primarily from frozen Wav2Vec2 Large and HuBERT Large representations plus linear probes. These are general-purpose speech encoders, not dedicated anti-spoof detectors. Do not describe the existing results as direct deepfake-detector performance.

Whisper is historical/paused. Do not launch new Whisper extraction or probe work unless the user explicitly resumes it.

## Data and artifact rules

- `/root/AuK` contains source code, main AuK data, encoder checkpoints, scripts, and historical reports.
- `/data` contains specialist models/environments, specialist outputs, large Content Editing outputs, embeddings, and experiment archives.
- Do not move, delete, regenerate, normalize, or overwrite large audio/model artifacts without explicit user authorization.
- Prefer the machine-readable `manifest.tsv`, `config.json`, `report.json`, and generation reports over prose summaries.
- Preserve matched-pair identity and speaker-disjoint splits. Never split different tasks from one pair or speaker across train/test.
- Do not use detector scores, embeddings, or subjective quality to filter samples unless a new protocol explicitly requires it.

## Local source modifications

The worktree is intentionally dirty. In particular, `src/auk/infer/infer_auk.py` and `src/auk/model/cfm_edit.py` contain local batch-inference/per-sample-seed modifications used for data generation. Do not discard or overwrite them as cleanup.

## Before a long or GPU task

1. Run `python scripts/audit_project_state.py`.
2. Check `nvidia-smi`; the GPU/driver state can change between machines or sessions.
3. Confirm the intended virtual environment and checkpoint.
4. Check whether valid outputs already exist and require resumable/skip-existing behavior.
5. Record the exact command, input manifest, output directory, seed, and console log.

## After material work

Update all applicable records:

- `CODEX_PROJECT_STATE.md` for current status and next action;
- `docs/codex/EXPERIMENT_LEDGER.md` for completed/failed/abandoned experiments;
- `docs/codex/ARTIFACT_INDEX.md` for new authoritative artifacts;
- `docs/codex/DECISIONS_AND_OPEN_QUESTIONS.md` for scientific or protocol decisions;
- regenerate `docs/codex/AUTO_INVENTORY.md` with `python scripts/audit_project_state.py`.

Use absolute dates (`YYYY-MM-DD`) and absolute paths for external artifacts. Distinguish verified facts, interpretations, and hypotheses.
