# AuK Research Snapshot

Snapshot date: 2026-10-07.

This worktree combines the upstream AuK repository with a local research project on task boundaries in unified speech generation models. It is not a clean upstream checkout.

## GitHub scope

The user approved publication in a public GitHub repository on 2026-10-07. Only the curated snapshot described here should be committed. It should contain:

- the upstream source history, based on commit `871bf3d4635c5ca0ecb6b85f3c4e29c4682a88c7`;
- the local batch-inference and per-sample-seed changes in `src/auk/infer/infer_auk.py` and `src/auk/model/cfm_edit.py`;
- research scripts under `scripts/`;
- project state, decisions, experiment ledger, artifact index, and environment notes under `docs/codex/`;
- research-method documentation under `docs/`;
- small summary reports and plots under `reports/` and `plots/`;
- `AGENTS.md`, which defines the repository operating rules;
- `data/README.md` and `data/MODEL_DATA_INVENTORY.md`, which describe local and external artifacts without uploading them.

## Excluded from GitHub

The `.gitignore` intentionally excludes:

- datasets, generated audio, embeddings, intermediate manifests, and experiment caches under `data/`;
- encoder checkpoints and model weights under `encoders/`, `ckpts/`, and `checkpoints/`;
- external artifacts under `/data`, which are outside this repository and are not backed up by Git;
- local PDF copies, archives, logs, PID files, scratch images, and ad-hoc test WAV files;
- credentials and environment overrides such as `.env`.

These exclusions avoid GitHub size limits, publication of licensed speech data, model redistribution, and disclosure of local or personal audio. The authoritative artifact locations remain documented in `docs/codex/ARTIFACT_INDEX.md`.

## Remote safety

The current `origin` points to the upstream Tencent repository:

```text
https://github.com/Tencent-Hunyuan/AuK.git
```

Do not push the research snapshot to that remote. The upstream remote is retained for fetching, while its push URL is disabled. The research repository is configured separately as `origin`:

```bash
origin   -> git@github-binhaotian:binhaotian/Deepfake-Detection-for-Unified-Speech-Generation-Models.git
upstream -> https://github.com/Tencent-Hunyuan/AuK.git (fetch only)
```

The active branch is `research-snapshot-2026-10-07`. Review `git status`, stage only the intended files, commit the snapshot, and then push the new branch. No commit or push had been performed when this document was last updated.

## Not a complete artifact backup

GitHub will preserve code, documentation, protocols, and compact results only. A separate storage backup is still required for `/root/AuK/data`, `/root/AuK/encoders`, `/root/AuK/ckpts`, and the authoritative artifacts under `/data`.
