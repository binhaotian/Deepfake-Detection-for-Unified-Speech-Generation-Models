# Artifact Index

Last curated: 2026-10-07. Absolute paths are used for artifacts outside the repository. Machine counts and availability should be refreshed with `python scripts/audit_project_state.py`.

## Authority rules

Priority when sources disagree:

1. Machine-readable per-run `report.json`, `config.json`, `manifest.tsv`, or generation report.
2. The exact output files referenced by that manifest.
3. `docs/codex/EXPERIMENT_LEDGER.md` and dated research archives.
4. Chat summaries or manually copied external notes.

## Repository snapshot records

| Artifact | Path | Status/role |
|---|---|---|
| GitHub snapshot scope | `/root/AuK/RESEARCH_SNAPSHOT.md` | authoritative include/exclude and remote-safety plan |
| Repository operating rules | `/root/AuK/AGENTS.md` | required workflow and preservation rules |
| Git ignore policy | `/root/AuK/.gitignore` | excludes datasets, audio, embeddings, checkpoints, archives, and local scratch artifacts |
| Initial public Git snapshot | commit `774c7ae20847083e2f14c3ac3dcd0aa3244b4f12` on branch `research-snapshot-2026-10-07` | pushed to `origin` on 2026-10-07 |

## Source data and mappings

| Artifact | Path | Status/role |
|---|---|---|
| ASVspoof/VCTK mapping archive | `/root/AuK/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz` | source mapping, preserve |
| VCTK 0.92 | `/root/AuK/data/VCTK-Corpus-0.92/` | transcript/speaker metadata and local corpus copy |
| VCTK archive copy | `/root/AuK/data/vctk_archive/` | large source archive/cache |
| AuK paper | `/root/AuK/AUK.pdf` | 31-page local PDF |
| What Counts as Real paper | `/root/AuK/What Counts as Real Speech Restoration and Voice Quality Conversion Pose New Challenges to Deepfake Detection.pdf` | 7-page local PDF |

## AuK matched data

| Artifact | Path | Verified scale |
|---|---|---:|
| Prepared full inputs | `/root/AuK/data/matched_full_5369_snr0_batch4/` | 5,369 pairs |
| Generated full set | `/root/AuK/data/matched_full_5369_snr0_batch4_generated/` | 5,369 pairs, 9 WAV each |
| Full manifest | `/root/AuK/data/matched_full_5369_snr0_batch4_generated/manifest.tsv` | 5,369 data rows |
| Generation config | `/root/AuK/data/matched_full_5369_snr0_batch4_generated/generation_config.json` | authoritative AuK settings |
| Formal documentation | `/root/AuK/data/AuK_matched_full_5369_documentation/` | metadata/reports |
| Packaged full set | `/root/AuK/data/AuK_matched_full_5369_snr0.zip` | archive; do not casually duplicate |
| RMS-normalized analysis copy | `/root/AuK/data/matched_full_5369_rmsnorm/` | analysis derivative, not source data |

## TTS variants

| Artifact | Path | Status |
|---|---|---|
| V2 active-duration set | `/root/AuK/data/tts_full_v2_v3_5369/v2_active_duration/` | complete historical variant |
| V3 text-duration set | `/root/AuK/data/tts_full_v2_v3_5369/v3_text_duration/` | complete historical variant |
| V2/V3 log | `/root/AuK/docs/experiment_log_v2_v3_and_asvspoof.md` | authoritative design/result summary |

## Encoder checkpoints

| Encoder | Path | Current use |
|---|---|---|
| Wav2Vec2 Large | `/root/AuK/encoders/wav2vec2-large-960h-lv60-self/` | active |
| HuBERT Large | `/root/AuK/encoders/hubert-large-ll60k/` | active |
| Whisper Large-v3 | `/root/AuK/encoders/whisper-large-v3/` | historical/paused |

## Specialist models and environments

| Task | Model root | Environment | Version record |
|---|---|---|---|
| TTS | `/data/specialist_models/cosyvoice3/` | `/data/venvs/cosyvoice3/` | `/data/specialist_experiment_documentation_20260930/model_installation/cosyvoice3_INSTALL.md` |
| SE | `/data/specialist_models/mossformer_gan_se/` | `/data/venvs/mossformer_gan_se/` | `/data/specialist_experiment_documentation_20260930/model_installation/mossformer_gan_se_INSTALL_MANIFEST.md` |
| TSE | `/data/specialist_models/dae_tse/` | `/data/venvs/dae_tse/` | `/data/specialist_experiment_documentation_20260930/model_installation/dae_tse_INSTALLATION.md` |
| CE | `/data/CosyEdit/repo/` | `/data/CosyEdit/venv/` | official commit `a2aea98dade9ae6bc3d22c80dd5448b10a04a57a` |

## Specialist outputs

| Task | Path | Verified output |
|---|---|---:|
| CosyVoice3 TTS | `/data/specialist_outputs/cosyvoice3/` | 5,369 `specialist_tts.wav` |
| MossFormerGAN SE | `/data/specialist_outputs/mossformer_gan_se/` | 5,369 `specialist_se.wav` |
| DAE-TSE | `/data/specialist_outputs/dae_tse/` | 5,369 `specialist_tse.wav` |

Main installation/generation archive:

```text
/data/specialist_experiment_documentation_20260930/
/data/specialist_experiment_documentation_20260930.zip
```

## Content Editing text and audio

| Artifact | Path | Verified scale/status |
|---|---|---|
| Text selection | `/root/AuK/data/content_edit_full_gpt_20261001/` | 3,734 accepted pair rows |
| Pair manifest | `/root/AuK/data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv` | source for both CE systems |
| AuK CE audio | `/data/AuK_content_edit_full_20261001/` | 3,734 `auk_content_edit.wav` |
| CosyEdit CE audio | `/data/CosyEdit/content_edit_full_3734_20261006/` | 3,734 flat `pair_*.wav` |
| CosyEdit manifest | `/data/CosyEdit/content_edit_full_3734_20261006/manifest.tsv` | authoritative output mapping/validation |
| CE research archive | `/data/content_edit_research_archive_20261006/` | process archive; its “generation ongoing” sentence is historical |

## G2 restoration pilot

| Artifact | Path | Verified scale/status |
|---|---|---|
| OpenSLR SLR28 download | `/data/resources/openslr28/rirs_noises.zip` | 1,311,166,223 bytes; ZIP integrity passed |
| Extracted RIR resource | `/data/resources/openslr28/extracted/RIRS_NOISES/` | 61,260 WAV total; 20,000 medium-room RIRs |
| AuK G2 pilot | `/data/AuK_g2_restoration_pilot_50_20261007/` | 50 pairs; 150 AuK outputs; automatic validation clean |
| AuK G2 full pilot archive | `/data/AuK_g2_restoration_pilot_50_20261007.zip` | 115 MB; ZIP and internal SHA256 checks passed |
| AuK G2 listening subset | `/data/AuK_g2_restoration_listening_10_20261007.zip` | 10 diverse pairs; 70 ordered WAV; integrity passed |
| AuK G2 package hashes | `/data/AuK_g2_restoration_packages_20261007.sha256` | SHA256 for both ZIP archives |
| G2 pilot report | `/root/AuK/docs/codex/G2_RESTORATION_PILOT_20261007.md` | authoritative human-readable protocol/result summary |
| G2 revision pilot | `/data/AuK_g2_revision_pilot_10_20261007/` | 10 pairs; 20 new AuK outputs; validation clean |
| G2 revision full archive | `/data/AuK_g2_revision_pilot_10_20261007.zip` | 27 MB; ZIP integrity passed |
| G2 revision listening archive | `/data/AuK_g2_revision_listening_10_20261007.zip` | 10 A/B pairs; 80 ordered WAV; ZIP integrity passed |
| G2 revision package hashes | `/data/AuK_g2_revision_packages_20261007.sha256` | SHA256 for revision ZIP archives |
| G2 revision report | `/root/AuK/docs/codex/G2_REVISION_PILOT_20261007.md` | Full 0/10 dB and channel-prompt A/B protocol/results |
| G2 canonical pilot workspace | `/data/AuK_g2_canonical_pilot_10_20261007/` | 10 pairs; 30 new AuK outputs; machine-readable config/report |
| G2 canonical audio-only directory | `/data/AuK_g2_canonical_listening_10_20261007/` | 10 sample folders; 5 WAV per folder; no metadata inside |
| G2 canonical audio-only archive | `/data/AuK_g2_canonical_listening_10_20261007.zip` | 50 WAV; 13 MB; ZIP/decode validation passed |
| G2 canonical archive hash | `/data/AuK_g2_canonical_listening_10_20261007.sha256` | SHA256 `4fc07393747f60a6fd3fdfcc620e9d5e274c64f4d1420361ef749fbaff11ecd1` |
| G2 canonical pilot report | `/root/AuK/docs/codex/G2_CANONICAL_AUDIO_PILOT_20261007.md` | authoritative protocol, prompts, layout, and generation summary |

Per-pair task outputs:

```text
dereverb/auk_dereverb.wav
full_enhancement/auk_full_enhance.wav
channel_restoration/auk_channel_restore.wav
```

## Embeddings

| Artifact | Path | Shape/order |
|---|---|---|
| Specialist Wav2Vec2 | `/data/embeddings/specialist_matched_wav2vec2/features.npy` | `(5369, 7, 1024)` |
| Specialist HuBERT | `/data/embeddings/specialist_matched_hubert/features.npy` | `(5369, 7, 1024)` |
| AuK CE Wav2Vec2 | `/data/embeddings/content_edit_full_20261001/wav2vec2/features.npy` | `(3734, 2, 1024)` |
| AuK CE HuBERT | `/data/embeddings/content_edit_full_20261001/hubert/features.npy` | `(3734, 2, 1024)` |
| CosyEdit CE Wav2Vec2 | `/data/embeddings/cosyedit_content_full_20261006/wav2vec2/features.npy` | `(3734, 2, 1024)` |
| CosyEdit CE HuBERT | `/data/embeddings/cosyedit_content_full_20261006/hubert/features.npy` | `(3734, 2, 1024)` |
| Partial CosyEdit Whisper | `/data/embeddings/cosyedit_content_full_20261006/whisper/features.npy` | non-authoritative; paused; do not use |

Seven-stream order for specialist embeddings:

```text
[Real, AuK-TTS, CosyVoice3-TTS, AuK-SE, MossFormer-SE, AuK-TSE, DAE-TSE]
```

CE paired feature order:

```text
[anchor, content_edit_output]
```

## Probe reports

### Historical AuK four-class

```text
/root/AuK/data/probe_results/wav2vec2-large-fourclass/report.json
/root/AuK/data/probe_results/hubert-large-fourclass/report.json
/root/AuK/data/probe_results/whisper-large-fourclass/report.json
/root/AuK/data/probe_results/tts_v2_wav2vec2/report.json
/root/AuK/data/probe_results/tts_v3_wav2vec2/report.json
/root/AuK/data/probe_results/tts_v2_hubert/report.json
/root/AuK/data/probe_results/tts_v3_hubert/report.json
```

### Current matched AuK/specialist four-class

```text
/data/probe_results/matched_fourclass_auk_wav2vec2_mi5000/report.json
/data/probe_results/matched_fourclass_specialist_wav2vec2_mi5000/report.json
/data/probe_results/matched_fourclass_auk_hubert_mi5000/report.json
/data/probe_results/matched_fourclass_specialist_hubert_mi5000/report.json
```

### Current five-class

```text
/data/probe_results/auk_fiveclass_content_edit_wav2vec2/report.json
/data/probe_results/auk_fiveclass_content_edit_hubert/report.json
/data/probe_results/specialist_fiveclass_cosyedit_wav2vec2/report.json
/data/probe_results/specialist_fiveclass_cosyedit_hubert/report.json
```

## Scripts that define current experiments

```text
/root/AuK/scripts/prepare_full_matched_dataset.py
/root/AuK/scripts/generate_full_matched_dataset.py
/root/AuK/scripts/generate_tts_v2_v3_pilot.py
/root/AuK/scripts/extract_specialist_matched_embeddings.py
/root/AuK/scripts/train_matched_fourclass_probe.py
/root/AuK/scripts/build_full_content_edit_manifest_gpt.py
/root/AuK/scripts/generate_full_content_edit_audio.py
/root/AuK/scripts/extract_content_edit_embeddings.py
/root/AuK/scripts/train_auk_fiveclass_content_edit_probe.py
/root/AuK/scripts/extract_cosyedit_content_embeddings.py
/root/AuK/scripts/train_specialist_fiveclass_content_edit.py
/root/AuK/scripts/prepare_g2_restoration_pilot.py
/root/AuK/scripts/generate_g2_restoration_pilot.py
/root/AuK/scripts/run_g2_revision_pilot.py
/root/AuK/scripts/run_g2_canonical_audio_pilot.py
```

## Storage status

Snapshot from 2026-10-07 before automatic refresh:

```text
/ filesystem: about 21 GiB free, about 90% used
/data: about 54 GiB free, about 47% used
```

Do not rely on this snapshot for new downloads; read `AUTO_INVENTORY.md` or rerun the audit.
