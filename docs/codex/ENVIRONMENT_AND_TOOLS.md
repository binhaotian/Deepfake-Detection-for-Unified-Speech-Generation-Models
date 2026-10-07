# Environment and Tools

Last manually audited: 2026-10-07. Treat GPU, disk, running processes, and package availability as host/session-specific. Run `python scripts/audit_project_state.py` before new compute work.

## 1. Shell and repository

```text
working repository: /root/AuK
upstream remote: https://github.com/Tencent-Hunyuan/AuK.git
upstream/local HEAD at audit: 871bf3d4635c5ca0ecb6b85f3c4e29c4682a88c7
external artifact root: /data
```

The worktree is intentionally dirty. Never use destructive Git cleanup commands. Inspect with:

```bash
git status --short
git diff --stat
git diff -- src/auk/infer/infer_auk.py src/auk/model/cfm_edit.py
```

## 2. Base shell tools

Observed on 2026-10-07:

```text
Python:      /root/anaconda3/bin/python, 3.11.7
Conda:       /root/anaconda3/bin/conda
Git:         /usr/bin/git, 2.43.0
ripgrep:     rg 15.2.0
FFmpeg:      /root/anaconda3/bin/ffmpeg, 4.3
FFprobe:     /root/anaconda3/bin/ffprobe
pdftotext:   /usr/bin/pdftotext, 24.02.0
pdfinfo:     /usr/bin/pdfinfo, 24.02.0
pdftoppm:    /usr/bin/pdftoppm
jq:          /root/anaconda3/bin/jq
zip/unzip:   /usr/bin/zip, /usr/bin/unzip
nvidia-smi:  /usr/bin/nvidia-smi
```

Not observed in the base shell:

```text
uv
sox / soxi
mutool
```

The base Python has NumPy/SciPy/scikit-learn/pandas but was not an inference-complete AuK environment at audit time: `transformers`, `soundfile`, and `librosa` were not importable there. Do not assume bare `python` can run model inference merely because analysis scripts work.

## 3. Model-specific environments

### CosyVoice3

```text
python: /data/venvs/cosyvoice3/bin/python
Python 3.11.7
torch 2.3.1+cu121
torchaudio 2.3.1+cu121
transformers 4.51.3
numpy 1.26.4
soundfile 0.12.1
```

Full environment record:

```text
/data/specialist_experiment_documentation_20260930/model_installation/cosyvoice3_requirements-installed.txt
```

### MossFormerGAN-SE

```text
python: /data/venvs/mossformer_gan_se/bin/python
Python 3.11.7
torch 2.2.2
torchaudio 2.2.2
numpy 1.26.4
soundfile 0.12.1
```

The preserved failed/incompatible environment is:

```text
/data/venvs/mossformer_gan_se-torch214-cu130-backup-20260928/
```

Do not select it for inference.

### DAE-TSE

```text
python: /data/venvs/dae_tse/bin/python
Python 3.11.7
torch 2.9.1+cu126
torchaudio 2.9.1+cu126
numpy 2.4.6
soundfile 0.14.0
```

Use its environment wrapper when documented:

```bash
source /data/specialist_models/dae_tse/env.sh
```

### CosyEdit

```text
python: /data/CosyEdit/venv/bin/python
Python 3.11.7
torch 2.3.1+cu121
torchaudio 2.3.1+cu121
transformers 4.51.3
numpy 1.26.4
soundfile 0.12.1
```

Official repository commit:

```text
a2aea98dade9ae6bc3d22c80dd5448b10a04a57a
```

## 4. GPU checks

Before GPU work:

```bash
nvidia-smi

python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda runtime', torch.version.cuda)
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('memory', torch.cuda.get_device_properties(0).total_memory)
PY
```

Run the Python check using the exact environment intended for the job. GPU availability changed during the 2026-10-07 audit: an earlier check could not communicate with the driver, while the generated inventory later detected an RTX 4090. This is why the current `AUTO_INVENTORY.md`, not an older prose snapshot, is authoritative for runtime availability.

## 5. Reading local PDFs

Metadata:

```bash
pdfinfo "/root/AuK/AUK.pdf"
pdfinfo "/root/AuK/What Counts as Real Speech Restoration and Voice Quality Conversion Pose New Challenges to Deepfake Detection.pdf"
```

Extract text while retaining approximate layout:

```bash
pdftotext -layout "/root/AuK/AUK.pdf" /tmp/auk_paper.txt
rg -n -i "content editing|speech enhancement|target speaker|TTS|metric" /tmp/auk_paper.txt
```

Render a page for visual inspection (`-f` and `-l` are one-indexed PDF page numbers):

```bash
pdftoppm -png -f 5 -l 5 -singlefile "/root/AuK/AUK.pdf" /tmp/auk_page_5
```

Do not rely only on extracted text for tables, equations, or multi-column figure captions. Render the relevant page when layout matters.

## 6. Inspecting audio

Container/stream metadata:

```bash
ffprobe -v error \
  -show_entries stream=codec_name,sample_rate,channels,duration \
  -show_entries format=duration,size \
  -of json audio.wav
```

Full decode check without writing output:

```bash
ffmpeg -v error -i audio.wav -f null -
```

Convert only when creating an explicit analysis derivative; never overwrite source audio:

```bash
ffmpeg -i input.wav -ac 1 -ar 16000 analysis_copy.wav
```

For numerical waveform checks, use an environment containing `soundfile`:

```bash
/data/venvs/cosyvoice3/bin/python - <<'PY'
import soundfile as sf
import numpy as np
x, sr = sf.read('audio.wav', dtype='float32', always_2d=False)
x = np.asarray(x)
print({'sr': sr, 'shape': x.shape, 'finite': bool(np.isfinite(x).all()),
       'peak': float(np.max(np.abs(x))), 'rms': float(np.sqrt(np.mean(x*x)+1e-12))})
PY
```

## 7. Inspecting reports, manifests, and embeddings

JSON report:

```bash
jq '.metrics.test' /data/probe_results/auk_fiveclass_content_edit_hubert/report.json
```

TSV header/count:

```bash
head -2 manifest.tsv
awk 'END {print NR-1}' manifest.tsv
```

NumPy array without loading all data into RAM:

```bash
python - <<'PY'
import numpy as np
x = np.load('/data/embeddings/specialist_matched_hubert/features.npy', mmap_mode='r')
print(x.shape, x.dtype)
PY
```

Fast repository search:

```bash
rg --files /root/AuK/scripts
rg -n "speaker-disjoint|max_iter|confusion_matrix" /root/AuK/scripts /root/AuK/docs /data/probe_results
```

## 8. Archives and integrity

```bash
unzip -t archive.zip
sha256sum file
tar -tf archive.tar.gz | head
```

Do not extract large archives into `/root/AuK` without checking available space and whether the unpacked content already exists.

## 9. Codex/Web/MCP availability

- Shell/filesystem tools are the primary interface for local artifacts.
- Web search should be used for current external model/paper/documentation facts when available; prefer primary sources.
- No MCP resources or MCP resource templates were configured when audited on 2026-10-07.
- Tool availability can differ between Codex sessions. If a referenced tool is absent, record the fallback actually used.

## 10. State audit

Run:

```bash
cd /root/AuK
python scripts/audit_project_state.py
```

Outputs:

```text
/root/AuK/docs/codex/AUTO_INVENTORY.md
/root/AuK/docs/codex/AUTO_INVENTORY.json
```

The audit is read-only with respect to datasets and model artifacts. It only rewrites its two generated inventory files.
