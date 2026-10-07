#!/usr/bin/env bash
set -euo pipefail

OUT="/data/AuK_content_edit_full_20261001"
ROOT="/root/AuK"
PID_FILE="$OUT/monitor_pid.txt"
MONITOR_LOG="$OUT/monitor.log"

mkdir -p "$OUT"
echo "$$" > "$PID_FILE"
exec >> "$MONITOR_LOG" 2>&1
echo "[$(date '+%F %T %Z')] monitor started pid=$$"

while pgrep -f 'scripts/generate_full_content_edit_audio.py --batch-size 4 --out /data/AuK_content_edit_full_20261001' >/dev/null; do
  count=$(find "$OUT" -name 'auk_content_edit.wav' | wc -l)
  echo "[$(date '+%F %T %Z')] generation still running; audio_files=$count"
  sleep 60
done

echo "[$(date '+%F %T %Z')] generation process ended"

EXPECTED=$(awk -F '\t' 'NR>1 && $9=="accepted" && $16!="" {n++} END{print n+0}' "$ROOT/data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv")
ACTUAL=$(find "$OUT" -name 'auk_content_edit.wav' | wc -l)
echo "[$(date '+%F %T %Z')] expected=$EXPECTED actual=$ACTUAL"

if [[ "$EXPECTED" -ne "$ACTUAL" ]]; then
  echo "[$(date '+%F %T %Z')] ERROR: output count mismatch; do not package"
  exit 2
fi

cp "$ROOT/data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv" "$OUT/"
cp "$ROOT/data/content_edit_full_gpt_20261001/content_edit_text_manifest.tsv" "$OUT/"
cp "$ROOT/data/content_edit_full_gpt_20261001/report.json" "$OUT/text_selection_report.json"

OUT="$OUT" EXPECTED="$EXPECTED" /root/anaconda3/envs/auk/bin/python - <<'PY'
import json, os
from pathlib import Path
import torchaudio
import torch

out = Path(os.environ["OUT"])
expected = int(os.environ["EXPECTED"])
files = sorted(out.rglob("auk_content_edit.wav"))
bad = []
durations = []
clipped = 0
for path in files:
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != 24000:
            bad.append({"file": str(path), "error": f"sample_rate={sr}"})
        if wav.numel() == 0 or not torch.isfinite(wav).all():
            bad.append({"file": str(path), "error": "empty_or_nonfinite"})
        frac = float((wav.abs() >= 0.999).float().mean()) if wav.numel() else 0.0
        if frac > 0.01:
            clipped += 1
        durations.append(float(wav.shape[-1] / sr))
    except Exception as exc:
        bad.append({"file": str(path), "error": repr(exc)})

report = {
    "expected_audio_files": expected,
    "actual_audio_files": len(files),
    "bad_file_count": len(bad),
    "files_with_clipping_fraction_over_1pct": clipped,
    "duration_min_sec": min(durations) if durations else None,
    "duration_max_sec": max(durations) if durations else None,
    "duration_mean_sec": sum(durations) / len(durations) if durations else None,
    "bad_files": bad[:100],
    "status": "pass" if len(files) == expected and not bad else "warning",
}
(out / "audio_validation_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
if len(files) != expected or bad:
    raise SystemExit(3)
PY

cat > "$OUT/README.md" <<'EOF'
# AuK Content Editing Full Audio Dataset

This directory contains accepted single-word content-editing outputs selected by GPT and independently reviewed.

Each `pair_XXXXX/` contains:

- `auk_content_edit.wav`: AuK edited speech output
- `instruction.txt`: exact instruction used for inference

The source anchor audio remains in the matched-set source directory referenced by `content_edit_pair_manifest.tsv`; it is not duplicated in this package.
EOF

PACKAGE="/data/AuK_content_edit_full_20261001.zip"
rm -f "$PACKAGE"
cd /data
zip -qr "$PACKAGE" "AuK_content_edit_full_20261001"
echo "[$(date '+%F %T %Z')] package created: $PACKAGE"
