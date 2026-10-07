#!/usr/bin/env bash
set -u

OUT="/data/AuK_content_edit_full_20261001"
ROOT="/root/AuK"
LOG="$OUT/gpu_resume_watcher.log"
mkdir -p "$OUT"
exec >> "$LOG" 2>&1
echo "[$(date '+%F %T %Z')] GPU resume watcher started pid=$$"

while true; do
  if nvidia-smi -L >/dev/null 2>&1; then
    echo "[$(date '+%F %T %Z')] GPU visible; launching resumable AuK generation"
    nohup setsid /root/anaconda3/envs/auk/bin/python "$ROOT/scripts/generate_full_content_edit_audio.py" \
      --batch-size 4 --out "$OUT" \
      > "$OUT/auk_resume_after_gpu.log" 2>&1 < /dev/null &
    sleep 3
    nohup setsid "$ROOT/scripts/monitor_and_package_content_edit.sh" \
      > "$OUT/monitor_after_gpu_launcher.log" 2>&1 < /dev/null &
    echo "[$(date '+%F %T %Z')] generator launched pid=$!"
    exit 0
  fi
  echo "[$(date '+%F %T %Z')] GPU not visible; retrying in 60s"
  sleep 60
done
