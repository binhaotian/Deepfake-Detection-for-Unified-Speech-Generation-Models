#!/usr/bin/env python3
"""Generate a targeted, read-only audit of the AuK research artifacts.

The script does not scan the complete repository or modify datasets/models. It
checks only registered high-value paths and rewrites AUTO_INVENTORY.md/json.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data")
OUT_DIR = ROOT / "docs" / "codex"
OUT_MD = OUT_DIR / "AUTO_INVENTORY.md"
OUT_JSON = OUT_DIR / "AUTO_INVENTORY.json"
NOW = datetime.now(ZoneInfo("Asia/Shanghai"))


def run(command: list[str], *, cwd: Path | None = None, timeout: int = 20) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_percent": round(100.0 * usage.used / usage.total, 2),
    }


def count_glob(root: Path, pattern: str) -> int | None:
    if not root.exists():
        return None
    return sum(1 for _ in root.glob(pattern))


def tsv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def tsv_value_counts(path: Path, column: str) -> dict[str, int] | None:
    if not path.exists():
        return None
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or column not in reader.fieldnames:
            return None
        for row in reader:
            value = (row.get(column) or "").strip() or "<empty>"
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def npy_info(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    try:
        array = np.load(path, mmap_mode="r")
        record.update(
            {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": path.stat().st_size,
            }
        )
    except Exception as exc:  # audit should report damage rather than crash
        record["error"] = repr(exc)
    return record


def report_info(name: str, path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        test = metrics.get("test", metrics.get("eval", {})) if isinstance(metrics, dict) else {}
        record.update(
            {
                "pair_count": payload.get("pair_count"),
                "sample_count": payload.get("sample_count"),
                "classes": payload.get("classes") or test.get("labels"),
                "accuracy": test.get("accuracy"),
                "balanced_accuracy": test.get("balanced_accuracy"),
                "macro_f1": test.get("macro_f1"),
                "confusion_matrix": test.get("confusion_matrix"),
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)
    return record


def git_snapshot() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    status = run(["git", "status", "--short"], cwd=ROOT)
    lines = status["stdout"].splitlines() if status["stdout"] else []
    important = [
        line
        for line in lines
        if any(
            key in line
            for key in (
                "src/auk/infer/infer_auk.py",
                "src/auk/model/cfm_edit.py",
                "AGENTS.md",
                "CODEX_PROJECT_STATE.md",
                "docs/codex",
            )
        )
    ]
    return {
        "head": head["stdout"] or None,
        "status_ok": status["ok"],
        "dirty_entry_count": len(lines),
        "important_entries": important,
    }


def gpu_snapshot() -> dict[str, Any]:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    devices = []
    if result["ok"]:
        for line in result["stdout"].splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 4:
                devices.append(
                    {
                        "name": fields[0],
                        "driver_version": fields[1],
                        "memory_total_mib": fields[2],
                        "memory_used_mib": fields[3],
                    }
                )
    return {
        "available": bool(devices),
        "devices": devices,
        "error": None if result["ok"] else (result["stderr"] or result["stdout"]),
    }


def active_research_processes() -> list[dict[str, str]]:
    result = run(["ps", "-eo", "pid=,etime=,cmd="], timeout=10)
    if not result["ok"]:
        return []
    tokens = (
        "generate_full",
        "content_edit",
        "extract_",
        "train_",
        "cosyedit",
        "cosyvoice",
        "mossformer",
        "dae_tse",
    )
    records = []
    for line in result["stdout"].splitlines():
        lower = line.lower()
        if "audit_project_state.py" in lower:
            continue
        if not any(token in lower for token in tokens):
            continue
        parts = line.strip().split(maxsplit=2)
        if len(parts) == 3:
            records.append({"pid": parts[0], "elapsed": parts[1], "command": parts[2]})
    return records


def environment_snapshot() -> list[dict[str, Any]]:
    environments = [
        ("base", Path("/root/anaconda3/bin/python")),
        ("cosyvoice3", Path("/data/venvs/cosyvoice3/bin/python")),
        ("mossformer_gan_se", Path("/data/venvs/mossformer_gan_se/bin/python")),
        ("dae_tse", Path("/data/venvs/dae_tse/bin/python")),
        ("cosyedit", Path("/data/CosyEdit/venv/bin/python")),
    ]
    records = []
    for name, executable in environments:
        record: dict[str, Any] = {"name": name, "python": str(executable), "exists": executable.exists()}
        if executable.exists():
            version = run([str(executable), "--version"], timeout=10)
            record["version"] = version["stdout"] or version["stderr"]
        records.append(record)
    return records


def build_snapshot() -> dict[str, Any]:
    matched = ROOT / "data" / "matched_full_5369_snr0_batch4_generated"
    cosyedit = DATA / "CosyEdit" / "content_edit_full_3734_20261006"
    content_manifest = ROOT / "data" / "content_edit_full_gpt_20261001" / "content_edit_pair_manifest.tsv"
    content_status_counts = tsv_value_counts(content_manifest, "status")

    expected_main_names = [
        "anchor.wav",
        "tts_reference.wav",
        "auk_tts.wav",
        "se_noise.wav",
        "se_input.wav",
        "auk_se.wav",
        "tse_interferer.wav",
        "tse_input.wav",
        "auk_tse.wav",
    ]
    main_counts = {name: count_glob(matched, f"pair_*/{name}") for name in expected_main_names}

    embeddings = [
        npy_info(DATA / "embeddings" / "specialist_matched_wav2vec2" / "features.npy"),
        npy_info(DATA / "embeddings" / "specialist_matched_hubert" / "features.npy"),
        npy_info(DATA / "embeddings" / "content_edit_full_20261001" / "wav2vec2" / "features.npy"),
        npy_info(DATA / "embeddings" / "content_edit_full_20261001" / "hubert" / "features.npy"),
        npy_info(DATA / "embeddings" / "cosyedit_content_full_20261006" / "wav2vec2" / "features.npy"),
        npy_info(DATA / "embeddings" / "cosyedit_content_full_20261006" / "hubert" / "features.npy"),
        {
            **npy_info(DATA / "embeddings" / "cosyedit_content_full_20261006" / "whisper" / "features.npy"),
            "authoritative": False,
            "status": "partial_paused",
        },
    ]

    reports = [
        report_info(
            "fourclass_auk_wav2vec2",
            DATA / "probe_results" / "matched_fourclass_auk_wav2vec2_mi5000" / "report.json",
        ),
        report_info(
            "fourclass_specialist_wav2vec2",
            DATA / "probe_results" / "matched_fourclass_specialist_wav2vec2_mi5000" / "report.json",
        ),
        report_info(
            "fourclass_auk_hubert",
            DATA / "probe_results" / "matched_fourclass_auk_hubert_mi5000" / "report.json",
        ),
        report_info(
            "fourclass_specialist_hubert",
            DATA / "probe_results" / "matched_fourclass_specialist_hubert_mi5000" / "report.json",
        ),
        report_info(
            "fiveclass_auk_wav2vec2",
            DATA / "probe_results" / "auk_fiveclass_content_edit_wav2vec2" / "report.json",
        ),
        report_info(
            "fiveclass_specialist_wav2vec2",
            DATA / "probe_results" / "specialist_fiveclass_cosyedit_wav2vec2" / "report.json",
        ),
        report_info(
            "fiveclass_auk_hubert",
            DATA / "probe_results" / "auk_fiveclass_content_edit_hubert" / "report.json",
        ),
        report_info(
            "fiveclass_specialist_hubert",
            DATA / "probe_results" / "specialist_fiveclass_cosyedit_hubert" / "report.json",
        ),
    ]

    return {
        "generated_at": NOW.isoformat(),
        "repository": str(ROOT),
        "disk": [disk_snapshot(Path("/")), disk_snapshot(DATA)],
        "git": git_snapshot(),
        "gpu": gpu_snapshot(),
        "active_research_processes": active_research_processes(),
        "environments": environment_snapshot(),
        "datasets": {
            "main_matched": {
                "root": str(matched),
                "pair_directories": count_glob(matched, "pair_*"),
                "manifest_rows": tsv_rows(matched / "manifest.tsv"),
                "wav_counts": main_counts,
                "total_registered_wavs": sum(value or 0 for value in main_counts.values()),
            },
            "specialist_outputs": {
                "cosyvoice3_tts": count_glob(DATA / "specialist_outputs" / "cosyvoice3", "**/specialist_tts.wav"),
                "mossformer_se": count_glob(DATA / "specialist_outputs" / "mossformer_gan_se", "**/specialist_se.wav"),
                "dae_tse": count_glob(DATA / "specialist_outputs" / "dae_tse", "**/specialist_tse.wav"),
            },
            "content_edit": {
                "pair_manifest_rows": tsv_rows(content_manifest),
                "status_counts": content_status_counts,
                "accepted_manifest_rows": (content_status_counts or {}).get("accepted"),
                "auk_outputs": count_glob(DATA / "AuK_content_edit_full_20261001", "pair_*/auk_content_edit.wav"),
                "cosyedit_outputs": count_glob(cosyedit, "pair_*.wav"),
                "cosyedit_manifest_rows": tsv_rows(cosyedit / "manifest.tsv"),
            },
        },
        "embeddings": embeddings,
        "reports": reports,
        "documents": {
            "agents": (ROOT / "AGENTS.md").exists(),
            "project_state": (ROOT / "CODEX_PROJECT_STATE.md").exists(),
            "research_brief": (OUT_DIR / "RESEARCH_BRIEF.md").exists(),
            "experiment_ledger": (OUT_DIR / "EXPERIMENT_LEDGER.md").exists(),
            "artifact_index": (OUT_DIR / "ARTIFACT_INDEX.md").exists(),
            "environment_tools": (OUT_DIR / "ENVIRONMENT_AND_TOOLS.md").exists(),
            "decisions": (OUT_DIR / "DECISIONS_AND_OPEN_QUESTIONS.md").exists(),
            "experiment_template": (OUT_DIR / "EXPERIMENT_ENTRY_TEMPLATE.md").exists(),
            "new_session_prompt": (OUT_DIR / "NEW_SESSION_PROMPT.md").exists(),
            "auk_task_inventory": (OUT_DIR / "AUK_TASK_INVENTORY.md").exists(),
            "auk_expansion_roadmap": (OUT_DIR / "AUK_EXPANSION_ROADMAP.md").exists(),
            "task_group_protocol_plan": (OUT_DIR / "TASK_GROUP_PROTOCOL_PLAN.md").exists(),
        },
    }


def fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        "# Auto-generated Project Inventory",
        "",
        f"Generated: `{snapshot['generated_at']}`.",
        "",
        "> Generated by `python scripts/audit_project_state.py`. Do not hand-edit; update curated context in the other Codex documents.",
        "",
        "## Runtime state",
        "",
    ]

    gpu = snapshot["gpu"]
    if gpu["available"]:
        for device in gpu["devices"]:
            lines.append(
                f"- GPU: `{device['name']}`, driver `{device['driver_version']}`, "
                f"memory `{device['memory_used_mib']}/{device['memory_total_mib']} MiB`."
            )
    else:
        lines.append(f"- GPU unavailable to `nvidia-smi`: `{gpu.get('error') or 'unknown error'}`")

    processes = snapshot["active_research_processes"]
    lines.append(f"- Matching active research processes: `{len(processes)}`.")
    for process in processes:
        lines.append(f"  - PID `{process['pid']}`, elapsed `{process['elapsed']}`: `{process['command']}`")

    lines.extend(["", "## Disk", "", "| Path | Used | Free | Total | Used % |", "|---|---:|---:|---:|---:|"])
    for disk in snapshot["disk"]:
        lines.append(
            f"| `{disk['path']}` | {human_bytes(disk['used_bytes'])} | {human_bytes(disk['free_bytes'])} | "
            f"{human_bytes(disk['total_bytes'])} | {disk['used_percent']:.2f}% |"
        )

    git = snapshot["git"]
    lines.extend(
        [
            "",
            "## Git",
            "",
            f"- HEAD: `{git['head']}`",
            f"- Dirty/untracked entries reported by Git: `{git['dirty_entry_count']}`",
        ]
    )
    for entry in git["important_entries"]:
        lines.append(f"  - `{entry}`")

    main = snapshot["datasets"]["main_matched"]
    specialist = snapshot["datasets"]["specialist_outputs"]
    ce = snapshot["datasets"]["content_edit"]
    lines.extend(
        [
            "",
            "## Registered datasets and outputs",
            "",
            f"- Main matched pair directories: `{main['pair_directories']}`; manifest rows: `{main['manifest_rows']}`.",
            f"- Main registered WAV total: `{main['total_registered_wavs']}`.",
        ]
    )
    for name, count in main["wav_counts"].items():
        lines.append(f"  - `{name}`: `{count}`")
    lines.extend(
        [
            f"- CosyVoice3 TTS outputs: `{specialist['cosyvoice3_tts']}`",
            f"- MossFormerGAN SE outputs: `{specialist['mossformer_se']}`",
            f"- DAE-TSE outputs: `{specialist['dae_tse']}`",
            f"- Accepted CE manifest rows: `{ce['accepted_manifest_rows']}`",
            f"- CE pair manifest total rows: `{ce['pair_manifest_rows']}`; status counts: `{ce['status_counts']}`",
            f"- AuK CE outputs: `{ce['auk_outputs']}`",
            f"- CosyEdit CE outputs: `{ce['cosyedit_outputs']}`; manifest rows: `{ce['cosyedit_manifest_rows']}`",
        ]
    )

    lines.extend(["", "## Embeddings", "", "| Path | Exists | Shape | dtype | Status |", "|---|---:|---|---|---|"])
    for embedding in snapshot["embeddings"]:
        shape = str(embedding.get("shape", "n/a"))
        status = embedding.get("status", "authoritative" if embedding.get("exists") else "missing")
        if "error" in embedding:
            status = f"error: {embedding['error']}"
        lines.append(
            f"| `{embedding['path']}` | {embedding['exists']} | `{shape}` | `{embedding.get('dtype', 'n/a')}` | {status} |"
        )

    lines.extend(
        [
            "",
            "## Probe reports",
            "",
            "| Name | Exists | Pairs | Accuracy | Macro-F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for report in snapshot["reports"]:
        lines.append(
            f"| `{report['name']}` | {report['exists']} | {report.get('pair_count') or 'n/a'} | "
            f"{fmt_metric(report.get('accuracy'))} | {fmt_metric(report.get('macro_f1'))} |"
        )

    lines.extend(["", "## Python environments", ""])
    for environment in snapshot["environments"]:
        lines.append(
            f"- `{environment['name']}`: `{environment['python']}`; exists={environment['exists']}; "
            f"version=`{environment.get('version', 'n/a')}`"
        )

    lines.extend(["", "## State-document completeness", ""])
    for name, exists in snapshot["documents"].items():
        lines.append(f"- `{name}`: `{exists}`")

    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "This inventory verifies artifact presence/counts and recorded metrics. It does not establish scientific validity, audio quality, absence of leakage, or dedicated deepfake-detector performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    snapshot = build_snapshot()
    atomic_write(OUT_JSON, json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    atomic_write(OUT_MD, render_markdown(snapshot))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
