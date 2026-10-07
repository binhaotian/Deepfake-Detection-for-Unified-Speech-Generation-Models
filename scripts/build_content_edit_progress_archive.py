#!/usr/bin/env python3
"""Build a reproducible progress archive for the Content Editing study."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


ARCHIVE_ROOT = Path("/data/content_edit_research_archive_20261006")
ZIP_BASE = Path("/data/content_edit_research_archive_20261006")
PAIR_MANIFEST = Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_pair_manifest.tsv")
TEXT_MANIFEST = Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_text_manifest.tsv")
SOURCE_ROOT = Path("/root/AuK/data/matched_full_5369_snr0_batch4_generated")
AU_OUT = Path("/data/AuK_content_edit_full_20261001")
CE_SAMPLE = Path("/data/AuK_content_edit_review_sample15_20261006")
COSY = Path("/data/CosyEdit")
COSY_OUT = COSY / "content_edit_full_3734_20261006"
EMB = Path("/data/embeddings/content_edit_full_20261001")


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file():
        shutil.copy2(src, dst)


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    if ARCHIVE_ROOT.exists():
        shutil.rmtree(ARCHIVE_ROOT)
    ARCHIVE_ROOT.mkdir(parents=True)

    accepted = [r for r in read_tsv(PAIR_MANIFEST) if r.get("status") == "accepted"]
    texts = [r for r in read_tsv(TEXT_MANIFEST) if r.get("status") == "accepted"]
    generated = list(COSY_OUT.glob("*.wav"))
    worker_pids = subprocess.run(
        "ps -eo cmd= | rg 'run_full_content_edit_workers' | rg -v rg | wc -l",
        shell=True, text=True, capture_output=True, check=False,
    ).stdout.strip()
    worker_count = int(worker_pids or 0)
    status = "running" if worker_count else "not_running_or_finished"

    # Lightweight source and review artifacts.
    for src in [
        Path("/root/AuK/data/content_edit_full_gpt_20261001/README.md"),
        PAIR_MANIFEST,
        TEXT_MANIFEST,
        Path("/root/AuK/data/content_edit_full_gpt_20261001/content_edit_text_manifest.json"),
        Path("/root/AuK/data/content_edit_full_gpt_20261001/source_text_groups.tsv"),
        Path("/root/AuK/data/content_edit_full_gpt_20261001/report.json"),
        Path("/root/AuK/data/content_edit_full_gpt_20261001/preview.html"),
    ]:
        copy(src, ARCHIVE_ROOT / "01_text_selection" / src.name)
    copy(Path("/root/AuK/data/content_edit_full_gpt_20261001/generation_batches"), ARCHIVE_ROOT / "01_text_selection/generation_batches")
    copy(Path("/root/AuK/data/content_edit_full_gpt_20261001/review_batches"), ARCHIVE_ROOT / "01_text_selection/review_batches")

    # Scripts are part of the reproducibility record.
    for src in [
        Path("/root/AuK/scripts/build_full_content_edit_manifest_gpt.py"),
        Path("/root/AuK/scripts/generate_full_content_edit_audio.py"),
        Path("/root/AuK/scripts/package_content_edit_review_sample15.py"),
        Path("/root/AuK/scripts/extract_content_edit_embeddings.py"),
        Path("/root/AuK/scripts/train_auk_fiveclass_content_edit_probe.py"),
        Path("/data/CosyEdit/run_batch.py"),
        Path("/data/CosyEdit/run_full_content_edit.py"),
        Path("/data/CosyEdit/run_full_content_edit_workers.py"),
    ]:
        copy(src, ARCHIVE_ROOT / "07_scripts" / src.name)

    # Human-auditable pilot package and CosyEdit pilot outputs.
    copy(CE_SAMPLE, ARCHIVE_ROOT / "02_auk_15pair_listening_sample")
    for src in [
        COSY / "outputs/batch_20261006/manifest.tsv",
        COSY / "outputs/batch_20261006/manifest.json",
        COSY / "logs/smoke_official_example.log",
        COSY / "logs/smoke_fp32.log",
        COSY / "logs/batch_20261006.log",
        COSY / "logs/batch_20261006.console.log",
    ]:
        copy(src, ARCHIVE_ROOT / "05_cosyedit_pilot" / src.name)
    copy(COSY / "outputs/batch_20261006", ARCHIVE_ROOT / "05_cosyedit_pilot/audio")

    # Full AuK generated-audio metadata and validation, without duplicating all WAVs.
    for src in [
        AU_OUT / "content_edit_pair_manifest.tsv",
        AU_OUT / "content_edit_text_manifest.tsv",
        AU_OUT / "text_selection_report.json",
        AU_OUT / "audio_validation_report.json",
        AU_OUT / "README.md",
        AU_OUT / "generation_report.json",
    ]:
        if src.exists():
            copy(src, ARCHIVE_ROOT / "03_auk_full_audio_metadata" / src.name)
    copy(Path("/data/AuK_content_edit_full_20261001/generation_progress.jsonl"), ARCHIVE_ROOT / "03_auk_full_audio_metadata/generation_progress.jsonl")

    # Encoder outputs and probe reports are directly reusable downstream.
    for enc in ("wav2vec2", "hubert"):
        copy(EMB / enc, ARCHIVE_ROOT / "04_embeddings" / enc)
    for src in [
        Path("/data/probe_results/auk_fiveclass_content_edit_wav2vec2/report.json"),
        Path("/data/probe_results/auk_fiveclass_content_edit_wav2vec2/split.tsv"),
        Path("/data/probe_results/auk_fiveclass_content_edit_hubert/report.json"),
        Path("/data/probe_results/auk_fiveclass_content_edit_hubert/split.tsv"),
        Path("/data/probe_results/matched_fourclass_auk_wav2vec2/report.json"),
        Path("/data/probe_results/matched_fourclass_specialist_wav2vec2/report.json"),
        Path("/data/probe_results/matched_fourclass_auk_hubert/report.json"),
        Path("/data/probe_results/matched_fourclass_specialist_hubert/report.json"),
    ]:
        if src.exists():
            copy(src, ARCHIVE_ROOT / "06_probe_results" / src.name)

    # CosyEdit full-generation state: manifests/logs only, not the full WAV tree.
    for src in [
        COSY_OUT / "manifest.tsv",
        COSY_OUT / "generation.log",
        COSY_OUT / "generation_report.json",
    ]:
        if src.exists():
            copy(src, ARCHIVE_ROOT / "05_cosyedit_full_generation" / src.name)
    for src in sorted(COSY_OUT.glob("worker_*.log")) + sorted(COSY_OUT.glob("worker_*.tsv")) + sorted(COSY_OUT.glob("worker_*.json")):
        copy(src, ARCHIVE_ROOT / "05_cosyedit_full_generation" / src.name)
    for src in sorted((COSY / "logs").glob("worker_*.console.log")) + [COSY / "logs/full_generation_console_4090.log"]:
        if src.exists():
            copy(src, ARCHIVE_ROOT / "05_cosyedit_full_generation/logs" / src.name)

    # Machine-readable inventory.
    inventory = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source_pairs": 5369,
        "accepted_content_edit_pairs": len(accepted),
        "accepted_unique_text_groups": len({r["text_group_id"] for r in texts}),
        "auk_full_audio_expected": 3734,
        "auk_full_audio_path": str(AU_OUT),
        "cosyedit_full_expected": 3734,
        "cosyedit_full_generated_at_archive_time": len(generated),
        "cosyedit_full_output_path": str(COSY_OUT),
        "cosyedit_worker_processes_at_archive_time": worker_count,
        "cosyedit_status_at_archive_time": status,
        "embedding_encoders": ["wav2vec2", "hubert"],
        "embedding_shape_per_encoder": [3734, 2, 1024],
        "five_class_order": ["real", "tts", "se", "tse", "content_edit"],
    }
    (ARCHIVE_ROOT / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = """# Content Editing 研究阶段归档

归档日期：2026-10-06。该压缩包记录从 Content Editing 数据构想开始，到 AuK 五分类和 CosyEdit specialist baseline 部署/全量生成阶段的过程、配置、结果和可复现实验文件。

## 研究目的

在相同真人 anchor、speaker、source corpus 和尽可能一致的内容条件下，比较 AuK 统一生成模型完成不同任务时的 frozen speech-encoder representation boundary，并与专用模型结果进行对照。Content Editing 作为新增任务，采用单词级 substitution：只改变一个词，保留其余句子。

## 1. 文本候选构造与审核

- 母体：5369 个已有 matched pairs，来自 VCTK/ASVspoof anchor。
- 唯一原句：3857 条。
- 方法：GPT 逐句提出单词替换候选；独立 GPT 复核、接受/修订/拒绝；程序执行精确一词替换验证。
- accepted：2708 个 text groups，3734 个 pair rows。
- rejected：1020 个 groups，1459 个 pair rows。
- needs_review：129 个 groups，176 个 pair rows。
- accepted 路线：B_semantic_contrast 与 C_natural_context。
- 只有 `accepted` 且指令非空的 pair 进入音频生成。

完整 manifest、JSON 报告、GPT generation/review 批次日志和浏览器预览位于 `01_text_selection/`。

## 2. AuK Content Editing 音频

AuK 输入为 anchor 音频和单词替换指令，输出保存为 `auk_content_edit.wav`。全量 3734 条 AuK 音频已经完成并在原始路径 `/data/AuK_content_edit_full_20261001/` 保存；归档只收录其 manifest、质量报告和生成日志，不重复复制约 GB 级音频。

另有 15 条覆盖不同 route、词性、说话人和估计音节变化的人工试听包，位于 `02_auk_15pair_listening_sample/`。

## 3. Frozen encoder embedding

使用 Wav2Vec2-Large 和 HuBERT-Large，音频统一重采样到 16 kHz，对最后一层 hidden state 做 attention-mask 加权 temporal mean pooling。每个 pair 保存 anchor 和 AuK Content Editing 两个 1024-d embedding，shape 为 `(3734, 2, 1024)`。完整 embedding 位于 `04_embeddings/`。

## 4. AuK 五分类 probe

类别顺序：`[Real, TTS, SE, TSE, Content Editing]`。使用同一批 3734 pairs、speaker-disjoint split、StandardScaler + multinomial LogisticRegression，train/validation/test 为 34/7/7 speakers 和 2533/635/566 pairs。

Wav2Vec2 test：accuracy 75.55%，macro-F1 75.47%。矩阵：

```text
             Real  TTS  SE  TSE  CE
Real          419    1  22   23 101
TTS             5  548   1    2  10
SE             10    1 418  121  16
TSE            32    1 167  340  26
Content Edit   98   17  21   17 413
```

HuBERT test：accuracy 92.97%，macro-F1 92.96%。矩阵：

```text
             Real  TTS  SE  TSE  CE
Real          558    0   2    1   5
TTS             0  563   0    0   3
SE              0    1 479   81   5
TSE             2    0  68  490   6
Content Edit    9    9   3    4 541
```

结果说明：Wav2Vec2 中 Content Editing 与 Real 有较多混淆；HuBERT 中 Content Editing 仍较容易被单独识别。SE/TSE 仍是最稳定的主要混淆对，TTS 最容易区分。

## 5. CosyEdit specialist baseline

为了把 Content Editing 纳入 AuK 与专用模型的对照框架，选择 CosyEdit 作为 specialist Content Editing 模型。官方接口为：`inference_edit(target_text, original_text, original_speech)`。部署使用官方仓库 commit `a2aea98`、官方 `CJY/CosyEdit` checkpoint、独立环境 `/data/CosyEdit/venv`，模型采样率 22050 Hz。

15 条 pilot 已全部成功，输出和日志位于 `05_cosyedit_pilot/`。pilot 共耗时约 162 秒，单条约 7–14 秒。

全量 3734 条任务使用 4 个独立 worker 并行，每个 worker 单独加载模型并处理不重叠 shard，支持断点续跑。归档创建时的确切进度见 `inventory.json` 和 `05_cosyedit_full_generation/`；全量输出 WAV 仍在原路径 `/data/CosyEdit/content_edit_full_3734_20261006/` 持续生成。

## 6. 后续实验

CosyEdit 全量完成后，应在同一批 3734 pairs 上提取 CosyEdit embedding，并构造：

```text
AuK:       Real / AuK-TTS / AuK-SE / AuK-TSE / AuK-CE
Specialist:Real / CosyVoice3-TTS / MossFormerGAN-SE / DAE-TSE / CosyEdit-CE
```

两组使用相同 encoder、speaker split、pooling 和 linear probe，比较统一模型与专用模型的 task boundary。

## 归档范围

压缩包包含文本审核原始记录、15 条 AuK 试听样本、CosyEdit pilot 音频、embedding、probe 报告、部署/生成脚本和全量生成日志。未复制 AuK 3734 条全量 WAV、CosyEdit 3734 条全量 WAV、CosyEdit checkpoint 和原始 5369-pair matched audio；这些文件通过上述绝对路径引用，并在当前机器保留。
"""
    (ARCHIVE_ROOT / "REPORT_CN.md").write_text(report, encoding="utf-8")
    (ARCHIVE_ROOT / "NOT_INCLUDED.md").write_text(
        "本归档不包含大体积全量音频和模型 checkpoint。它们仍保留在：\n\n"
        f"- AuK full audio: `{AU_OUT}`\n"
        f"- CosyEdit full audio: `{COSY_OUT}`\n"
        f"- CosyEdit checkpoint: `{COSY / 'pretrained_models/CosyEdit'}`\n"
        f"- Original matched audio: `{SOURCE_ROOT}`\n",
        encoding="utf-8",
    )

    # Zip the archive directory.
    zip_path = shutil.make_archive(str(ZIP_BASE), "zip", root_dir=ARCHIVE_ROOT.parent, base_dir=ARCHIVE_ROOT.name)
    print(json.dumps({"archive": zip_path, "root": str(ARCHIVE_ROOT), "inventory": inventory}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
