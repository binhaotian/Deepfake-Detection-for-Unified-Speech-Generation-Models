# Model/Data Inventory

审计日期：2026-10-06

## 更正结论

之前的结论不准确。之前只检查了仓库内的 `/root/AuK/data`，没有检查同一台机器上的外部实验目录 `/data`。三类专用模型的完整结果确实已经保存，没有丢失；只是大体量音频、embedding 和 probe 报告没有复制进 Git 仓库的 `data/` 目录。

这里的三个专用模型是：

1. TTS：CosyVoice3
2. SE：MossFormerGAN-SE
3. TSE：DAE-TSE

## 三类专用模型的完整输出

| 任务 | 模型 | 完整输出目录 | 已核对结果 | 音频格式 |
|---|---|---|---:|---|
| TTS | CosyVoice3 | `/data/specialist_outputs/cosyvoice3/` | 5,369 个 `specialist_tts.wav` | 24 kHz、单声道、PCM16 |
| SE | MossFormerGAN-SE | `/data/specialist_outputs/mossformer_gan_se/` | 5,369 个标准 `specialist_se.wav`，失败 0 | 16 kHz、单声道、PCM16 |
| TSE | DAE-TSE | `/data/specialist_outputs/dae_tse/` | 5,369 个 `specialist_tse.wav`，失败 0 | 16 kHz、单声道、PCM16 |

三个目录都按 `pair_*/` 保存，每个 pair 对应一条专用模型输出。已直接用文件系统计数复核：

- CosyVoice3：5,369 个 `specialist_tts.wav`；目录约 576 MB；
- MossFormerGAN-SE：5,369 个标准 `specialist_se.wav`；目录约 572 MB；
- DAE-TSE：5,369 个 `specialist_tse.wav`；目录约 681 MB。

MossFormerGAN-SE 目录中另外还保留了重试试听包里的少量 WAV；上表的 5,369 个是主输出集合，不把这些辅助文件重复计入。

## 生成记录和完整性报告

- CosyVoice3 记录：`/data/specialist_experiment_documentation_20260930/generation_records/cosyvoice3_generation_record.md`
- MossFormerGAN-SE 汇总：`/data/specialist_outputs/mossformer_gan_se/summary.json`
  - `requested_count=5369`
  - `final_output_count=5369`
  - `final_failed_count=0`
  - `quality_ok_count=5369`
- DAE-TSE 汇总：`/data/specialist_outputs/dae_tse/run_report.json`
  - `requested_count=5369`
  - `success_count=5369`
  - `failure_count=0`
  - 有 605 条非致命的 `no_isolated_jump_candidates` 检查告警，但没有失败输出。

因此，之前所说的“专用模型没有输出”应更正为：“专用模型输出在 `/data/specialist_outputs`，不在仓库内的 `/root/AuK/data`。”

## 专用模型 embedding 和 probe

这些实验结果也已经保存：

- HuBERT embedding：`/data/embeddings/specialist_matched_hubert/`
- Wav2Vec2 embedding：`/data/embeddings/specialist_matched_wav2vec2/`
- 两者的特征形状均为 `[5369, 7, 1024]`；每个目录约 148 MB；
- 三任务 probe 报告：
  - `/data/specialist_experiment_documentation_20260930/embedding_and_probe/specialist_task_hubert_report.json`
  - `/data/specialist_experiment_documentation_20260930/embedding_and_probe/specialist_task_wav2vec2_report.json`
- 另有包含 `real/tts/se/tse` 的四分类 probe 报告：`fourclass_specialist_*_report.json`。

## 仓库内的 AuK 数据

仓库 `data/` 下保存的是 AuK 模型的结果，不是上面三类 specialist 的完整原始输出：

| 数据根目录 | 规模 | TTS | SE | TSE | 说明 |
|---|---:|---:|---:|---:|---|
| `matched_full_5369_snr0_batch4_generated` | 5,369 pairs | 5,369 | 5,369 | 5,369 | AuK Base 全量结果 |
| `tts_full_v2_v3_5369/v2_active_duration` | 5,369 pairs | 5,369 | 5,369 | 5,369 | AuK TTS V2 变体 |
| `tts_full_v2_v3_5369/v3_text_duration` | 5,369 pairs | 5,369 | 5,369 | 5,369 | AuK TTS V3 变体 |
| `matched_full_5369_rmsnorm` | 5,369 pairs | 5,369 | 5,369 | 5,369 | RMS-normalized AuK 版本 |
| `matched_full_100_snr0_batch4` | 100 pairs | 100 | 100 | 100 | 100-pair pilot |

## 专用模型试音包

已从完整 specialist 结果中抽取同一组 15 个 pair，便于横向比较；试音包保留原始采样率，没有重新采样或归一化：

- `data/audition_15_specialist_cosyvoice3_20261006.zip`
- `data/audition_15_specialist_mossformer_se_20261006.zip`
- `data/audition_15_specialist_dae_tse_20261006.zip`
- `data/audition_15_specialist_tasks_20261006.zip`：三类合并包，共 45 个 WAV

展开目录为 `data/audition_15_specialist_20261006/`，每个任务目录有 15 个 WAV 和 `manifest.tsv`。打包脚本为 `scripts/package_specialist_audition_15.py`。
