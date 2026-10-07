# AuK 实验进展摘要包

生成日期：2026-09-29

本压缩包用于记录 AuK matched dataset 生成完成之后的主要实验设置和结果。

本包**不包含**音频、模型权重或 embedding 原始文件，只包含说明文档、配置文件、分类结果、分布分析结果和混淆矩阵图片。

## 目录说明

### `docs/`

- `experiment_log_v2_v3_and_asvspoof.md`
  - 当前最重要的实验日志。
  - 包含 V1/V2/V3 的定义、encoder、embedding 提取方法、线性 probe、speaker split、实验结果和 ASVspoof baseline。
- `data_generation_logic.md`
  - 说明 5369 个 matched pairs 的构造方式，包括 Real、AuK-TTS、AuK-SE、AuK-TSE、reference、noise 和 interferer。
- `data_README.md`
  - 数据目录和生成数据的简要说明。

### `configs/`

- `v1_generation_config.json`
  - 初始完整 matched set 的 AuK 生成配置。
- `v2_generation_config.json`
  - V2：基于 anchor 有效语音时长，并进行 boundary 对齐。
- `v3_generation_config.json`
  - V3：基于文本长度估计目标时长，并进行 boundary 对齐。

三个版本使用相同的核心 TTS prompt：

```text
Say the following with the same voice: '{anchor transcript}'
```

V2/V3 的主要差别不是 prompt，而是 TTS 时长控制和前后静音处理。

### `probe_results/`

所有结果都是冻结 encoder 后接 `StandardScaler + multinomial LogisticRegression` 的分类结果。

四分类标签为：

```text
real / tts / se / tse
```

- `wav2vec2_v1_fourclass.json`
- `wav2vec2_v2_fourclass.json`
- `wav2vec2_v3_fourclass.json`
  - Wav2Vec 2.0 Large 的 V1/V2/V3 结果。
- `hubert_v1_fourclass.json`
- `hubert_v2_fourclass.json`
- `hubert_v3_fourclass.json`
  - HuBERT Large 的 V1/V2/V3 结果。
- `whisper_fourclass.json`
- `whisper_rmsnorm_fourclass.json`
  - Whisper Large-v3 的原始音频和 RMS normalization 结果。
- `asvspoof_wav2vec2.json`
- `asvspoof_hubert.json`
  - 在 ASVspoof2019 LA 上进行 bona fide/spoof 二分类的 baseline。

### `distribution/`

三个 encoder 的 embedding 分布分析：

- `wav2vec2_distribution.json`
- `hubert_distribution.json`
- `whisper_distribution.json`

内容包括 PCA、类别中心距离、音频时长、RMS 和每个样本的投影信息。

### `plots/`

- `confusion_matrices_original_vs_rmsnorm_counts.png`
  - 原始音频和 RMS normalization 的计数混淆矩阵。
- `confusion_matrices_original_vs_rmsnorm_row_normalized.png`
  - 行归一化混淆矩阵。

## 关键结果摘要

### Wav2Vec 2.0

```text
V1: Accuracy 82.17%, Macro-F1 82.19%
V2: Accuracy 76.81%, Macro-F1 76.83%
V3: Accuracy 77.63%, Macro-F1 77.54%
```

### HuBERT

```text
V1: Accuracy 94.00%, Macro-F1 94.01%
V2: Accuracy 93.65%, Macro-F1 93.67%
V3: Accuracy 93.59%, Macro-F1 93.61%
```

### Whisper Large-v3

```text
原始音频：Accuracy 91.05%, Macro-F1 91.12%
RMS normalization：Accuracy 90.77%, Macro-F1 90.85%
```

当前主要观察是：`real` 与 `tts` 较容易区分，`se` 与 `tse` 之间混淆更多；V2/V3 对 Wav2Vec2 有一定影响，但对 HuBERT 影响很小。
