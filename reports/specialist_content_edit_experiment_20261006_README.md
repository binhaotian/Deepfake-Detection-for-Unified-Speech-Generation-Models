# Specialist Content Editing: five-class probe (2026-10-06)

本包记录 CosyEdit 作为专用 Content Editing 模型加入后，与已有专用 TTS/SE/TSE 模型进行统一五分类 probe 的结果。

## 实验输入

每个 matched pair 使用五个类别：

1. Real：真实 anchor
2. TTS：CosyVoice3 专用 TTS
3. SE：MossFormerGAN-SE 专用 speech enhancement
4. TSE：DAE-TSE 专用 target speaker extraction
5. Content Editing：CosyEdit 专用 content editing

CosyEdit 共接受并生成 3734 条 pair；专用 TTS/SE/TSE 的 embedding 使用此前完整 5369-pair specialist 特征中的对应流，并取相同 3734 个 pair 的交集。

## 特征提取

- Frozen Wav2Vec2 Large：最后一层 hidden state，attention-mask 加权 temporal mean pooling，1024 维。
- Frozen HuBERT Large：最后一层 hidden state，attention-mask 加权 temporal mean pooling，1024 维。
- 本轮按用户要求暂不使用 Whisper。

CosyEdit paired embedding 的数组形状为 `(3734, 2, 1024)`，顺序为 `[anchor, cosyedit_content_edit]`。

## Probe

- StandardScaler + multinomial LogisticRegression
- `max_iter=5000`
- speaker-disjoint split
- train/validation/test speakers = 34/7/7
- train/validation/test pairs = 2533/635/566
- test samples = 2830（5 类 × 566）
- random seed = 20260925

## Specialist 五分类结果

类别顺序：`[Real, TTS, SE, TSE, Content Editing]`

### Wav2Vec2 Large

- Accuracy: 0.830035
- Macro-F1: 0.830461

```text
[[440,  2, 66, 25, 33],
 [  3,546,  1,  3, 13],
 [ 48,  1,437, 56, 24],
 [ 22,  0, 62,461, 21],
 [ 32, 13, 33, 23,465]]
```

### HuBERT Large

- Accuracy: 0.975265
- Macro-F1: 0.975332

```text
[[524,  1, 36,  3,  2],
 [  0,566,  0,  0,  0],
 [  4,  0,561,  1,  0],
 [  1,  0,  0,563,  2],
 [  1,  1, 16,  2,546]]
```

## 对照：AuK 五分类

相同 speaker split、pair 数量、probe 和类别顺序下：

| Encoder | AuK accuracy | Specialist accuracy | Specialist - AuK |
|---|---:|---:|---:|
| Wav2Vec2 Large | 0.755477 | 0.830035 | +0.074558 |
| HuBERT Large | 0.929682 | 0.975265 | +0.045583 |

这不是“CosyEdit 一定比 AuK 更容易被 deepfake detector 识别”的单独结论；它表示在当前 frozen encoder + 线性 probe 的五类 representation 空间中，专用模型组合的任务/系统边界比 AuK 组合更清晰。后续仍需结合二分类、pair-level 分析及更严格的 cross-system 控制来解释。

## 文件

- `embeddings/wav2vec2/features.npy`：CosyEdit paired features
- `embeddings/wav2vec2/manifest.tsv`
- `embeddings/wav2vec2/config.json`
- `embeddings/hubert/features.npy`
- `embeddings/hubert/manifest.tsv`
- `embeddings/hubert/config.json`
- `probe/wav2vec2/report.json`
- `probe/wav2vec2/split.tsv`
- `probe/hubert/report.json`
- `probe/hubert/split.tsv`

原始 CosyEdit 音频及全量生成 manifest 位于：
`/data/CosyEdit/content_edit_full_3734_20261006/`
