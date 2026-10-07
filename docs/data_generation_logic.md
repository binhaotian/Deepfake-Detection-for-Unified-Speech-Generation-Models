# AuK Matched Dataset: Data Construction and Generation Logic

本文档给出本项目数据集的完整构造逻辑，面向没有阅读过本仓库的读者，说明数据
从哪里来、为什么这样筛选、如何构造 matched pair、如何合成 SE/TSE 输入、如何
调用 AuK，以及最终数据的规模、元数据和限制。此前的 100-pair 数据只用于 pilot，
不是正式全量集。

## 1. 研究目标与总体设计

本数据集用于研究同一个 unified generation model 在不同音频任务下产生的输出，
是否具有相似的 deepfake detector score 或内部 representation。第一阶段包含三
类 AuK 任务：text-to-speech (TTS)、speech enhancement (SE) 和 target speaker
extraction (TSE)。

研究单位不是孤立的 wav 文件，而是以一条真实语音为中心的 matched pair。每个
pair 尽量固定 target speaker、linguistic content、source-domain 和 anchor
recording，只改变任务或输入扰动：

```text
same ASVspoof bona fide anchor
├── Real:    anchor.wav
├── AuK-TTS: same-speaker reference + anchor transcript
├── AuK-SE:  anchor + environmental noise
└── AuK-TSE: anchor + different-speaker overlapping speech
```

这里的 matched 指协议和条件匹配，不表示四个 wav 是同一真人波形的平行录音。
TTS 必然重新合成声学内容；SE 和 TSE 则是对 anchor 加入可控干扰后再交给 AuK
处理。

Timbre Editing 暂不放入主集，因为“改变音色后是否应该被 deepfake detector 判
为 spoof”在任务定义上并不明确。将它加入会把检测目标定义问题与跨任务痕迹问题
混在一起。

## 2. 数据来源

### 2.1 ASVspoof2019 LA

Real anchor 来自 ASVspoof2019 Logical Access (LA) evaluation split 的 bona fide
trials。使用 ASVspoof 文件本身，而不是用 VCTK 同 ID 文件替换，原因是：

1. ASVspoof bona fide 是后续 AASIST/ASVspoof-domain 分析的直接真实域；
2. 它已经是 benchmark 中实际使用的 16 kHz、单声道音频；
3. VCTK 0.92 提供的是 silence-trimmed、48 kHz recording，与 ASVspoof 文件不是
   同一波形，不能把二者视为 identical recording。

ASVspoof 到 VCTK 的 utterance-level mapping 来自：

```text
ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz
└── ASVspoof2019_LA_VCTK_MetaInfo.tsv
```

LA CM protocol 用于 bona fide/spoof 标签；LA ASV protocol 用于确定 target ID
及其 enrollment recordings。

### 2.2 VCTK 0.92

VCTK 0.92 提供 VCTK utterance transcript、speaker identity、gender 和 speaker
metadata，以及与 ASVspoof utterance 的语义对应关系。VCTK 0.92 包含约 110 位
说话人、约 44,583 条文本和约 88,328 个 silence-trimmed FLAC。主集不把 VCTK
waveform 作为 Real anchor，而只使用其文本和 metadata；Real 始终来自 ASVspoof
LA eval。

### 2.3 ASV enrollment recordings

LA eval 的 ASV enrollment recordings（通常是 `LA_E_A*.flac`）是独立真实录音，
用于为 TTS 提供同 target speaker 的 voice reference，并为 TSE 提供另一 speaker
的 interfering speech。anchor 不作为自己的 TTS reference；interferer 不与
target speaker 相同。

### 2.4 ESC-50 environmental noise

SE 噪声来自 ESC-50 的环境声音子集。正式全量使用六类、每类 40 条、共 240 条：

```text
engine, rain, sea_waves, thunderstorm, water_drops, wind
```

不使用 speech 或 music 类别。每条噪声的类别、文件名和原始 URL 保存在
`noise_source_metadata.csv`。

## 3. Anchor 筛选

ASVspoof2019 LA eval 中有 7,355 条 bona fide。筛选结果如下：

| 筛选阶段 | 数量 |
|---|---:|
| LA eval bona fide | 7,355 |
| 属于有 ASV target enrollment 的 48 位 target speakers | 5,370 |
| 具有官方 VCTK transcript、可进入主 matched set | 5,369 |
| non-target speakers 的 bona fide | 1,985 |
| 因官方 transcript 缺失而排除 | 1 |

正式主集定义为 **5,369 个 anchors，覆盖 48 位 target speakers**。唯一因
transcript 缺失排除的 target-speaker bona fide 是：

```text
ASVspoof ID: LA_E_3017534
VCTK ID:    p351_361
```

我们不把全部 7,355 条 bona fide 强行放在一起，因为没有 ASV target enrollment
的 1,985 条无法按照同样的 reference/interferer 规则构成主 matched set。它们
可以作为 supplementary set，但不混入当前主实验。

每个保留 anchor 必须满足：ASVspoof LA eval 的 bona fide FLAC 存在；mapping 中有
唯一 VCTK utterance ID；VCTK 0.92 中官方 transcript 存在；VCTK speaker 能通过
ASV enrollment 映射到唯一 target speaker。

### 3.1 VCTK waveform 缺失不等于 anchor 无效

VCTK 0.92 的发布包存在删除、缺失或只保留一个 microphone 版本的录音。因此有些
ASVspoof-to-VCTK ID 在 VCTK audio directory 中找不到对应 wav/FLAC。这不影响本
主集：真实 anchor 使用 ASVspoof 文件，TTS 文本使用 VCTK txt。当前设计需要
VCTK transcript 和 identity metadata 完整，不要求 VCTK waveform 替换 ASVspoof
anchor。

## 4. Pair 结构与类别含义

每个 pair 以一条 ASVspoof bona fide anchor 为中心：

```text
pair_00001/
├── anchor.wav
├── tts_reference.wav
├── auk_tts.wav
├── se_noise.wav
├── se_input.wav
├── auk_se.wav
├── tse_interferer.wav
├── tse_input.wav
└── auk_tse.wav
```

| 文件 | 含义 | 是否作为 Real 类别 |
|---|---|---|
| `anchor.wav` | ASVspoof bona fide anchor | 是 |
| `tts_reference.wav` | 独立同 speaker reference | 否，条件输入 |
| `auk_tts.wav` | AuK TTS 输出 | 否，生成条件 |
| `se_noise.wav` | 实际加入的噪声 | 否，中间量 |
| `se_input.wav` | anchor + noise | 否，不能当作 Real |
| `auk_se.wav` | AuK SE 输出 | 否，生成条件 |
| `tse_interferer.wav` | 不同 speaker 的干扰语音 | 否，中间量 |
| `tse_input.wav` | anchor + interferer | 否，不能当作 Real |
| `auk_tse.wav` | AuK TSE 输出 | 否，生成条件 |

保存 `se_input.wav` 和 `tse_input.wav` 是为了让输入构造、混音强度和 AuK 条件
可以被独立审计；它们不是额外的 detector 类别。

## 5. TTS reference 选择

对于每一位 target speaker，从该 speaker 的 ASV enrollment recordings 中选择一
条固定 reference：按录音时长排序，选择中位 enrollment，并在该 speaker 的全部
anchors 中复用。reference 是独立真实录音，不携带 anchor 的相同声学片段；用中位
时长可以避免 reference 过短、表达 speaker identity 不足，或过长造成不必要计算。

AuK-TTS 的条件为：

```text
fixed same-speaker reference + anchor transcript
```

所以 target speaker 和 linguistic content 与 anchor 对齐，变化主要来自 synthesis
process。

## 6. TSE interferer 选择

对于每一个 target speaker，先固定三位不同的、优先同性别的 interferer speakers：

1. 候选为其余 47 位 target speakers；
2. 优先保留与 target gender 相同的候选；
3. 若同性别候选不足，退化到所有不同 speaker；
4. 由 target speaker ID 的稳定 hash 确定起始位置，取旋转列表中的前三位；
5. 同一 target speaker 的 anchors 按 speaker-local index 在三位 interferer 间轮换。

对具体 anchor，再从选定 interferer speaker 的 enrollment recordings 中进行
duration matching：优先选择不短于 target anchor 且长度差最小的录音；如果没有
足够长的录音，则选择最长录音并重复铺满。精确 recording ID 和重复次数写入
manifest。这样保留真实 overlapping-speaker 输入，同时避免单个固定 interferer
主导全体结果。

## 7. 音频规范与混音公式

所有生成前音频统一为 mono、24 kHz；混音使用 float32，最终以 16-bit PCM WAV
保存。AuK 原生推理采样率为 24 kHz，因此混音在 24 kHz 完成。

令 anchor 为 $x(t)$，噪声为 $n(t)$，interferer 为 $i(t)$：

$$
RMS(z) = \sqrt{\frac{1}{T}\sum_t z(t)^2 + 10^{-12}}.
$$

### 7.1 Speech enhancement 输入

正式版本固定 `SNR = 0 dB`。噪声先循环或截断到 anchor 长度，再缩放为：

$$
n'(t) = n(t) \cdot
\frac{RMS(x)}{10^{SNR/20} RMS(n)}.
$$

因此当 SNR=0 dB 时，$RMS(n')=RMS(x)$，并构造：

$$
y_{SE}(t)=x(t)+n'(t).
$$

AuK 指令为：

```text
Remove only the background noise, preserve everything else,
and output audio of the same length.
```

噪声类别按六类循环，类别内 clip 按文件名确定性轮换。5,369 个 pairs 中，六类
使用次数为 895、895、895、895、895、894；240 条 noise clips 全部使用。

### 7.2 Target speaker extraction 输入

正式版本固定 `SIR = 0 dB`。interferer 先循环或截断到 anchor 长度，再缩放为：

$$
i'(t) = i(t) \cdot
\frac{RMS(x)}{10^{SIR/20} RMS(i)}.
$$

构造：

$$
y_{TSE}(t)=x(t)+i'(t).
$$

target 和 interferer 从 $t=0$ 开始完全重叠，不使用隐藏的随机 offset。AuK 指令为：

```text
Keep only the speaker who says "{anchor transcript}"
and remove all other speakers.
```

### 7.3 Pair-level shared scaling

如果 anchor、reference、noise、interferer、SE input、TSE input 的共同峰值超过
0.98，则对该 pair 的六个生成前文件使用同一缩放因子：

$$
\alpha = \frac{0.98}{\max |z|}, \qquad z \leftarrow \alpha z.
$$

所有信号共同缩放不会改变 SNR/SIR 比值，也避免分别写入 PCM 时产生不一致 clipping。
实际测量的 SNR/SIR 写入 manifest，而不是只记录目标值。

## 8. AuK 推理逻辑

每个 pair 生成三条 AuK 输出：

1. `auk_tts.wav`：使用固定同 speaker reference 和 anchor transcript，输出时长设为
   anchor 时长；
2. `auk_se.wav`：输入 `se_input.wav`，要求去除背景噪声并保持时长；
3. `auk_tse.wav`：输入 `tse_input.wav`，要求保留 target speaker 并移除其他 speaker。

正式配置：

```text
model: AuK Base
checkpoint: ckpts/AuK/auk_base.safetensors
dtype: BF16
NFE: 32
CFG strength: 2.0
sway sampling coefficient: -1.0
batch size: 4
sample rate: 24 kHz
```

推理按任务分组，再按输入时长排序组成 batch，以减少 variable-length padding。模型
只加载一次。每个 pair 有固定基础 seed，三个任务使用 task-specific offset：

```text
tts_seed = seed
se_seed  = seed + 1,000,000
tse_seed = seed + 2,000,000
```

batch 内每条样本使用独立 seed。因此 checkpoint、batch size、任务顺序、instruction
和 seed policy 都是数据协议的一部分，重新生成时必须保持一致。

## 9. Manifest、审计和质量控制

`manifest.tsv` 对每个 pair 记录 ASVspoof anchor ID、VCTK ID、target ID、speaker、
gender、transcript、TTS reference ID、interferer target/speaker/recording ID、
noise ID/category、nominal/measured SNR/SIR、anchor duration、noise/interferer
repeat 次数、base seed、三个 task seeds 和 pair 相对目录。

准备阶段保存：

```text
preparation_report.json
excluded_anchors.tsv
noise_source_metadata.csv
```

AuK 生成阶段保存：

```text
generation_config.json
generation_progress.jsonl
generation_report.json
```

生成报告检查每个输出是否存在、可读取、mono、24 kHz、无 NaN/Inf、时长与目标
一致，并保存 peak、clipping fraction 和 SHA-256。不能根据 AASIST score 或主观
听感事后删除样本；异常应记录并披露，否则会引入 detector-driven selection。

## 10. 数据规模与当前状态

已完成的全量输入准备规模是：

```text
5,369 pairs × 6 input/intermediate WAVs = 32,214 WAVs
```

完整 AuK 输出规模是：

```text
5,369 pairs × 3 tasks = 16,107 AuK WAVs
```

最终每组 9 个 WAV，总规模为：

```text
5,369 pairs × 9 WAVs = 48,321 WAVs
```

当前仓库状态：`data/matched_full_5369_snr0_batch4/` 保存生成前的 5,369 个 pair；
`data/matched_full_5369_snr0_batch4_generated/` 已包含全部 16,107 条 AuK 输出，
即 5,369 个完整 pair、每组 9 个 WAV，共 48,321 个 WAV。
全量自动校验覆盖了 16,107 条模型输出，未发现缺失、格式、时长或 NaN/Inf 错误；
43 条输出超过预设的 0.1% clipping-fraction 告警线，均保留并记录在
`generation_report.json`，没有据此筛除样本。`data/matched_full_100_snr0_batch4/`
是此前的 100-pair pilot。

## 11. 论文中应说明的限制

第一，matched pair 控制 speaker、文本、anchor 和输入构造条件，但不意味着不同
任务输出具有相同低层波形。TTS 是重新合成，SE/TSE 是对真实 anchor 加扰后的处理。

第二，ASVspoof bona fide 并不是为本实验专门采集的同文本、同条件平行录音。本工作
构造的是 anchor-centric、protocol-level matched set，而不是严格 paired recording
corpus。

第三，0 dB SNR 和 0 dB SIR 是第一版固定工作点，代表明显会影响听感和处理难度的
条件，不覆盖全部真实环境。多 SNR/SIR 扩展应将条件作为显式分层变量。

第四，ESC-50 噪声和 ASV enrollment recordings 在 5,369 个 anchors 中会重复使用。
这种重复不是隐藏随机性：clip、speaker、recording ID 和选择规则均写入 manifest。

第五，主集只覆盖有 ASV target enrollment 的 48 位 speakers。其余 19 位 non-target
speakers 的 bona fide 可作为 supplementary set，但不能在没有重新定义
reference/interferer 规则的情况下直接并入主集。
