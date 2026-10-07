# Starting a New Codex Session

The repository-level `AGENTS.md` is the persistent entry point. Start Codex with `/root/AuK` as the working directory so the instructions apply to the project.

Recommended launch command:

```bash
codex -C /root/AuK
```

Or start the session with the initial instruction in one command:

```bash
codex -C /root/AuK '请先按照 AGENTS.md 读取项目状态文档，先总结当前研究状态，不要立即运行模型。'
```

## Minimal prompt

Use this when opening a fresh session for discussion:

```text
请先按照 /root/AuK/AGENTS.md 的顺序读取项目状态文档，特别是
CODEX_PROJECT_STATE.md、docs/codex/RESEARCH_BRIEF.md、
docs/codex/DECISIONS_AND_OPEN_QUESTIONS.md、
docs/codex/ARTIFACT_INDEX.md 和相关 EXP 记录。

先用你自己的话总结：研究问题、已经完成的实验、当前能支持和不能支持的结论、
Whisper 状态、当前最重要的开放问题。暂时不要运行模型或修改数据。
```

## Prompt for continuing an experiment

```text
这是 /root/AuK 的 unified speech model / deepfake 研究。
请先读取 AGENTS.md 和它规定的状态文档，并运行：

python scripts/audit_project_state.py

我要继续处理的问题是：<在这里写具体问题>。

开始前请先确认：
1. 这个问题对应哪个现有 EXP；
2. 哪些输入和结果已经存在，避免重复生成；
3. 计划会不会改变既有数据协议或研究问题；
4. 如果需要 GPU，先核验当前 GPU 和正确的 Python 环境。

先汇报核验结果和执行计划，再开始实质工作。
```

## Prompt for a read-only scientific discussion

```text
请读取 /root/AuK/AGENTS.md 规定的项目文档。这个窗口只讨论实验设计和论文解释，
不要启动推理、训练、下载或批量修改文件。

本次问题：<问题>。

回答时请区分：磁盘中已验证的事实、目前的科学解释、替代解释、尚未被实验支持的说法。
```

## Prompt for a long-running generation or extraction task

```text
请先读取 /root/AuK/AGENTS.md 和状态文档，并运行项目审计。
本次任务是：<任务>。

要求：
- 为任务分配新的 EXP 编号；
- 使用已有 manifest 和 speaker split；
- 输出放到 /data，不覆盖权威源文件；
- 支持断点续跑和 skip-existing；
- 保存完整 command、config、seed、manifest、console log 和 report；
- 完成后更新 CODEX_PROJECT_STATE.md、EXPERIMENT_LEDGER.md、ARTIFACT_INDEX.md；
- 最后重新运行 scripts/audit_project_state.py。
```

## What not to paste every time

There is no need to paste the full project history into every prompt if:

1. the new session starts in `/root/AuK`;
2. `AGENTS.md` is present;
3. the state documents are current.

The user prompt should specify only the new objective and any change in authorization or scientific scope. If starting outside `/root/AuK`, explicitly tell Codex to `cd /root/AuK` and read `/root/AuK/AGENTS.md` before acting.
