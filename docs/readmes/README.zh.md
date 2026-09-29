<div align="center">
  <h1>Wald-Q4B v1.1</h1>
  <p><strong>直接决策，按需思考，返回概率。</strong></p>
  <p><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="https://huggingface.co/Harry19081/Wald-4B">Weights on Hugging Face</a> · <a href="../api.md">API</a></p>
</div>

**Wald-Q4B v1.1 是一个开放权重的 4B 决策模型：给它一段状态和一组选项，它为每个选项返回校准过的概率。** 它面向构建 agent 和数据流水线的开发者：需要一个快速、可自行部署的组件来选择工具、路由请求、分类输入，或判断是否需要向用户澄清。和聊天模型不同，它不生成需要再解析的答案，而是一遍读出所有选项的概率（effort `none` 时单张 RTX PRO 6000 上中位延迟 33 ms），也可以先思考再回答。它提供与 Jev 兼容的 `POST /v1/systemone` API，基于 Qwen3.5-4B-Base，以 Apache-2.0 发布。

Wald-Q4B 是 TypeSafe 托管 Jev API 之外、可自行部署的独立替代方案。它不是 Jev，不含 Jev 权重，与 TypeSafe AI 没有隶属或背书关系。Hugging Face 仓库为 `Harry19081/Wald-4B`（旧名 Wald-4B）。

**W**ait **A** bit, **L**ook, then **D**ecide：稍等一下，看清楚，再决定。名字也致敬序贯分析先驱 Abraham Wald：证据足够时就停止。

## 一览

- **4B 参数**，基于 Qwen3.5-4B-Base，BF16 权重（8.4 GB）。
- **每个选项都有概率。** 题型：`choice`（1–255 个命名选项）、`noul`（是/否）、`score`（有序等级）。
- **可调思考程度**：`none`、`low`、`medium`、`high`，以及多次思考的 `high-k`。
- **完整 Decision Index 0.2.1：54.59**，使用 `high`，150,317 个请求全部成功。作者自测，等待维护者验证。
- **JevBench 公开集：203/231**，使用 `none`，ECE 0.041，p50 33 ms / p95 168 ms。用 JevBench 自己的评测工具自评。
- **可自行部署的 API**：`POST /v1/systemone`，最多 131,072 个提示 token。

## 快速开始

在装有 NVIDIA GPU 和 [`uv`](https://docs.astral.sh/uv/) 的 Linux 机器上：

```sh
hf download Harry19081/Wald-4B --revision v1.1 --local-dir ./Wald-Q4B
cd Wald-Q4B
EFFORT=none ./run.sh "$PWD"     # 一遍读出，延迟最低
# ./run.sh "$PWD"               # 默认 high，即 Decision Index 的评测配置
```

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The customer wants to return a damaged kettle.",
    "effort": "medium",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Choose the support queue.",
        "criteria": {
          "returns": "Returns and refunds",
          "delivery": "Delivery tracking",
          "other": "Other enquiries"
        }
      }
    }
  }'
```

`route` 的答案包含选中的键，以及 `returns`、`delivery`、`other` 各自的概率。请求与响应字段、是非题和评分题、澄清判断示例见 [API 说明](../api.md)。`GET /health` 返回当前生效策略。服务使用 vLLM 0.30.0 和仓库内的 `wald-serve`；Docker 与精确评测配置见 [RUNBOOK.md](../../RUNBOOK.md)。普通文本生成接口不会复现决策 API 的读出流程。

## 思考程度

直接决策可从 `none` 开始；按置信度触发思考用 `medium`；复现 v1.1 的评测配置用 `high`。

| Effort | 何时思考 | 思考预算 |
|---|---|---|
| `none` | 直接读取选项概率 | 不生成思考文本 |
| `low` | 初次最高概率 < 0.5 | 最多 512 token |
| `medium` | 初次最高概率 < 0.7 | 最多 512 token |
| **`high`（默认）** | 每个符合条件的问题 | 最多 512 token |
| `high-k2` … `high-k8` | 多次思考，平均答案分布 | 每次最多 512 token |

思考适用于 2–26 个选项且上下文空间足够的问题。更多选项采用分组读取，再比较各组优胜项；若放不下思考文本，则保留初次答案。提高 effort 会增加计算量，但不保证每道题都更准确。**54.59 仅对应 `high`；203/231 仅对应 `none`。**

通过 `EFFORT=medium ./run.sh "$PWD"` 设置服务默认值，也可在单次请求中传入 `"effort": "none"` 覆盖。

## 如何工作

Wald 先从普通文本提示末尾读取选项字母的 logits，得到初始概率。如果 effort 策略触发思考，就生成一段短思考，再读取选项。返回的概率经过分桶温度校准。

v1.1 结合全参数决策训练、LoRA 精修、短思考蒸馏和 RLCD。训练使用我们的自生成决策语料 **WaldGen**，并混合公开训练数据。[数据来源](../../PROVENANCE.md) · [评测说明](../../CONTAMINATION.md)

## 评测

| 基准 | 配置 | 结果 | 状态 |
|---|---|---:|---|
| **Decision Index 0.2.1 完整套件** | v1.1 · `high` | **54.59** | 作者自测；[PR #30](https://github.com/apolinario/decision-index/pull/30) 等待维护者验证 |
| **JevBench 公开集（231 题）** | v1.1 · `none` | **203/231**（87.9%）· ECE 0.041 · Brier 0.188 | 自评；[issue #146](https://github.com/fstandhartinger/jevbench/issues/146) 请维护者自行测量 |

**Decision Index：** 150,317/150,317 个请求全部成功，包含 HLE。在单张 RTX PRO 6000 96 GB 上使用固定版本的复现工具运行。[完整结果](https://huggingface.co/datasets/Harry19081/Wald-Q4B-decision-index-results/tree/805716601b2466be324ed6716407b4c3d9267faa/runs/wald-q4b-22d0-f7-full021) · [分项成绩](../../evaluation/benchmark-summary.json) · [复现指南](../../RUNBOOK.md)

**JevBench：** 使用 JevBench 自己的命令行工具（`fstandhartinger/jevbench` @ `9ec6f15a`，`typesafe` 适配器），在单张 RTX PRO 6000 上通过本机回环逐条请求打包服务。easy 48/48、original 72/72、hard 83/111；不生成任何 token。`medium` 为 205/231，p95 1.80 s。公开题在开发中被用作计分板（从未作为训练数据），因此这不是留出集结果。JevBench 排行榜只在维护者自己跑过模型后才公布分数。

### 速度

使用 `none` 时，上述 JevBench 运行的单次决策**中位延迟 33 ms、p95 168 ms**。`high` 在同一 GPU 上的 32 请求串行预检中，**中位延迟为 821 ms**；这只是小规模预检，不代表完整套件延迟或 Decision Index 维护者的准入测试。effort、上下文长度、选项数量和并发都会影响速度。

## 相关项目与对比

多个项目在做带校准选项概率的结构化决策。以下名称归各自所有者；Wald 与它们都没有隶属关系。

- **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** 是 TypeSafe AI 通过 `/v1/systemone` API 提供的托管决策模型，权重不公开。Wald 接受相同的请求格式，运行在你自己的 GPU 上。
- **[Kev](https://github.com/jaredpalmer/kev)** 是 Jared Palmer 的开放权重项目，在 Qwen3.5 基座（0.8B、4B、9B）上加 LoRA 和指针头。Wald 的请求解析改编自 Kev 的 Apache-2.0 代码（见 [NOTICE](../../NOTICE)）；Wald 从语言模型头读取选项字母，不使用单独的头。
- **[Laya](https://huggingface.co/convaiinnovations/laya)**（[代码](https://github.com/NandhaKishorM/laya)）是开放权重的 421M ModernBERT-large 编码器加决策头。它比 Wald 小得多，最多读取 512 个 token。

**JevBench 公开集，同一批 231 题（数据集哈希相同），JevBench 命令行工具，由我们运行：**

| 系统 | 运行方式 | 答对 |
|---|---|---:|
| Wald-Q4B v1.1 · `none` | 自行部署，RTX PRO 6000，2026-09-29 | 203/231 |
| Jev（`jev-1.13.0`） | TypeSafe 托管 API，2026-09-25 | 200/231 |
| Laya（英文 checkpoint `55cf4c4e`） | 自行部署，NVIDIA L4，2026-09-26 | 134/231 |

231 题上 3 题的差距在多次运行和抽样的噪声范围内。公开题影响过 Wald 的开发；231 题中有 52 题的状态超过 Laya 的 512 token 窗口。

**Decision Index 0.2.1：**

| 系统 | 指数 | 来源 |
|---|---:|---|
| Jev（`jev-1.13.0`） | 57.91 | [排行榜](https://huggingface.co/spaces/multimodalart/jev-decision-index)，维护者运行（2026-09-28 数据） |
| Wald-Q4B v1.1 · `high` | 54.59 | 作者自测完整套件；尚未上榜（[PR #30](https://github.com/apolinario/decision-index/pull/30)） |
| Kev 9B | 38.48 | 排行榜，维护者运行（2026-09-28 数据） |
| Kev 4B | 34.64 | 排行榜，维护者运行（2026-09-28 数据） |

排行榜各行由维护者评分；Wald 的分数是用官方工具自测的，验证后可能变化。

## 常见问题

**有开源的 Jev 替代品吗？** Wald-Q4B 是一个开放权重的选择：Apache-2.0 的权重和服务代码，自己部署，提供与 Jev 兼容的 `/v1/systemone` API。Kev 和 Laya（见上）是其他开放项目。Wald 是独立项目，不是 TypeSafe 的发布。

**能用 Jev 客户端调用自部署模型吗？** 把客户端指向你自己的端点。内置服务在 `POST /v1/systemone` 接收 `state` 和带类型的 `questions`（`choice`、`noul`、`score`），按 TypeSafe 的答案键返回。服务不校验 API key。见 [API 说明](../api.md)。

**如何做工具路由，或判断是否需要向用户提问？** 把对话或任务作为 `state` 发送。工具路由：提一个 `choice` 问题，选项就是你的工具。是否澄清：提一个 `noul` 问题，例如“这个请求是否足够具体，可以不问就执行？”概率高就执行，概率低就提问，两个阈值都在你自己的验证数据上确定。模型只选工具，不生成工具参数。

**概率校准得怎么样？** 在 JevBench 公开集上使用 `none`，期望校准误差为 0.041（10 个分桶），Brier 分数为 0.188。温度是在我们自己开发数据的留出行上拟合的，不含 JevBench 题目，并排除了已知的 Decision Index 匹配项。置信度不是保证，请在你的任务上检查校准。

**能在单张 GPU 或笔记本上运行吗？** 内置服务需要 Linux 上的一张 NVIDIA GPU（vLLM 0.30.0）；BF16 权重为 8.4 GB。v1.1 的测量来自 RTX PRO 6000 96 GB；同一 4B 架构的早期版本也曾用 vLLM 在 24 GB 的 NVIDIA L4 上以 16K 上下文上限运行。内置服务不支持 CPU、Apple Silicon 和笔记本环境，也没有测试过。

**Kev 和 Wald、Laya 和 Wald 怎么选？** 三者都开放权重。Kev 在 Qwen3.5 基座上加指针头和 LoRA；Laya 是带决策头的小型编码器；Wald 是完整训练的 4B 解码器，可选思考。我们在同一协议下的测量见上表。请根据你自己的任务、延迟预算和硬件选择。

**能针对我的任务微调吗？** 它是标准的 Transformers checkpoint，常见的 LoRA 工具都适用。v1.0 时我们为单个任务训练 LoRA，每个花费 $0.12–$1.81 的 GPU 时间；这套工具尚未公开，这些 adapter 也未在 v1.1 上验证（[v1.0 说明](../../history/v1.0/README.md)）。

**许可证是什么？** 权重与代码为 Apache-2.0。基座 Qwen3.5-4B-Base 也是 Apache-2.0。部分公开训练数据有各自的条款或没有注明许可证，列在 [PROVENANCE.md](../../PROVENANCE.md)。模型与代码的许可证不授予这些文本的权利。

## 使用限制

置信度不保证正确性，应在自己的任务上验证阈值。超过上下文限制的提示会被拒绝，不会截断。已过滤已知的严格训练重叠，但无法排除语义重叠和预训练污染；开发过程中使用了可见的基准样本。各来源文本的使用权不同，详见[评测说明](../../CONTAMINATION.md)与[来源声明](../../PROVENANCE.md)。

## 版本

| 版本 | Checkpoint | 默认 effort |
|---|---|---|
| **v1.1 — 当前版本** | `022D0-f7` | `high` |
| [v1.0 — 历史版本](https://huggingface.co/Harry19081/Wald-4B/tree/v1.0) | `021A0-f10` | `medium` |

v1.0 模型卡保留其 XL、任务 LoRA 和延迟报告，[v1.0 讲解幻灯片](https://claude.ai/artifact/XfCHVaCuj9A5ectrpaWzV5)只描述 v1.0。这些测量属于各自注明的模型版本。

## 引用

Wald-Q4B v1.1 (2026), an open-weight 4B decision model with calibrated option probabilities. https://huggingface.co/Harry19081/Wald-4B

```bibtex
@misc{wald_q4b_2026,
  title        = {Wald-Q4B v1.1: an open-weight 4B decision model with calibrated option probabilities},
  author       = {{Wald-4B authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Harry19081/Wald-4B}},
  note         = {Revision v1.1}
}
```

机器可读：[CITATION.cff](../../CITATION.cff) · [llms.txt](../../llms.txt) · [model-info.json](../../model-info.json)

---

[模型](https://huggingface.co/Harry19081/Wald-4B) · [GitHub](https://github.com/Harry19081/wald-4b) · [Decision Index 结果](https://huggingface.co/datasets/Harry19081/Wald-Q4B-decision-index-results) · 权重与代码：Apache-2.0。[第三方声明](../../NOTICE) · [训练数据使用说明](../../PROVENANCE.md)
