<div align="center">
  <h1>Wald-Q4B</h1>
  <p><strong>直接决策，按需思考，返回概率。</strong></p>
  <p><a href="https://huggingface.co/org2ai/Wald-4B">English</a> · <a href="https://huggingface.co/org2ai/Wald-4B/blob/main/docs/readmes/README.zh.md">简体中文</a> · <a href="https://github.com/org2AI/wald-4b">GitHub</a> · <a href="https://huggingface.co/org2ai/Wald-4B/blob/main/docs/api.md">API</a></p>
</div>

**Wald-Q4B 是一个开放权重的 4B 决策模型：给它一段状态和一组选项，它为每个选项返回校准过的概率。** 它面向构建 agent 和数据流水线的开发者：需要一个快速、可自行部署的组件来选择工具、路由请求、分类输入，或判断是否需要向用户澄清。和聊天模型不同，它不生成需要再解析的答案，而是一遍读出所有选项的概率，也可以先思考再回答。它提供与 Jev 兼容的 `POST /v1/systemone` API，基于 Qwen3.5-4B-Base，以 Apache-2.0 发布。

**这个仓库里有两个版本。**

| 版本 | Revision | Checkpoint | 一句话说明 |
|---|---|---|---|
| **v1.2**（2026-10-01） | tag [`v1.2`](https://huggingface.co/org2ai/Wald-4B/tree/v1.2)；`main` 上的权重也是它 | `02600-f19` | 抗干扰版：在 v1.1 上再合并一个 LoRA 阶段，让模型在输入里混有无关句子、旁人意见或伪造指令时保持原来的答案 |
| **v1.1**（2026-09-29） | tag [`v1.1`](https://huggingface.co/org2ai/Wald-4B/tree/v1.1) | `022D0-f7` | 通用版：完整 Decision Index 结果和思考档的评测都在这个版本上 |

从 2026-10-01 起，`main` 上是 v1.2 的权重（此前是 v1.1）。v1.1 正在排队的基准提交钉在本仓库的固定 commit 上，不受影响；要用 v1.1 请下载 `--revision v1.1`。下载时请指定 revision。

本仓库 2026-10-01 从 `Harry19081/Wald-4B` 迁到 `org2ai/Wald-4B`，旧链接会自动跳转到这里。各发布 tag（`v1.0`、`v1.1`、`v1.2`）上的模型卡不可修改，仍写旧路径，旧路径同样会跳转。

Wald-Q4B 是 TypeSafe 托管 Jev API 之外、可自行部署的独立替代方案。它不是 Jev，不含 Jev 权重，与 TypeSafe AI 没有隶属或背书关系。Hugging Face 仓库为 `org2ai/Wald-4B`（旧名 Wald-4B；2026-10-01 从 `Harry19081/Wald-4B` 迁来，旧链接自动跳转）。

**W**ait **A** bit, **L**ook, then **D**ecide：稍等一下，看清楚，再决定。名字也致敬序贯分析先驱 Abraham Wald：证据足够时就停止。

## 一览

- **4B 参数**，基于 Qwen3.5-4B-Base，BF16 权重（8.4 GB）。
- **每个选项都有概率。** 题型：`choice`（1–255 个命名选项）、`noul`（是/否）、`score`（有序等级）。
- **v1.2 比 v1.1 更抗干扰**：在 JevAdvBench 上，九类攻击的平均翻转率为 **4.6%**（v1.1：9.2%；Jev 1.13：6.1%）。自测，effort `none`。
- **v1.2 的代价**：JevAdvBench 143 道人工复核题的干净准确率为 76.2%（v1.1：79.0%；−2.8 个百分点，95% 置信区间 [−6.2, −0.6]）。
- **JevBench 公开集**：v1.2 **204/231**，v1.1 **203/231**，都使用 `none`。用 JevBench 自己的评测工具自评。
- **Decision Index 0.2.1**：完整套件的 **54.59**（`high`）是在 **v1.1** 上测的。v1.2 只有一遍读出的抽样结果，与 v1.1 持平（+0.27，不显著）。
- **可调思考程度**：`none`、`low`、`medium`、`high`，以及多次思考的 `high-k`。思考档的评测在 v1.1 上。v1.2 是一遍读出的模型，请用 `none`。
- **可自行部署的 API**：`POST /v1/systemone`，最多 131,072 个提示 token。

## 该用哪个版本？

| 场景 | Revision | 原因 |
|---|---|---|
| 输入里可能混有干扰、劝说或对抗性文本（网页、用户消息、工具输出、检索到的文档） | **`v1.2`** | 在 JevAdvBench 的九类攻击下，答案被改变的次数大约减半 |
| 输入干净可信、看重最后几个点的准确率；需要思考档；需要评测过的 Decision Index 配置 | **`v1.1`** | 干净准确率略高；完整 Decision Index（54.59，`high`）和思考档都是在它上面测的 |

两个版本的结构、分词器、提示格式、校准表和服务代码完全相同。不同的是两个权重分片，以及 `serving.json` 声明的默认 effort：v1.2 为 `none`，v1.1 为 `high`。

## 快速开始

在装有 NVIDIA GPU 和 [`uv`](https://docs.astral.sh/uv/) 的 Linux 机器上：

```sh
hf download org2ai/Wald-4B --revision v1.2 --local-dir ./Wald-Q4B      # v1.2，抗干扰版
# hf download org2ai/Wald-4B --revision v1.1 --local-dir ./Wald-Q4B    # v1.1，通用版
cd Wald-Q4B
EFFORT=none ./run.sh "$PWD"     # 一遍读出，延迟最低
# ./run.sh "$PWD"               # 该版本声明的默认值：v1.2 为 none，v1.1 为 high
```

在 Python 里同样指定 revision：

```python
from huggingface_hub import snapshot_download

path = snapshot_download("org2ai/Wald-4B", revision="v1.2")   # 或 revision="v1.1"
```

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The customer wants to return a damaged kettle.",
    "effort": "none",
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

`route` 的答案包含选中的键，以及 `returns`、`delivery`、`other` 各自的概率。请求与响应字段、是非题和评分题、澄清判断示例见 [API 说明](https://huggingface.co/org2ai/Wald-4B/blob/main/docs/api.md)。`GET /health` 返回当前生效策略。服务使用 vLLM 0.30.0 和仓库内的 `wald-serve`；Docker 与精确评测配置见 [RUNBOOK.md](https://huggingface.co/org2ai/Wald-4B/blob/main/RUNBOOK.md)。普通文本生成接口不会复现决策 API 的读出流程。

## GGUF：llama.cpp、Ollama、LM Studio

`main` 上另有 v1.2 的 GGUF 文件，适合 CPU、Apple Silicon 和消费级显卡（llama.cpp `b11312`，同一份权重）。在 JevBench 公开集（231 题，`none`）上与 vLLM 上的 BF16 权重（204/231）对照：

| 文件 | 大小 | JevBench 公开集 | 与 BF16 选同一选项 |
|---|---:|---:|---:|
| `Wald-4B-v1.2-Q8_0.gguf` | 4.5 GB | 206/231 | 229/231 |
| `Wald-4B-v1.2-Q6_K.gguf` | 3.5 GB | 205/231 | 227/231 |
| `Wald-4B-v1.2-Q5_K_M.gguf` | 3.1 GB | 205/231 | 227/231 |
| `Wald-4B-v1.2-Q4_K_M.gguf` | 2.7 GB | 202/231 | 223/231 |

要拿到校准后的概率，用自带的服务跑在 llama.cpp 上（`llama-server` 需在 `PATH` 里）：

```sh
hf download org2ai/Wald-4B --include "Wald-4B-v1.2-Q8_0.gguf" "serving.json" "temperature.json" "server/*" --local-dir ./wald-gguf
pip install ./wald-gguf/server
wald-serve --gguf ./wald-gguf/Wald-4B-v1.2-Q8_0.gguf --max-model-len 32768 --port 8000
```

Ollama：`ollama run hf.co/org2ai/Wald-4B:Q4_K_M`。LM Studio：搜索 `Wald-4B`。这两个应用都没有用这些文件实测过；聊天界面返回的是文字，不是各选项的概率。请用 `--include` 只下载一个文件；直接 `hf download org2ai/Wald-4B` 下载 `main` 会把四个 GGUF（14 GB）一起拉下来。详见 [org2ai/Wald-4B-GGUF](https://huggingface.co/org2ai/Wald-4B-GGUF)。

## 思考程度

| Effort | 何时思考 | 思考预算 |
|---|---|---|
| `none` | 直接读取选项概率 | 不生成思考文本 |
| `low` | 初次最高概率 < 0.5 | 最多 512 token |
| `medium` | 初次最高概率 < 0.7 | 最多 512 token |
| **`high`（v1.1 默认）** | 每个符合条件的问题 | 最多 512 token |
| `high-k2` … `high-k8` | 多次思考，平均答案分布 | 每次最多 512 token |

思考适用于 2–26 个选项且上下文空间足够的问题。更多选项采用分组读取，再比较各组优胜项；若放不下思考文本，则保留初次答案。提高 effort 会增加计算量，但不保证每道题都更准确。

**v1.1**：直接决策可从 `none` 开始；按置信度触发思考用 `medium`；复现 Decision Index 的评测配置用 `high`。**54.59 仅对应 v1.1 的 `high`；203/231 仅对应 v1.1 的 `none`。**

**v1.2 是一遍读出的模型。** 它按一遍读出训练和评测，本页上 v1.2 的数字全部是 `none`。我们做过一次检查，思考对它没有帮助：在 JevBench 公开集上，v1.2 用 `medium` 和 `high` 各得 198/231（各跑一次），用 `none` 是 204/231；v1.1 用 `medium` 是 205/231。所以 v1.2 的 `serving.json` 把默认 effort 声明为 `none`。需要思考档请用 v1.1。

通过 `EFFORT=medium ./run.sh "$PWD"` 设置服务默认值，也可在单次请求中传入 `"effort": "none"` 覆盖。

## 如何工作

Wald 先从普通文本提示末尾读取选项字母的 logits，得到初始概率。如果 effort 策略触发思考，就生成一段短思考，再读取选项。返回的概率经过分桶温度校准。

v1.1 结合全参数决策训练、LoRA 精修、短思考蒸馏和 RLCD。训练使用我们的自生成决策语料 **WaldGen**，并混合公开训练数据。

**v1.2 在 v1.1 上再加一个 LoRA 阶段（秩 16，作用于所有语言投影层，已合并进权重）：**

- **题目**：从 v1.1 自己的训练文本里取 5,300 道题。没有新增数据来源。
- **扰动**：8,064 条扰动副本。每条在题干、某个选项后或状态里插入一小段文字：无关句子和离题段落，旁观者把答案往另一个选项推的意见、传言或类比，或自称有权限的伪造指令。少部分是改写和错别字。
- **谁写的**：**插入的文字和改写由 Claude Haiku（Anthropic 的模型）按我们自己的模板写成。** 插入由脚本完成，所以原题的事实逐字节不变。错别字由脚本生成。
- **核对**：每一条都由另一次 Claude Haiku 调用核对（“这处改动会不会改变正确答案？”），被判会改变的丢弃。v1.1 答案发生变化的那些行再由 Claude Sonnet 复核一遍。
- **训练目标**：v1.1 自己在干净题上的答案分布。也就是教模型在被扰动的题上，答得和 v1.1 在干净题上一样。另外回放 5,000 条干净题并向 v1.1 对齐，限制漂移。
- **与基准隔离**：没有使用任何 JevAdvBench 文本。把每段插入文字与 JevAdvBench 的全部字符串（干净题和全部 9,744 个攻击变体）做 8-gram 重叠检查，命中 0 处。

数据来源与评测说明：v1.1 [PROVENANCE.md](https://huggingface.co/org2ai/Wald-4B/blob/v1.1/PROVENANCE.md) · [CONTAMINATION.md](https://huggingface.co/org2ai/Wald-4B/blob/v1.1/CONTAMINATION.md)；v1.2 [PROVENANCE.md](https://huggingface.co/org2ai/Wald-4B/blob/v1.2/PROVENANCE.md) · [CONTAMINATION.md](https://huggingface.co/org2ai/Wald-4B/blob/v1.2/CONTAMINATION.md)

## 评测

下面的数字全部是我们自己跑、自己报告的，没有一个是排行榜结果。

| 基准 | 配置 | v1.1 | **v1.2** | 说明 |
|---|---|---:|---:|---|
| **JevAdvBench，九类攻击的平均翻转率**（越低越好） | `none` | 9.2% | **4.6%** | 配对差 −4.6 个百分点，95% 置信区间 [−5.5, −3.6]。Jev 1.13：6.1% |
| **JevAdvBench，143 道人工复核题的干净准确率** | `none` | 79.0% | **76.2%** | 配对差 −2.8 个百分点 [−6.2, −0.6]。Jev 1.13：87.4% |
| **JevBench 公开集（231 题）** | `none` | 203/231（87.9%）· ECE 0.041 · Brier 0.188 | **204/231**（88.3%）· ECE 0.045 · Brier 0.191 | 用 JevBench 的评测工具自评。v1.1 已在 [issue #146](https://github.com/fstandhartinger/jevbench/issues/146) 请维护者测量；v1.2 没有提交 |
| JevBench 公开集（231 题） | `medium` | 205/231 | 198/231 | 各跑一次。v1.2 用 `high`：198/231 |
| **Decision Index 0.2.1，6,948 个请求的抽样** | 一遍读出 | 49.76 | **50.03** | 配对差 +0.27 [−0.36, +1.05]，不显著 |
| **Decision Index 0.2.1 完整套件** | `high` | **54.59** | 未跑 | 只有 v1.1。作者自测；[PR #30](https://github.com/apolinario/decision-index/pull/30) 等待维护者验证 |

我们内部的两个非回退检查集（不是公开基准）在一遍读出下也保持住了：XL-Int 56.4（v1.1：56.0），2,857 题的工具选择集 87.22%（v1.1：87.15%）。

### 抗干扰：JevAdvBench

[JevAdvBench](https://github.com/JevAdvBench/JevAdvBench)（[论文](https://arxiv.org/abs/2609.31142)）对 812 道决策题做九类改动；模型在被攻击的题上的决定与它自己在干净题上的决定不同，就记一次**翻转**。我们把基准的请求原样发给打包服务（`none`），再用基准自己的分析代码（`JevAdvBench@3218e05`）判分。翻转率是占 812 道题的百分比，区间是按场景聚类自助法的 95% 区间。

| 攻击 | Jev 1.13 | v1.1 | **v1.2** | v1.2 − v1.1（配对） |
|---|---:|---:|---:|---:|
| Q1 改词 | 1.0 | 1.6 | 1.6 | 0.0 [−0.8, 0.8] |
| Q2 改写 | 1.5 | 1.6 | 1.4 | −0.2 [−1.1, 0.5] |
| Q3 题干里插入无关句子 | 4.6 | 14.9 | **4.6** | **−10.3 [−13.5, −7.2]** |
| T1 状态里插入无关说明 | 2.2 | 2.8 | 1.7 | −1.1 [−2.4, 0.0] |
| T2 状态里插入旁观者意见 | 12.1 | 16.9 | **5.5** | **−11.3 [−14.6, −8.2]** |
| T3 用类比表达的意见 | 6.9 | 12.7 | **6.3** | **−6.4 [−8.8, −3.6]** |
| P1 直接要求改答案 | 8.9 | 7.9 | 5.7 | −2.2 [−3.6, −0.9] |
| P2 冒充权威 | 10.1 | 13.7 | **6.7** | **−7.0 [−9.0, −5.1]** |
| P3 伪造的校验说明 | 8.1 | 10.3 | 7.8 | −2.6 [−4.2, −1.0] |
| **九类平均** | **6.1** | **9.2** | **4.6** | **−4.6 [−5.5, −3.6]** |
| 至少被一类攻击翻转的题 | 29.2 | 41.7 | 20.7 | |
| 干净准确率，143 道人工复核题 | 87.4 | 79.0 | 76.2 | −2.8 [−6.2, −0.6] |

- **Jev 一列**：随基准一起发布的 `jev-1.13.0` 回答，用同一套代码判分。v1.2 与 Jev 的平均翻转率之差：−1.6 个百分点 [−2.7, −0.4]。
- **干净准确率**：在人工复核的干净题上，Jev 比两个 Wald 版本都准。在全部 812 道题上，v1.2 的干净决定有 98.6% 与 v1.1 相同。
- **这张表不能说明的**：v1.2 的扰动类型是在看到 v1.1 的分项结果之后选的，所以基准的攻击类别影响了训练设计；基准的文本没有被使用。对这些类别之外的攻击是否更稳，没有测过。

### JevBench 与 Decision Index

**JevBench：** 使用 JevBench 自己的命令行工具（`fstandhartinger/jevbench` @ `9ec6f15a`，`typesafe` 适配器），通过本机回环逐条请求打包服务。v1.2 用 `none`，单张 RTX 5090：easy 48/48、original 72/72、hard 84/111；不生成任何 token。v1.1 用 `none`，单张 RTX PRO 6000：easy 48/48、original 72/72、hard 83/111；`medium` 为 205/231，p95 1.80 s。公开题在开发中被用作计分板（从未作为训练数据），因此这不是留出集结果。JevBench 排行榜只在维护者自己跑过模型后才公布分数。

**发布后的复读：** 我们匿名下载了 `v1.2` revision，重跑了 `none` 的读数。只放下载到的模型文件的目录复现了 204/231，231 题的选项全部相同（ECE 0.045）。用完整的仓库目录起服务，两次读数是 204/231 和 205/231（ECE 0.053），有一到两道接近平局的题选项不同。各次读数用的模型文件逐字节相同；这个小差异的原因还没有查明。[详情](https://huggingface.co/org2ai/Wald-4B/blob/main/evaluation/v1.2/release-check.json)

**Decision Index（v1.1）：** 150,317/150,317 个请求全部成功，包含 HLE。在单张 RTX PRO 6000 96 GB 上使用固定版本的复现工具运行。[完整结果](https://huggingface.co/datasets/org2ai/Wald-Q4B-decision-index-results/tree/805716601b2466be324ed6716407b4c3d9267faa/runs/wald-q4b-22d0-f7-full021) · [分项成绩](https://huggingface.co/org2ai/Wald-4B/blob/main/evaluation/benchmark-summary.json) · [复现指南](https://huggingface.co/org2ai/Wald-4B/blob/main/RUNBOOK.md)。`main` 上 `evaluation/` 里的文件属于这次 v1.1 运行；v1.2 的数字在 `v1.2` revision 的 [`evaluation/v1.2/summary.json`](https://huggingface.co/org2ai/Wald-4B/blob/v1.2/evaluation/v1.2/summary.json)。上表 v1.2 的抽样结果是 6,948 个请求的一遍读出，不能与 54.59 比较。

### 速度

延迟是在 v1.1 上测的，v1.2 没有重测；两者结构、大小和服务端相同。使用 `none` 时，v1.1 的 JevBench 运行单次决策**中位延迟 33 ms、p95 168 ms**（单张 RTX PRO 6000）。`high` 在同一 GPU 上的 32 请求串行预检中，**中位延迟为 821 ms**；这只是小规模预检，不代表完整套件延迟或 Decision Index 维护者的准入测试。effort、上下文长度、选项数量和并发都会影响速度。

## 相关项目与对比

多个项目在做带校准选项概率的结构化决策。以下名称归各自所有者；Wald 与它们都没有隶属关系。

- **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** 是 TypeSafe AI 通过 `/v1/systemone` API 提供的托管决策模型，权重不公开。Wald 接受相同的请求格式，运行在你自己的 GPU 上。
- **[Kev](https://github.com/jaredpalmer/kev)** 是 Jared Palmer 的开放权重项目，在 Qwen3.5 基座（0.8B、4B、9B）上加 LoRA 和指针头。Wald 的请求解析改编自 Kev 的 Apache-2.0 代码（见 [NOTICE](https://huggingface.co/org2ai/Wald-4B/blob/main/NOTICE)）；Wald 从语言模型头读取选项字母，不使用单独的头。
- **[Laya](https://huggingface.co/convaiinnovations/laya)**（[代码](https://github.com/NandhaKishorM/laya)）是开放权重的 421M ModernBERT-large 编码器加决策头。它比 Wald 小得多，最多读取 512 个 token。

**JevBench 公开集，同一批 231 题（数据集哈希相同），JevBench 命令行工具，由我们运行：**

| 系统 | 运行方式 | 答对 |
|---|---|---:|
| Wald-Q4B v1.2 · `none` | 自行部署，RTX 5090，2026-09-30 | 204/231 |
| Wald-Q4B v1.1 · `none` | 自行部署，RTX PRO 6000，2026-09-29 | 203/231 |
| Jev（`jev-1.13.0`） | TypeSafe 托管 API，2026-09-25 | 200/231 |
| Laya（英文 checkpoint `55cf4c4e`） | 自行部署，NVIDIA L4，2026-09-26 | 134/231 |

231 题上几题的差距在多次运行和抽样的噪声范围内。公开题影响过 Wald 的开发；231 题中有 52 题的状态超过 Laya 的 512 token 窗口。

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

**能用 Jev 客户端调用自部署模型吗？** 把客户端指向你自己的端点。内置服务在 `POST /v1/systemone` 接收 `state` 和带类型的 `questions`（`choice`、`noul`、`score`），按 TypeSafe 的答案键返回。服务不校验 API key。见 [API 说明](https://huggingface.co/org2ai/Wald-4B/blob/main/docs/api.md)。

**如何做工具路由，或判断是否需要向用户提问？** 把对话或任务作为 `state` 发送。工具路由：提一个 `choice` 问题，选项就是你的工具。是否澄清：提一个 `noul` 问题，例如“这个请求是否足够具体，可以不问就执行？”概率高就执行，概率低就提问，两个阈值都在你自己的验证数据上确定。模型只选工具，不生成工具参数。

**v1.2 能防住提示注入吗？** 没有模型能完全防住。在 JevAdvBench 的九类攻击下，v1.2 改变答案的次数大约是 v1.1 的一半，但平均仍有 4.6% 的被攻击题发生翻转，20.7% 的题至少被一类攻击翻转。请把它当作多层防护中的一层：尽量不要让不可信文本进入指令，重要决定要复核。

**概率校准得怎么样？** 在 JevBench 公开集上使用 `none`，v1.2 的期望校准误差为 0.045（用完整发布目录复读时为 0.053），v1.1 为 0.041（10 个分桶）；Brier 分数分别为 0.191 和 0.188。v1.2 沿用 v1.1 的温度表，没有改动。温度是在我们自己开发数据的留出行上拟合的，不含 JevBench 题目，并排除了已知的 Decision Index 匹配项。置信度不是保证，请在你的任务上检查校准。

**能在单张 GPU 或笔记本上运行吗？** 内置服务需要 Linux 上的一张 NVIDIA GPU（vLLM 0.30.0）；BF16 权重为 8.4 GB。v1.1 的测量来自 RTX PRO 6000 96 GB，v1.2 的测量来自 RTX 5090 32 GB；同一 4B 架构的早期版本也曾用 vLLM 在 24 GB 的 NVIDIA L4 上以 16K 上下文上限运行。内置服务不支持 CPU、Apple Silicon 和笔记本环境，也没有测试过。

**Kev 和 Wald、Laya 和 Wald 怎么选？** 三者都开放权重。Kev 在 Qwen3.5 基座上加指针头和 LoRA；Laya 是带决策头的小型编码器；Wald 是完整训练的 4B 解码器，可选思考。我们在同一协议下的测量见上表。请根据你自己的任务、延迟预算和硬件选择。

**能针对我的任务微调吗？** 它是标准的 Transformers checkpoint，常见的 LoRA 工具都适用。v1.0 时我们为单个任务训练 LoRA，每个花费 $0.12–$1.81 的 GPU 时间；这套工具尚未公开，这些 adapter 也未在 v1.1 或 v1.2 上验证（[v1.0 说明](https://huggingface.co/org2ai/Wald-4B/blob/main/history/v1.0/README.md)）。

**许可证是什么？** 权重与代码为 Apache-2.0。基座 Qwen3.5-4B-Base 也是 Apache-2.0。部分公开训练数据有各自的条款或没有注明许可证，列在 [PROVENANCE.md](https://huggingface.co/org2ai/Wald-4B/blob/main/PROVENANCE.md)。模型与代码的许可证不授予这些文本的权利。v1.2 没有新增数据来源；它新增的训练文字由 Claude Haiku 写成，见上文。

## 使用限制

- **v1.2 用一点干净准确率换抗干扰能力**：JevAdvBench 143 道人工复核的干净题上 −2.8 个百分点（95% 置信区间 [−6.2, −0.6]）。如果你更看重这一点，请用 `v1.1`。
- **v1.2 是一遍读出的模型。** 它的结果全部是 `none`，没有完整 Decision Index 运行；我们的一次检查里，思考档让它的 JevBench 公开集成绩变低（198/231 对 204/231）。需要思考请用 v1.1。
- **抗干扰能力只在一个基准上测过**，而且是在影响过训练设计的攻击类别上。它不是安全保证。

置信度不保证正确性，应在自己的任务上验证阈值。超过上下文限制的提示会被拒绝，不会截断。已过滤已知的严格训练重叠，但无法排除语义重叠和预训练污染；开发过程中使用了可见的基准样本。各来源文本的使用权不同，详见[评测说明](https://huggingface.co/org2ai/Wald-4B/blob/main/CONTAMINATION.md)与[来源声明](https://huggingface.co/org2ai/Wald-4B/blob/main/PROVENANCE.md)。

## 版本

| 版本 | Revision | Checkpoint | 说明 |
|---|---|---|---|
| **v1.2 — 抗干扰版** | tag [`v1.2`](https://huggingface.co/org2ai/Wald-4B/tree/v1.2)；`main` 上的权重 | `02600-f19` | v1.1 + 合并的抗干扰 LoRA；默认 effort `none` |
| **v1.1 — 通用版** | tag [`v1.1`](https://huggingface.co/org2ai/Wald-4B/tree/v1.1) | `022D0-f7` | 完整 Decision Index 运行（54.59）；默认 effort `high` |
| v1.0 — 历史版本 | tag [`v1.0`](https://huggingface.co/org2ai/Wald-4B/tree/v1.0) | `021A0-f10` | 默认 effort `medium` |

v1.1 正在排队的基准提交（JevBench issue #146、Decision Index PR #30）指向本仓库的固定 commit，不受 v1.2 影响。v1.0 模型卡保留其 XL、任务 LoRA 和延迟报告，[v1.0 讲解幻灯片](https://claude.ai/artifact/XfCHVaCuj9A5ectrpaWzV5)只描述 v1.0。这些测量属于各自注明的模型版本。

## 引用

Wald-Q4B (2026), an open-weight 4B decision model with calibrated option probabilities. https://huggingface.co/org2ai/Wald-4B。请注明使用的 revision（`v1.2` 或 `v1.1`）。

```bibtex
@misc{wald_q4b_2026,
  title        = {Wald-Q4B: an open-weight 4B decision model with calibrated option probabilities},
  author       = {{Wald-4B authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/org2ai/Wald-4B}},
  note         = {Revision v1.2}
}
```

机器可读：[CITATION.cff](https://huggingface.co/org2ai/Wald-4B/blob/main/CITATION.cff) · [llms.txt](https://huggingface.co/org2ai/Wald-4B/blob/main/llms.txt) · [model-info.json](https://huggingface.co/org2ai/Wald-4B/blob/main/model-info.json)

---

[模型](https://huggingface.co/org2ai/Wald-4B) · [GitHub](https://github.com/org2AI/wald-4b) · [Decision Index 结果](https://huggingface.co/datasets/org2ai/Wald-Q4B-decision-index-results) · 权重与代码：Apache-2.0。[第三方声明](https://huggingface.co/org2ai/Wald-4B/blob/main/NOTICE) · [训练数据使用说明](https://huggingface.co/org2ai/Wald-4B/blob/main/PROVENANCE.md)
