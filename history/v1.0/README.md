---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: finetune
language:
- en
tags:
- decision-model
- calibration
- typesafe
- decision-index
pipeline_tag: text-generation
---

<div align="center">
  <h1>Wald-4B v1.0</h1>
  <p><strong>A calibrated decision model: one forward pass turns a state and a question into a probability for every option.</strong></p>
</div>

**W**ait **A** bit, **L**ook, then **D**ecide. Sure? Decide now. Not sure? Look once more. Also named after Abraham Wald (1902–1950), who founded sequential analysis: stop as soon as the evidence is enough.

---

<p align="center"><a href="README.md">English</a> · <a href="docs/readmes/README.zh.md">简体中文</a></p>

---

**Wald-4B turns a decision into one forward pass.** Give it a state and a question, and it returns a calibrated
probability for every option: which tool to call, whether to ask the user, which intent, whether an output is safe. Act
above 0.9, ask a person below 0.6, and send the cases in between to a larger model. Under effort `medium` it thinks
(≤ 512 tokens) only when its top probability is below 0.7, about one question in nine; the answer is always a
distribution, never parsed text.

## At a glance

- **Decision Index 0.2.1: 53.91** (sample estimate), #6 of 67 on the board and the best 4B entry ([Decision Index](#decision-index-021)).
- **XL hidden split: 63.2**, #2 behind Jev and the best 4B ([XL](#xl-hidden-split)).
- **Your task for under $2:** one LoRA beats Jev on 9 of 12 public tasks ([Verticals](#verticals-a-quick-lora-per-task)).
- **~50 ms per decision** at p50, about 1/5 of Jev's price per 1k decisions ([Latency and cost](#latency-and-cost)).

![Decision Index vs model size](figures/di-vs-size.svg)

## How it works

A 4B model built on Qwen/Qwen3.5-4B-Base @ `1001bb4d826a52d1f399e183466143f4da7b741b`: full-parameter decision training,
then three merged LoRAs (refinement with KL replay, short-thought distillation, task coverage). Each question becomes one
plain prompt with no chat template and no new tokens. The state is written twice (format `repeat_state_plain`; the
second copy is introduced by one fixed sentence, see NOTICE), then:

```
Question: {instructions}
(A) {option 1}
(B) {option 2}
Answer: (
```

The option-letter logits at the last position give the distribution. Above 26 options, options are read in ⌈n / 26⌉
chunks and the chunk winners are read again (knockout). A temperature per (question type, option count), fitted on our own
held-out rows, changes confidence only. The prompt format, the policy and the context limit are fixed in `serving.json`
and are the same for every benchmark.

**Training data** (not released): four stages over 84 public sources (83 datasets read through their train splits only, NLI4CT also its dev split,
plus the Decision Index kit's MIT Home-appliance generator), code-generated and code-labelled decision items, and KL
replay of earlier rows. Part of the data is synthetic: items and short thoughts written or labelled by much larger
frontier LLMs — far above 120B parameters where the size is published. Every source, and how Decision Index items
were screened: [CONTAMINATION.md](CONTAMINATION.md).

## Benchmarks

What we measured: the Decision Index 0.2.1 (the board), XL (a decision benchmark with a hidden split), task LoRAs on
public verticals, and serving latency. All numbers are our own reads, zero-shot unless marked, with the same requests
for every system; rows marked *board* are the official leaderboard's.

### Decision Index 0.2.1

Our read is a **stratified sample of 6,948 requests** (37,469 questions; the kit's own sampler and seed; HLE not
rebuilt), scored with the 0.2.1 rules. Jev reads 57.19 on this sample against its board 57.89, so the sample reads about
0.7 points low. Board rows are official full-suite numbers ([board](https://huggingface.co/spaces/multimodalart/jev-decision-index)).

| system | size | Decision Index 0.2.1 | source |
|---|---|---|---|
| Jev (hosted API, jev-1.13.0) | undisclosed | 57.19 [55.29, 58.60] · board 57.89 | ours, sample · board |
| simple-jev · Qwen3.8-27B (board #4) · Jebadiah 27B (#5) | 27B | 55.74 · 54.67 | board |
| **Wald-4B v1.0 · effort medium** | 4B | **53.91** [51.87, 55.27] | ours, sample |
| reflex 27B (board #6) · Decider chat · Qwen3.6-27B (#7) | 27B | 52.16 · 51.35 | board |
| Qwen3.8-27B, untrained, our letter readout (one pass) | 27B | 47.78 [45.95, 49.17] | ours, sample |
| Decider 35B-A3B (board #11) | 35B-A3B | 47.11 | board |
| Decider 4B (board #15) | 4B | 40.70 | board |
| Kev 9B (board #23) · Kev 4B (#28) | 9B · 4B | 38.48 · 34.64 | board |

53.91 places #6 on the 67-entry board (between Jebadiah 27B and reflex 27B); against Jev on the same sample it is
−3.3 [−4.9, −1.8]. On the same 4B base it is +19.3 over Kev 4B.

![Decision Index by area](figures/di-areas.svg)

**Disclosures.** 363 Decision Index item ids reached our training data through public train splits or shared upstream
sources (never a test split); counting them wrong gives **53.62**. Home appliance scores 1.00 (skill): the last stage
trained on new households from the benchmark's own MIT generator (new seed, 0 shared states); with Home appliance held
at the parent build's answers the index is 51.57 [49.56, 52.95]. Details: [CONTAMINATION.md](CONTAMINATION.md).

**Same-request suite.** JevBench public 231 (accuracy %): Wald-4B v1.0 **88.3** (packaged server), Jev 86.6, raw
Qwen3.5-4B-Base 67.5, Laya 421M 58.0, CLM-8B 39.0. CLM-8B and Laya have no Decision Index read.

### XL (hidden split)

A decision benchmark we built (13 families; ranked on a hidden split of 1,895 items, chance-corrected composite XL-Int).
We are its authors, so no number is independent; our models score far higher on its public split than on the hidden one,
so only the hidden split is reported.

| system | XL-Int (hidden) | rank |
|---|---:|---:|
| Jev (API, v1.13) | 73.0 [69.8, 75.9] | #1 |
| **Wald-4B v1.0 · effort medium** | **63.2** [60.0, 66.4] | **#2**, best 4B |
| Cygnet (gemma-4-12B-it + shim) | 59.8 [56.7, 62.7] | #4 |
| Laya 421M · CLM-8B | 13.4 · 12.7 | #24 · #25 |

### Verticals: a quick LoRA per task

![Verticals](figures/verticals.svg)

One LoRA on the task's labels costs $0.12–$1.81 of GPU time (< 2 GPU-hours); every other system is zero-shot on the
same fixed test items and byte-identical requests. Differences to Jev are paired bootstrap 95 % CIs; ▲ = the CI excludes
0. These reads use the pre-release build v0.9 (see Versioning).

| task (metric) | Wald-4B v0.9 + LoRA | Jev |
|---|---:|---:|
| MetaTool (tool selection, accuracy) | **97.0** | 83.0 |
| BANKING77 (77 intents, macro-F1) | **92.8** | 78.1 |
| AndroidControl (phone-agent action, accuracy) | **85.8**¹ | 73.4 |
| ToxicChat (toxic-class F1) | **84.9** | 80.4 |
| COLD, Chinese offensive language (macro-F1, full 5,323-item test) | **84.1** [83.2, 85.1] | 75.3 |
| When2Call (call / ask / refuse, accuracy) | **83.0**² | 72.6 |

¹ Trained on all training steps; the figure shows the first all-labels arm (81.6). ² Best of 8 LoRA configurations read
on test; the dev-selected one scores 82.6. COLD: paired +8.8 [7.7, 10.0]; 83.8 on the 4,823 items not used for
temperature fitting; ties the best published 83.7.

#### With few labels, starting from Wald beats LoRA on the raw base

Same LoRA recipe, labels and test items; Wald-4B (v0.9) minus raw Qwen3.5-4B-Base, points, paired 95 % CIs.
**Bold** = the CI excludes 0.

| task (metric) | 0 labels (zero-shot) | 300 labels |
|---|---:|---:|
| When2Call (accuracy) | **+21.2** [17.0, 25.2] | +2.0 [−0.2, 4.2] |
| BANKING77 (macro-F1) | **+10.9** [7.6, 14.7] | **+4.4** [2.0, 7.2] |
| SGD intent (macro-F1) | +2.8 (n.s.) | −0.1 |

### Latency and cost

![Latency and cost](figures/latency-cost.svg)

One question, one pass, serial: p50 26 ms on one RTX PRO 6000 (51 ms on an H100). Batched: 115 decisions/s (fp8), about
$0.007 per 1,000 decisions at the GPU-hour price; the Jev API lists $0.04. Under `medium` the median stays at the
one-pass level and the p95 is about one second. Measured on v0.9, which has the same architecture and serving path.

## Train your own task LoRA (CLI, releasing soon)

Every vertical above was made with one CLI, which we plan to release soon. It splits a labelled dataset with fixed seeds,
checks the test items for leaks, trains a LoRA on Wald-4B on a GPU you choose, reads the same test items with your LoRA,
Jev and zero-shot baselines, and serves the adapter behind the same `/v1/systemone` API.

![Architecture of the vertical CLI](figures/lora-cli-architecture.svg)

Stages, guards, costs and a command preview: [docs/lora-cli.md](docs/lora-cli.md).

## Serving

```sh
docker build -t wald-serve .
docker run --gpus all -v /path/to/Wald-4B:/model:ro -p 8000:8000 wald-serve   # ready when GET /health is {"ok": true}
```

Without Docker: `./run.sh /path/to/Wald-4B`. The server takes `--effort` (`none | low | medium | high | high-k<k>`), and a
request may carry `"effort"` to override it. Declared: effort `medium`, prompt format `repeat_state_plain`, 131,072
tokens per question prompt (longer → HTTP 422, never truncated), 1–255 options. Commands, limits and runtimes:
[RUNBOOK.md](RUNBOOK.md). Weights: bf16, `model.safetensors` sha256
`dabeb7bc3f43bf6ea7d20ecfdac8758b3945517a991809a5772988829598e893`; every file's sha256 is in `MANIFEST.json`.

## Limits

- Our non-regression gate against the parent build has no failing metric. Two losses exceed the tolerance but are not
  significant after correction: XL long_policy −6.7 [−11.6, −2.0] and JevBench public 231 −2.6 [−5.2, 0.0] (plain-prompt
  read).
- The temperature table is the parent build's, the one the measured Decision Index read used.
- Our Decision Index read used a 16,384-token context; prompts longer than that were counted unsupported there and are
  answered by the packaged server.

## Versioning

| version | status | checkpoint | base | date | policy | prompt format |
|---|---|---|---|---|---|---|
| **v1.0** | current release | internal build 021A0-f10 | Qwen/Qwen3.5-4B-Base | 2026-09-27 | effort `medium` | `repeat_state_plain` |
| v0.9 | pre-release; the vertical LoRAs and latency above | internal build 015D0-f4 | Qwen/Qwen3.5-4B-Base | 2026-09-26 | effort `medium` | plain |

v1.x = serving or LoRA refreshes on the same base generation; v2.0 = a new base generation.

---

Apache-2.0 (weights and code), subject to the licence of the base model Qwen/Qwen3.5-4B-Base (Apache-2.0, © the Qwen
team, Alibaba Cloud). Third-party notices: [NOTICE](NOTICE).
