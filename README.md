---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- decision-model
- typed-decisions
- calibration
- calibrated-probabilities
- classification
- tool-selection
- tool-use
- agent-routing
- clarification
- robustness
- decision-index
- jevbench
- jev
- jev-compatible
- open-jev
- typesafe-compatible
- systemone
- kev
- laya
- wald
- wald-q4b
- qwen3.5
- 4b
- vllm
- gguf
- llama.cpp
- ollama
- reasoning
model-index:
- name: Wald-Q4B v1.1 (revision v1.1)
  results:
  - task:
      type: text-classification
      name: Structured decisions (option probabilities)
    dataset:
      name: Decision Index 0.2.1, complete suite (150,317 requests)
      type: decision-index
      revision: 87d4650b42b377c0291a89c1f1a879f9b31082bf
    metrics:
    - type: balanced_skill_index
      name: Balanced-skill index, effort high (author-run, not leaderboard-verified)
      value: 54.59
      verified: false
    source:
      name: 'Author-run with the official kit; submission PR #30 awaits maintainer validation'
      url: https://github.com/apolinario/decision-index/pull/30
  - task:
      type: text-classification
      name: Structured decisions (option probabilities)
    dataset:
      name: JevBench public set (231 items)
      type: jevbench
      split: public
      revision: 9ec6f15a
    metrics:
    - type: accuracy
      name: Accuracy, effort none (self-scored, 203/231)
      value: 87.88
      verified: false
    - type: ece
      name: Expected calibration error, 10 bins (self-scored)
      value: 0.041
      verified: false
    - type: brier
      name: Brier score (self-scored)
      value: 0.188
      verified: false
    source:
      name: 'Self-scored with the JevBench harness; row requested in jevbench issue #146'
      url: https://github.com/fstandhartinger/jevbench/issues/146
- name: Wald-Q4B v1.2 (main, revision v1.2)
  results:
  - task:
      type: text-classification
      name: Structured decisions (option probabilities)
    dataset:
      name: JevBench public set (231 items)
      type: jevbench
      split: public
      revision: 9ec6f15a
    metrics:
    - type: accuracy
      name: Accuracy, effort none (self-scored, 204/231)
      value: 88.31
      verified: false
    - type: ece
      name: Expected calibration error, 10 bins (self-scored)
      value: 0.045
      verified: false
    - type: brier
      name: Brier score (self-scored)
      value: 0.191
      verified: false
    source:
      name: Self-scored with the JevBench harness; not submitted
      url: https://huggingface.co/org2ai/Wald-4B/blob/v1.2/evaluation/v1.2/summary.json
  - task:
      type: text-classification
      name: Robustness of structured decisions to distracting and adversarial text
    dataset:
      name: JevAdvBench (812 questions, 9 delivered attack types)
      type: jevadvbench
      revision: "3218e05"
    metrics:
    - type: flip_rate
      name: Mean flip rate in %, effort none, lower is better (self-run)
      value: 4.6
      verified: false
    source:
      name: Self-run with the benchmark's request bytes and analysis code; not submitted
      url: https://huggingface.co/org2ai/Wald-4B/blob/v1.2/evaluation/v1.2/summary.json
---

<div align="center">
  <h1>Wald-Q4B</h1>
  <p><strong>Decide directly. Think when needed. Return probabilities.</strong></p>
  <p><a href="https://huggingface.co/org2ai/Wald-4B">English</a> · <a href="https://huggingface.co/org2ai/Wald-4B/blob/main/docs/readmes/README.zh.md">简体中文</a> · <a href="https://github.com/org2AI/wald-4b">GitHub</a> · <a href="https://huggingface.co/org2ai/Wald-4B/blob/main/docs/api.md">API</a></p>
</div>

**Wald-Q4B is an open-weight 4B decision model: give it a state and a set of options, and it returns a calibrated probability for every option.** It is for developers who build agents and pipelines and need a fast, self-hosted component to pick a tool, route a request, classify an input or decide whether to ask the user. Unlike a chat model, it does not write an answer you have to parse. It reads the options in one pass and can optionally think first. It serves a Jev-compatible `POST /v1/systemone` API, is built on Qwen3.5-4B-Base and is released under Apache-2.0.

**Two releases live in this repository.**

| Release | Revision | Checkpoint | In one line |
|---|---|---|---|
| **v1.2** (2026-10-01) | tag [`v1.2`](https://huggingface.co/org2ai/Wald-4B/tree/v1.2); also the weights on `main` | `02600-f19` | The robustness release: v1.1 plus one merged LoRA stage that teaches the model to keep its answer when the input contains distracting sentences, third-party opinions or fake instructions |
| **v1.1** (2026-09-29) | tag [`v1.1`](https://huggingface.co/org2ai/Wald-4B/tree/v1.1) | `022D0-f7` | The general release, with the complete Decision Index run and the evaluated thinking efforts |

Since 2026-10-01 `main` holds the v1.2 weights (before that it held v1.1). The pending benchmark requests for v1.1 name fixed commits of this repository and are unaffected; for v1.1, download `--revision v1.1`. Pin a revision when you download.

This repository moved from `Harry19081/Wald-4B` to `org2ai/Wald-4B` on 2026-10-01; old links redirect here. The cards stored at the release tags (`v1.0`, `v1.1`, `v1.2`) are immutable and still show the old path, which also redirects.

Wald-Q4B is an independent, self-hosted alternative to TypeSafe's hosted Jev API. It is not Jev, contains no Jev weights, and is not affiliated with or endorsed by TypeSafe AI. The Hugging Face repository is `org2ai/Wald-4B` (earlier name: Wald-4B; moved from `Harry19081/Wald-4B` on 2026-10-01, old links redirect).

**W**ait **A** bit, **L**ook, then **D**ecide. Also named after Abraham Wald, the pioneer of sequential analysis: stop when the evidence is enough.

## At a glance

- **4B parameters**, built on Qwen3.5-4B-Base; BF16 weights (8.4 GB).
- **Every option gets a probability.** Question types: `choice` (1–255 named options), `noul` (yes/no) and `score` (ordered levels).
- **v1.2 is more robust than v1.1:** on JevAdvBench the mean flip rate over nine attack types is **4.6 %** (v1.1: 9.2 %; Jev 1.13: 6.1 %). Self-run, effort `none`.
- **The cost of v1.2:** clean accuracy on JevAdvBench's 143 human-reviewed questions is 76.2 % (v1.1: 79.0 %; −2.8 points, 95 % CI [−6.2, −0.6]).
- **JevBench public set:** v1.2 **204/231**, v1.1 **203/231**, both with `none`. Self-scored with JevBench's own harness.
- **Decision Index 0.2.1:** the complete-suite result, **54.59** with `high`, was measured on **v1.1**. v1.2 has only a one-pass sample read, which is level with v1.1 (+0.27, not significant).
- **Adjustable thinking:** `none`, `low`, `medium`, `high`, or several thoughts with `high-k`. Thinking is evaluated on v1.1. v1.2 is a one-pass model: use it with `none`.
- **Self-hosted API:** `POST /v1/systemone`, up to 131,072 prompt tokens.

## Which version should I use?

| Use | Revision | Why |
|---|---|---|
| Inputs may contain distracting, persuasive or adversarial text (web pages, user messages, tool outputs, retrieved documents) | **`v1.2`** | About half as many changed answers under the nine JevAdvBench attacks |
| Clean, trusted inputs where the last points of accuracy matter; thinking efforts; the evaluated Decision Index configuration | **`v1.1`** | Slightly higher clean accuracy; the complete Decision Index run (54.59, `high`) and the thinking efforts were measured on it |

Both revisions share the same architecture, tokenizer, prompt format, calibration table and server code. The two weight shards differ, and so does the default effort declared in `serving.json`: `none` for v1.2, `high` for v1.1.

## Quick start

On a Linux machine with an NVIDIA GPU and [`uv`](https://docs.astral.sh/uv/):

```sh
hf download org2ai/Wald-4B --revision v1.2 --local-dir ./Wald-Q4B      # v1.2, robustness release
# hf download org2ai/Wald-4B --revision v1.1 --local-dir ./Wald-Q4B    # v1.1, general release
cd Wald-Q4B
EFFORT=none ./run.sh "$PWD"     # one pass, lowest latency
# ./run.sh "$PWD"               # the release's declared default: none for v1.2, high for v1.1
```

From Python, pin the revision the same way:

```python
from huggingface_hub import snapshot_download

path = snapshot_download("org2ai/Wald-4B", revision="v1.2")   # or revision="v1.1"
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

The answer for `route` contains the chosen key and a probability for each of `returns`, `delivery` and `other`. Request and response fields, yes/no and score questions, and a clarification example: [API reference](https://huggingface.co/org2ai/Wald-4B/blob/main/docs/api.md). `GET /health` reports the effective policy. The server uses vLLM 0.30.0 and the included `wald-serve` package; Docker and exact evaluation settings are in [RUNBOOK.md](https://huggingface.co/org2ai/Wald-4B/blob/main/RUNBOOK.md). Generic text-generation calls do not reproduce the decision API's readout.

## GGUF: llama.cpp, Ollama, LM Studio

`main` also carries v1.2 as GGUF files for CPUs, Apple Silicon and consumer GPUs (llama.cpp `b11312`, the same weights). Parity on the JevBench public set (231 items, `none`), against the BF16 weights on vLLM (204/231):

| File | Size | JevBench public | Same option as BF16 |
|---|---:|---:|---:|
| `Wald-4B-v1.2-Q8_0.gguf` | 4.5 GB | 206/231 | 229/231 |
| `Wald-4B-v1.2-Q6_K.gguf` | 3.5 GB | 205/231 | 227/231 |
| `Wald-4B-v1.2-Q5_K_M.gguf` | 3.1 GB | 205/231 | 227/231 |
| `Wald-4B-v1.2-Q4_K_M.gguf` | 2.7 GB | 202/231 | 223/231 |

For calibrated probabilities, run the included server on llama.cpp (`llama-server` on your `PATH`):

```sh
hf download org2ai/Wald-4B --include "Wald-4B-v1.2-Q8_0.gguf" "serving.json" "temperature.json" "server/*" --local-dir ./wald-gguf
pip install ./wald-gguf/server
wald-serve --gguf ./wald-gguf/Wald-4B-v1.2-Q8_0.gguf --max-model-len 32768 --port 8000
```

Ollama: `ollama run hf.co/org2ai/Wald-4B:Q4_K_M`. LM Studio: search for `Wald-4B`. Neither app was tested with these files, and a chat session returns text, not the option probabilities. Download a single file with `--include`; a plain `hf download org2ai/Wald-4B` of `main` also fetches all four GGUFs (14 GB). Details: [org2ai/Wald-4B-GGUF](https://huggingface.co/org2ai/Wald-4B-GGUF).

## Thinking effort

| Effort | When it thinks | Thought budget |
|---|---|---|
| `none` | Direct option readout | No generated thought |
| `low` | Top initial probability < 0.5 | Up to 512 tokens |
| `medium` | Top initial probability < 0.7 | Up to 512 tokens |
| **`high` (v1.1 default)** | Every eligible question | Up to 512 tokens |
| `high-k2` … `high-k8` | Multiple thoughts; average their answer distributions | Up to 512 tokens per thought |

Thinking applies to questions with 2–26 options when context space permits. Larger option sets use grouped readout and a final winner comparison; if a thought cannot fit, the initial answer is kept. Increasing effort spends more computation; it does not guarantee a better answer.

**On v1.1**, start with `none` for direct decisions, `medium` for confidence-gated thinking, or `high` for the evaluated Decision Index configuration. **54.59 applies to v1.1 with `high` only; 203/231 applies to v1.1 with `none` only.**

**v1.2 is a one-pass model.** It was trained and evaluated on the one-pass readout, and every v1.2 number on this card is `none`. In our one check, thinking did not help it: on the JevBench public set, v1.2 scored 198/231 with `medium` and 198/231 with `high` (one run each) against 204/231 with `none`, while v1.1 scored 205/231 with `medium`. v1.2's `serving.json` therefore declares `none` as its default. If you want the thinking efforts, use v1.1.

Set the server default with `EFFORT=medium ./run.sh "$PWD"`, or override it per request with `"effort": "none"`.

## How it works

Wald first reads option-letter logits from a plain prompt and turns them into a probability distribution. If the effort policy asks for thinking, it generates a short thought and reads the options again. Bucketed temperature scaling calibrates the returned probabilities.

v1.1 combines full-parameter decision training, LoRA refinement, short-thought distillation and RLCD. Training uses **WaldGen**, our generated decision corpus, together with public training datasets.

**v1.2 adds one LoRA stage on v1.1 (rank 16 on every language projection, merged into the weights):**

- **Questions:** 5,300 questions drawn from v1.1's own training text. No new source dataset.
- **Perturbations:** 8,064 perturbed copies. Each one has a short text inserted into the question, after an option or into the state: unrelated sentences and off-topic passages, a bystander's opinion, rumour or analogy that pushes another option, or a fake instruction that claims authority. A smaller share are paraphrases and typos.
- **Who wrote them:** **the inserted texts and the paraphrases were written by Claude Haiku (an Anthropic model) from our own templates.** A script inserted them, so the original facts stay byte-identical. Typos were made by a script.
- **Checks:** a separate Claude Haiku call checked every row ("does the edit change the correct answer?") and rejected rows were dropped. Rows on which v1.1 changed its answer were checked a second time by Claude Sonnet.
- **Targets:** v1.1's own answer distribution on the clean question. The model is taught to answer the perturbed question as v1.1 answers the clean one. 5,000 clean rows are replayed against v1.1 to limit drift.
- **Separation from the benchmark:** no JevAdvBench text was used. An 8-gram overlap check of every inserted text against every JevAdvBench string (clean questions and all 9,744 attacked variants) found 0 hits.

Data sources and evaluation notes: v1.1 [PROVENANCE.md](https://huggingface.co/org2ai/Wald-4B/blob/v1.1/PROVENANCE.md) · [CONTAMINATION.md](https://huggingface.co/org2ai/Wald-4B/blob/v1.1/CONTAMINATION.md); v1.2 [PROVENANCE.md](https://huggingface.co/org2ai/Wald-4B/blob/v1.2/PROVENANCE.md) · [CONTAMINATION.md](https://huggingface.co/org2ai/Wald-4B/blob/v1.2/CONTAMINATION.md)

## Benchmarks

All numbers below are self-run and self-reported. None of them is a leaderboard result.

| Benchmark | Configuration | v1.1 | **v1.2** | Note |
|---|---|---:|---:|---|
| **JevAdvBench, mean flip rate over 9 attack types** (lower is better) | `none` | 9.2 % | **4.6 %** | Paired −4.6 points, 95 % CI [−5.5, −3.6]. Jev 1.13: 6.1 % |
| **JevAdvBench, clean accuracy on 143 human-reviewed questions** | `none` | 79.0 % | **76.2 %** | Paired −2.8 points [−6.2, −0.6]. Jev 1.13: 87.4 % |
| **JevBench public set (231 items)** | `none` | 203/231 (87.9 %) · ECE 0.041 · Brier 0.188 | **204/231** (88.3 %) · ECE 0.045 · Brier 0.191 | Self-scored with JevBench's harness. v1.1 row requested in [issue #146](https://github.com/fstandhartinger/jevbench/issues/146); v1.2 not submitted |
| JevBench public set (231 items) | `medium` | 205/231 | 198/231 | One run each. `high` on v1.2: 198/231 |
| **Decision Index 0.2.1, 6,948-request sample** | one pass | 49.76 | **50.03** | Paired +0.27 [−0.36, +1.05], not significant |
| **Decision Index 0.2.1, complete suite** | `high` | **54.59** | not run | v1.1 only. Author-run; [PR #30](https://github.com/apolinario/decision-index/pull/30) awaits maintainer validation |

Two of our internal non-regression sets, which are not public benchmarks, also held in one pass: XL-Int 56.4 (v1.1: 56.0) and a 2,857-item tool-selection set 87.22 % (v1.1: 87.15 %).

### Robustness: JevAdvBench

[JevAdvBench](https://github.com/JevAdvBench/JevAdvBench) ([paper](https://arxiv.org/abs/2609.31142)) attacks 812 decision questions with nine kinds of edits and counts a **flip** when the model's decision on the attacked question differs from its own decision on the clean question. We sent the benchmark's request bytes to the packaged server with `none` and scored the answers with the benchmark's analysis code (`JevAdvBench@3218e05`). Flip rates are in % of the 812 questions; intervals are 95 % scenario-cluster bootstrap intervals.

| Attack | Jev 1.13 | v1.1 | **v1.2** | v1.2 − v1.1 (paired) |
|---|---:|---:|---:|---:|
| Q1 word edits | 1.0 | 1.6 | 1.6 | 0.0 [−0.8, 0.8] |
| Q2 paraphrase | 1.5 | 1.6 | 1.4 | −0.2 [−1.1, 0.5] |
| Q3 unrelated sentences in the question | 4.6 | 14.9 | **4.6** | **−10.3 [−13.5, −7.2]** |
| T1 unrelated note in the state | 2.2 | 2.8 | 1.7 | −1.1 [−2.4, 0.0] |
| T2 observer's opinion in the state | 12.1 | 16.9 | **5.5** | **−11.3 [−14.6, −8.2]** |
| T3 opinion through an analogy | 6.9 | 12.7 | **6.3** | **−6.4 [−8.8, −3.6]** |
| P1 direct override | 8.9 | 7.9 | 5.7 | −2.2 [−3.6, −0.9] |
| P2 authority impersonation | 10.1 | 13.7 | **6.7** | **−7.0 [−9.0, −5.1]** |
| P3 fake validation note | 8.1 | 10.3 | 7.8 | −2.6 [−4.2, −1.0] |
| **Mean of the nine** | **6.1** | **9.2** | **4.6** | **−4.6 [−5.5, −3.6]** |
| Questions flipped by at least one attack | 29.2 | 41.7 | 20.7 | |
| Clean accuracy, 143 human-reviewed questions | 87.4 | 79.0 | 76.2 | −2.8 [−6.2, −0.6] |

- **Jev column:** `jev-1.13.0`'s responses as released by the benchmark authors, scored by the same code. v1.2 − Jev on the mean: −1.6 points [−2.7, −0.4].
- **Clean accuracy:** Jev is more accurate than both Wald versions on the clean human-reviewed questions. v1.2 makes the same clean decision as v1.1 on 98.6 % of the 812 questions.
- **What this does not show:** the perturbation kinds for v1.2 were chosen after we saw v1.1's per-attack results, so the benchmark's attack families informed the training design. Its texts did not. Robustness to attack types outside these families has not been measured.

### JevBench and Decision Index

**JevBench:** JevBench's own CLI (`fstandhartinger/jevbench` at `9ec6f15a`, `typesafe` adapter) against the packaged server on loopback, one request at a time. v1.2 with `none` on one RTX 5090: easy 48/48, original 72/72, hard 84/111; no tokens generated. v1.1 with `none` on one RTX PRO 6000: easy 48/48, original 72/72, hard 83/111; with `medium` 205/231, p95 1.80 s. The public items were used as a development scoreboard (never as training data), so this is not a held-out result. The JevBench leaderboard publishes a score only after its maintainers run the model themselves.

**Re-read after release:** we downloaded the `v1.2` revision anonymously and repeated the `none` read. From a directory holding only the downloaded model files it reproduced 204/231 with the same option on all 231 items (ECE 0.045). From the complete repository directory, two reads gave 204/231 and 205/231 (ECE 0.053), with a different option on one or two near-tied items. The model files are byte-identical in every case; we have not yet explained this small difference. [Details](https://huggingface.co/org2ai/Wald-4B/blob/main/evaluation/v1.2/release-check.json)

**Decision Index (v1.1):** 150,317/150,317 requests succeeded, including HLE. Measured on one RTX PRO 6000 96 GB with the pinned reproduction kit. [Full results](https://huggingface.co/datasets/org2ai/Wald-Q4B-decision-index-results/tree/805716601b2466be324ed6716407b4c3d9267faa/runs/wald-q4b-22d0-f7-full021) · [Per-benchmark scores](https://huggingface.co/org2ai/Wald-4B/blob/main/evaluation/benchmark-summary.json) · [Reproduction guide](https://huggingface.co/org2ai/Wald-4B/blob/main/RUNBOOK.md). The files under `evaluation/` on `main` belong to this v1.1 run; v1.2's numbers are in [`evaluation/v1.2/summary.json`](https://huggingface.co/org2ai/Wald-4B/blob/v1.2/evaluation/v1.2/summary.json) at the `v1.2` revision. The v1.2 sample row above is a one-pass read of 6,948 requests and cannot be compared with 54.59.

### Latency

Latency was measured on v1.1 and not again on v1.2, which has the same architecture, size and server. With `none`, the v1.1 JevBench run measured **33 ms median and 168 ms p95** per decision on one RTX PRO 6000. A 32-request serial preflight of `high` measured **821 ms median** on the same GPU; this small preflight is not a full-suite latency result or the Decision Index maintainers' admission test. Effort, context length, option count and concurrency all affect speed.

## Related projects and how Wald compares

Several projects implement or approximate structured decisions with calibrated option probabilities. The names below belong to their owners; Wald is not affiliated with any of them.

- **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** is TypeSafe AI's hosted decision model behind the `/v1/systemone` API; its weights are closed. Wald accepts the same request shape and runs on your own GPU.
- **[Kev](https://github.com/jaredpalmer/kev)** by Jared Palmer is an open-weight project that adds a LoRA and a pointer head to Qwen3.5 base models (0.8B, 4B and 9B). Wald's request parsing adapts Kev's Apache-2.0 code ([NOTICE](https://huggingface.co/org2ai/Wald-4B/blob/main/NOTICE)); Wald reads option letters from the language-model head instead of a separate head.
- **[Laya](https://huggingface.co/convaiinnovations/laya)** ([code](https://github.com/NandhaKishorM/laya)) is an open-weight 421M ModernBERT-large encoder with a decision head. It is much smaller than Wald and reads up to 512 tokens.

**JevBench public set, same 231 items (identical dataset hash), JevBench CLI, run by us:**

| System | How it was run | Correct |
|---|---|---:|
| Wald-Q4B v1.2 · `none` | Self-hosted, RTX 5090, 2026-09-30 | 204/231 |
| Wald-Q4B v1.1 · `none` | Self-hosted, RTX PRO 6000, 2026-09-29 | 203/231 |
| Jev (`jev-1.13.0`) | TypeSafe's hosted API, 2026-09-25 | 200/231 |
| Laya (English checkpoint `55cf4c4e`) | Self-hosted, NVIDIA L4, 2026-09-26 | 134/231 |

A difference of a few items on 231 is within run-to-run and sampling noise. The public items informed Wald's development, and 52 of the 231 states are longer than Laya's 512-token window.

**Decision Index 0.2.1:**

| System | Index | Source |
|---|---:|---|
| Jev (`jev-1.13.0`) | 57.91 | [Leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index), maintainer-run (data of 2026-09-28) |
| Wald-Q4B v1.1 · `high` | 54.59 | Author-run complete suite; not on the leaderboard yet ([PR #30](https://github.com/apolinario/decision-index/pull/30)) |
| Kev 9B | 38.48 | Leaderboard, maintainer-run (data of 2026-09-28) |
| Kev 4B | 34.64 | Leaderboard, maintainer-run (data of 2026-09-28) |

Leaderboard rows are scored by the maintainers; Wald's number is self-run with the official kit and may change after validation.

## FAQ

**Is there an open-source alternative to Jev?** Wald-Q4B is one open-weight option: Apache-2.0 weights and serving code that you run yourself, with a Jev-compatible `/v1/systemone` API. Kev and Laya (above) are other open projects. Wald is independent and is not a TypeSafe release.

**Can I use a Jev client with a self-hosted model?** Point the client at your own endpoint. The included server accepts `state` plus typed `questions` (`choice`, `noul`, `score`) at `POST /v1/systemone` and answers with TypeSafe's answer keys. It does not check API keys. See the [API reference](https://huggingface.co/org2ai/Wald-4B/blob/main/docs/api.md).

**How do I route tools or decide whether to ask the user?** Send the conversation or task as `state`. For tool routing, ask a `choice` question whose options are your tools. To decide whether to ask a clarifying question, ask a `noul` question such as "Is the request specific enough to act on without asking?" Act when the probability is high, ask when it is low, and set both thresholds on your own validation data. The model picks the tool; it does not write the tool's arguments.

**Is v1.2 safe against prompt injection?** No model is. v1.2 changes its answer about half as often as v1.1 under the nine JevAdvBench attacks, and it still flips on 4.6 % of attacked questions on average and on 20.7 % of questions under at least one attack. Treat it as one layer: keep untrusted text out of the instructions where you can, and check high-stakes decisions.

**How calibrated are the probabilities?** On the JevBench public set with `none`, expected calibration error is 0.045 for v1.2 (0.053 in the re-read from the complete release directory) and 0.041 for v1.1 (10 bins); the Brier scores are 0.191 and 0.188. v1.2 uses v1.1's temperature table unchanged. It was fitted on held-out rows of our own development data, with no JevBench items and with known Decision Index matches excluded. A confidence is not a guarantee; check calibration on your task.

**Does it run on a single GPU or a laptop?** The shipped server needs one NVIDIA GPU on Linux (vLLM 0.30.0); the BF16 weights are 8.4 GB. The v1.1 measurements come from an RTX PRO 6000 96 GB and the v1.2 measurements from an RTX 5090 32 GB; earlier builds of the same 4B architecture have also been served with vLLM on a 24 GB NVIDIA L4 at a 16K context limit. CPU, Apple Silicon and laptop setups are not supported by the shipped server and have not been tested.

**Kev vs Wald, or Laya vs Wald?** All three are open-weight. Kev adds a pointer head and LoRA to Qwen3.5 base models; Laya is a small encoder with a decision head; Wald is a fully trained 4B decoder with optional thinking. Our same-protocol measurements are in the tables above. Choose on your own task, latency budget and hardware.

**Can I fine-tune it for my task?** It is a standard Transformers checkpoint, so common LoRA tooling applies. For v1.0 we trained per-task LoRAs for $0.12–$1.81 of GPU time each; that tooling is not public yet, and those adapters are not validated on v1.1 or v1.2 ([v1.0 notes](https://huggingface.co/org2ai/Wald-4B/blob/main/history/v1.0/README.md)).

**What is the license?** Apache-2.0 for the weights and code. The base model, Qwen3.5-4B-Base, is also Apache-2.0. Some public training sources have their own terms or no stated licence; they are listed in [PROVENANCE.md](https://huggingface.co/org2ai/Wald-4B/blob/main/PROVENANCE.md). The model and code licence does not grant rights in those texts. v1.2 adds no new source dataset; its added training text was written by Claude Haiku, as described above.

## Limits

- **v1.2 trades a little clean accuracy for robustness:** −2.8 points on JevAdvBench's 143 human-reviewed clean questions (95 % CI [−6.2, −0.6]). Use `v1.1` if that matters more to you.
- **v1.2 is a one-pass model.** Its results are all `none`, it has no complete Decision Index run, and in our one check the thinking efforts lowered its JevBench public score (198/231 against 204/231). Use v1.1 for thinking.
- **Robustness is measured on one benchmark** and on the attack families that informed the training design. It is not a security guarantee.
- Confidence is not a guarantee of correctness; validate thresholds on your own task. Oversized prompts are rejected rather than truncated.
- Known exact training overlaps were filtered, but semantic overlap and pretraining contamination are not ruled out; development used visible benchmark samples. Source-text rights vary. See [evaluation notes](https://huggingface.co/org2ai/Wald-4B/blob/main/CONTAMINATION.md) and [source attribution](https://huggingface.co/org2ai/Wald-4B/blob/main/PROVENANCE.md).

## Versioning

| Version | Revision | Checkpoint | What it is |
|---|---|---|---|
| **v1.2 — robustness release** | tag [`v1.2`](https://huggingface.co/org2ai/Wald-4B/tree/v1.2); weights on `main` | `02600-f19` | v1.1 + merged robustness LoRA; default effort `none` |
| **v1.1 — general release** | tag [`v1.1`](https://huggingface.co/org2ai/Wald-4B/tree/v1.1) | `022D0-f7` | Complete Decision Index run (54.59); default effort `high` |
| v1.0 — archive | tag [`v1.0`](https://huggingface.co/org2ai/Wald-4B/tree/v1.0) | `021A0-f10` | Default effort `medium` |

The pending benchmark requests for v1.1 (JevBench issue #146, Decision Index PR #30) name fixed commits of this repository and are not affected by v1.2. The v1.0 model card retains its XL, task-LoRA and latency reports, and the [v1.0 walkthrough slides](https://claude.ai/artifact/XfCHVaCuj9A5ectrpaWzV5) describe v1.0 only. Those measurements belong to their documented builds.

## Citation

Wald-Q4B (2026), an open-weight 4B decision model with calibrated option probabilities. https://huggingface.co/org2ai/Wald-4B. Name the revision you used (`v1.2` or `v1.1`).

```bibtex
@misc{wald_q4b_2026,
  title        = {Wald-Q4B: an open-weight 4B decision model with calibrated option probabilities},
  author       = {{Wald-4B authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/org2ai/Wald-4B}},
  note         = {Revision v1.2}
}
```

Machine-readable: [CITATION.cff](https://huggingface.co/org2ai/Wald-4B/blob/main/CITATION.cff) · [llms.txt](https://huggingface.co/org2ai/Wald-4B/blob/main/llms.txt) · [model-info.json](https://huggingface.co/org2ai/Wald-4B/blob/main/model-info.json)

---

[Model](https://huggingface.co/org2ai/Wald-4B) · [GitHub](https://github.com/org2AI/wald-4b) · [Decision Index results](https://huggingface.co/datasets/org2ai/Wald-Q4B-decision-index-results) · Apache-2.0 for weights and code. [Third-party notices](https://huggingface.co/org2ai/Wald-4B/blob/main/NOTICE) · [Training-data usage notes](https://huggingface.co/org2ai/Wald-4B/blob/main/PROVENANCE.md)
