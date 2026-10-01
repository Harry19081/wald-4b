<div align="center">
  <h1>Wald-Q4B v1.1</h1>
  <p><strong>Decide directly. Think when needed. Return probabilities.</strong></p>
  <p><a href="README.md">English</a> · <a href="docs/readmes/README.zh.md">简体中文</a> · <a href="https://huggingface.co/org2ai/Wald-4B">Weights on Hugging Face</a> · <a href="docs/api.md">API</a></p>
</div>

**Wald-Q4B v1.1 is an open-weight 4B decision model: give it a state and a set of options, and it returns a calibrated probability for every option.** It is for developers who build agents and pipelines and need a fast, self-hosted component to pick a tool, route a request, classify an input or decide whether to ask the user. Unlike a chat model, it does not write an answer you have to parse. It reads the options in one pass (33 ms median on one RTX PRO 6000 with effort `none`) and can optionally think first. It serves a Jev-compatible `POST /v1/systemone` API, is built on Qwen3.5-4B-Base and is released under Apache-2.0.

Wald-Q4B is an independent, self-hosted alternative to TypeSafe's hosted Jev API. It is not Jev, contains no Jev weights, and is not affiliated with or endorsed by TypeSafe AI. The Hugging Face repository is `org2ai/Wald-4B` (earlier name: Wald-4B; moved from `Harry19081/Wald-4B` on 2026-10-01, old links redirect).

**W**ait **A** bit, **L**ook, then **D**ecide. Also named after Abraham Wald, the pioneer of sequential analysis: stop when the evidence is enough.

## At a glance

- **4B parameters**, built on Qwen3.5-4B-Base; BF16 weights (8.4 GB).
- **Every option gets a probability.** Question types: `choice` (1–255 named options), `noul` (yes/no) and `score` (ordered levels).
- **Adjustable thinking:** `none`, `low`, `medium`, `high`, or several thoughts with `high-k`.
- **Decision Index 0.2.1: 54.59** on the complete suite with `high`; 150,317 requests, all successful. Author-run, pending maintainer validation.
- **JevBench public set: 203/231** with `none`, ECE 0.041, p50 33 ms / p95 168 ms. Self-scored with JevBench's own harness.
- **Self-hosted API:** `POST /v1/systemone`, up to 131,072 prompt tokens.

## Quick start

On a Linux machine with an NVIDIA GPU and [`uv`](https://docs.astral.sh/uv/):

```sh
hf download org2ai/Wald-4B --revision v1.1 --local-dir ./Wald-Q4B
cd Wald-Q4B
EFFORT=none ./run.sh "$PWD"     # one pass, lowest latency
# ./run.sh "$PWD"               # default: high, the evaluated Decision Index configuration
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

The answer for `route` contains the chosen key and a probability for each of `returns`, `delivery` and `other`. Request and response fields, yes/no and score questions, and a clarification example: [API reference](docs/api.md). `GET /health` reports the effective policy. The server uses vLLM 0.30.0 and the included `wald-serve` package; Docker and exact evaluation settings are in [RUNBOOK.md](RUNBOOK.md). Generic text-generation calls do not reproduce the decision API's readout.

## Thinking effort

Start with `none` for direct decisions, `medium` for confidence-gated thinking, or `high` for the evaluated v1.1 configuration.

| Effort | When it thinks | Thought budget |
|---|---|---|
| `none` | Direct option readout | No generated thought |
| `low` | Top initial probability < 0.5 | Up to 512 tokens |
| `medium` | Top initial probability < 0.7 | Up to 512 tokens |
| **`high` (default)** | Every eligible question | Up to 512 tokens |
| `high-k2` … `high-k8` | Multiple thoughts; average their answer distributions | Up to 512 tokens per thought |

Thinking applies to questions with 2–26 options when context space permits. Larger option sets use grouped readout and a final winner comparison; if a thought cannot fit, the initial answer is kept. Increasing effort spends more computation; it does not guarantee a better answer. **54.59 applies to `high` only; 203/231 applies to `none` only.**

Set the server default with `EFFORT=medium ./run.sh "$PWD"`, or override it per request with `"effort": "none"`.

## How it works

Wald first reads option-letter logits from a plain prompt and turns them into a probability distribution. If the effort policy asks for thinking, it generates a short thought and reads the options again. Bucketed temperature scaling calibrates the returned probabilities.

v1.1 combines full-parameter decision training, LoRA refinement, short-thought distillation and RLCD. Training uses **WaldGen**, our generated decision corpus, together with public training datasets. [Data sources](PROVENANCE.md) · [Evaluation notes](CONTAMINATION.md)

## Benchmarks

| Benchmark | Configuration | Result | Status |
|---|---|---:|---|
| **Decision Index 0.2.1 — complete suite** | v1.1 · `high` | **54.59** | Author-run; [PR #30](https://github.com/apolinario/decision-index/pull/30) awaits maintainer validation |
| **JevBench public set (231 items)** | v1.1 · `none` | **203/231** (87.9%) · ECE 0.041 · Brier 0.188 | Self-scored; [issue #146](https://github.com/fstandhartinger/jevbench/issues/146) asks the maintainers to measure it |

**Decision Index:** 150,317/150,317 requests succeeded, including HLE. Measured on one RTX PRO 6000 96 GB with the pinned reproduction kit. [Full results](https://huggingface.co/datasets/org2ai/Wald-Q4B-decision-index-results/tree/805716601b2466be324ed6716407b4c3d9267faa/runs/wald-q4b-22d0-f7-full021) · [Per-benchmark scores](evaluation/benchmark-summary.json) · [Reproduction guide](RUNBOOK.md)

**JevBench:** JevBench's own CLI (`fstandhartinger/jevbench` at `9ec6f15a`, `typesafe` adapter) against the packaged server on loopback, one request at a time, on one RTX PRO 6000. Easy 48/48, original 72/72, hard 83/111; no tokens generated. With `medium`: 205/231, p95 1.80 s. The public items were used as a development scoreboard (never as training data), so this is not a held-out result. The JevBench leaderboard publishes a score only after its maintainers run the model themselves.

### Latency

With `none`, the JevBench run above measured **33 ms median and 168 ms p95** per decision. A 32-request serial preflight of `high` measured **821 ms median** on the same GPU; this small preflight is not a full-suite latency result or the Decision Index maintainers' admission test. Effort, context length, option count and concurrency all affect speed.

## Related projects and how Wald compares

Several projects implement or approximate structured decisions with calibrated option probabilities. The names below belong to their owners; Wald is not affiliated with any of them.

- **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** is TypeSafe AI's hosted decision model behind the `/v1/systemone` API; its weights are closed. Wald accepts the same request shape and runs on your own GPU.
- **[Kev](https://github.com/jaredpalmer/kev)** by Jared Palmer is an open-weight project that adds a LoRA and a pointer head to Qwen3.5 base models (0.8B, 4B and 9B). Wald's request parsing adapts Kev's Apache-2.0 code ([NOTICE](NOTICE)); Wald reads option letters from the language-model head instead of a separate head.
- **[Laya](https://huggingface.co/convaiinnovations/laya)** ([code](https://github.com/NandhaKishorM/laya)) is an open-weight 421M ModernBERT-large encoder with a decision head. It is much smaller than Wald and reads up to 512 tokens.

**JevBench public set, same 231 items (identical dataset hash), JevBench CLI, run by us:**

| System | How it was run | Correct |
|---|---|---:|
| Wald-Q4B v1.1 · `none` | Self-hosted, RTX PRO 6000, 2026-09-29 | 203/231 |
| Jev (`jev-1.13.0`) | TypeSafe's hosted API, 2026-09-25 | 200/231 |
| Laya (English checkpoint `55cf4c4e`) | Self-hosted, NVIDIA L4, 2026-09-26 | 134/231 |

A 3-item difference on 231 items is within run-to-run and sampling noise. The public items informed Wald's development, and 52 of the 231 states are longer than Laya's 512-token window.

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

**Can I use a Jev client with a self-hosted model?** Point the client at your own endpoint. The included server accepts `state` plus typed `questions` (`choice`, `noul`, `score`) at `POST /v1/systemone` and answers with TypeSafe's answer keys. It does not check API keys. See the [API reference](docs/api.md).

**How do I route tools or decide whether to ask the user?** Send the conversation or task as `state`. For tool routing, ask a `choice` question whose options are your tools. To decide whether to ask a clarifying question, ask a `noul` question such as "Is the request specific enough to act on without asking?" Act when the probability is high, ask when it is low, and set both thresholds on your own validation data. The model picks the tool; it does not write the tool's arguments.

**How calibrated are the probabilities?** On the JevBench public set with `none`, expected calibration error is 0.041 (10 bins) and the Brier score is 0.188. Temperatures were fitted on held-out rows of our own development data, with no JevBench items and with known Decision Index matches excluded. A confidence is not a guarantee; check calibration on your task.

**Does it run on a single GPU or a laptop?** The shipped server needs one NVIDIA GPU on Linux (vLLM 0.30.0); the BF16 weights are 8.4 GB. The v1.1 measurements come from an RTX PRO 6000 96 GB; earlier builds of the same 4B architecture have also been served with vLLM on a 24 GB NVIDIA L4 at a 16K context limit. CPU, Apple Silicon and laptop setups are not supported by the shipped server and have not been tested.

**Kev vs Wald, or Laya vs Wald?** All three are open-weight. Kev adds a pointer head and LoRA to Qwen3.5 base models; Laya is a small encoder with a decision head; Wald is a fully trained 4B decoder with optional thinking. Our same-protocol measurements are in the tables above. Choose on your own task, latency budget and hardware.

**Can I fine-tune it for my task?** It is a standard Transformers checkpoint, so common LoRA tooling applies. For v1.0 we trained per-task LoRAs for $0.12–$1.81 of GPU time each; that tooling is not public yet, and those adapters are not validated on v1.1 ([v1.0 notes](history/v1.0/README.md)).

**What is the license?** Apache-2.0 for the weights and code. The base model, Qwen3.5-4B-Base, is also Apache-2.0. Some public training sources have their own terms or no stated licence; they are listed in [PROVENANCE.md](PROVENANCE.md). The model and code licence does not grant rights in those texts.

## Limits

Confidence is not a guarantee of correctness; validate thresholds on your own task. Oversized prompts are rejected rather than truncated. Known exact training overlaps were filtered, but semantic overlap and pretraining contamination are not ruled out; development used visible benchmark samples. Source-text rights vary. See [evaluation notes](CONTAMINATION.md) and [source attribution](PROVENANCE.md).

## Versioning

| Version | Checkpoint | Default effort |
|---|---|---|
| **v1.1 — current** | `022D0-f7` | `high` |
| [v1.0 — archive](https://huggingface.co/org2ai/Wald-4B/tree/v1.0) | `021A0-f10` | `medium` |

The v1.0 model card retains its XL, task-LoRA and latency reports, and the [v1.0 walkthrough slides](https://claude.ai/artifact/XfCHVaCuj9A5ectrpaWzV5) describe v1.0 only. Those measurements belong to their documented builds.

## Citation

Wald-Q4B v1.1 (2026), an open-weight 4B decision model with calibrated option probabilities. https://huggingface.co/org2ai/Wald-4B

```bibtex
@misc{wald_q4b_2026,
  title        = {Wald-Q4B v1.1: an open-weight 4B decision model with calibrated option probabilities},
  author       = {{Wald-4B authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/org2ai/Wald-4B}},
  note         = {Revision v1.1}
}
```

Machine-readable: [CITATION.cff](CITATION.cff) · [llms.txt](llms.txt) · [model-info.json](model-info.json)

---

[Model](https://huggingface.co/org2ai/Wald-4B) · [GitHub](https://github.com/org2AI/wald-4b) · [Decision Index results](https://huggingface.co/datasets/org2ai/Wald-Q4B-decision-index-results) · Apache-2.0 for weights and code. [Third-party notices](NOTICE) · [Training-data usage notes](PROVENANCE.md)
