# Fine-tune Wald-4B on your task (CLI, releasing soon)

> **Preview.** The CLI is not public yet; we plan to release it soon. The commands below are preview syntax and may
> change before the release.

**One dataset in, one task model out, with an honest comparison to Jev.** The `vertical` CLI takes a labelled dataset (a
Hugging Face repo, a URL, or your own file) and trains a LoRA on Wald-4B for that one task. It then reads the task's
fixed test items with your LoRA, with Jev and with zero-shot baselines, using byte-identical requests, and writes a report
with paired confidence intervals and calibration. The verticals in the [README](../README.md#verticals-a-quick-lora-per-task)
were all made this way. One LoRA costs $0.12–$1.81 of GPU time (under 2 GPU-hours).

![Architecture of the vertical CLI](../figures/lora-cli-architecture.svg)

## The pipeline

Free stages run on your machine. Paid stages run on a GPU backend you choose: any SSH GPU box, RunPod or Modal. Every
stage is resumable. Reads are cached per item, a finished LoRA is not trained again, and each run keeps a manifest with
the sha256 of every output.

| stage | command | what it does |
|---|---|---|
| task spec | `init` · `inspect` · `validate` | Scaffold `task.yaml` from a dataset: the source pinned to a commit, fields, labels, question, metric and split sizes. `inspect` renders example records as the model will see them. |
| prep | `prep` | Download the data and hash every file. Render each item once, both as a decision record and as the exact request Jev receives. Make a seeded, label-stratified, group-aware split into train · calib · dev · test, and dedupe across splits. |
| overlap check | `prep` | Compare held-out items with the train pool, with Decision Index items and with Wald-4B's training data. The counts go in the data manifest. |
| audit | `audit` · `approve` | Free, deterministic rule checks before and after every stage, plus a review of a sample by a cheap model. A FAIL blocks the next paid stage. |
| augment (optional) | `augment` | Extra training rows: soft labels for unlabelled texts, synthetic items, or paraphrases. The teacher is your own coding agent or any OpenAI-compatible API. Every row is re-checked on ingest (schema, labels, dedupe, overlap with the held-out splits). |
| baseline | `baseline` | Read calib, dev and test with Jev (cached per request, so re-runs are free) and with zero-shot Wald-4B and the raw base. Optionally add frontier LLMs, plus random and class-prior floors. |
| train | `train` · `watch` | One LoRA per arm. An arm is a training size, for example 300 labels, 1,000 labels, or all of them. The recipe is rank 32 on every projection, lr 1e-4, 2 epochs, and KL replay toward the base's own answers, so the base keeps its other skills. `watch` tails the log and stops a bad run. |
| eval | `eval` | One vLLM server holds Wald-4B plus every adapter. Reads use the serving reader: letter readout, knockout above 26 options, and optional thinking effort. |
| calibrate | `calibrate` | Fit one temperature per system on the calib split, never on test. It changes confidence, never the answer. Jev is scored as served. |
| report | `report` · `compare` | `report.md` + `report.json`: the test metric with 2,000 paired bootstrap resamples against Jev and the base, ECE, high-confidence errors, and dev numbers for picking a configuration. `compare` gives paired differences between two runs, for example Wald-4B vs the raw base. |
| deploy | `serve` · `try` · `latency` | Serve the adapter and its temperature on Wald-4B behind the same `/v1/systemone` decision API. The adapter stays a separate file, and Wald-4B itself is never changed. `try` sends one item through Jev and your model side by side, and `latency` measures serial latency on an otherwise idle card. |

## Guards

A task model is only useful if its numbers are real. The CLI enforces these rules itself:

- **Same bytes everywhere.** Each item is rendered once, into one request. Jev receives that request, our reader reads
  it, and your LoRA trains on the same bytes.
- **Leak gate.** Before training, any test item whose word 5-gram Jaccard similarity with a train or calib item is 0.8 or
  higher stops the run. You can drop those items from this run (`--drop-near-dups`), or a person can approve keeping them.
- **Audit gate.** A FAIL (exit code 4) blocks every paid stage. Only a person can override it: `vertical approve` asks
  y/N on a real terminal, or you click *Approve override* in the UI. The approval is signed, and it expires when the
  data or the audit result changes. An agent cannot sign it.
- **Held-out discipline.** Temperatures are fitted on calib. Configurations are chosen on dev. Test is reported once per
  configuration, and a best-of-N read on test is labelled as best-of-N.
- **Cost gate.** Paid stages print a cost estimate and run only with `--yes`, or when the estimate fits under `--max-usd`.
  `--dry-run` prints the plan and changes nothing.
- **Training watch.** A NaN or infinite loss, a zero learning rate, the wrong base, or a cost above twice the estimate
  stops the job.
- **Secrets.** Keys are passed at run time and never reach the GPU box's disk or the logs.

## Built for coding agents

Every command takes `--json` (one JSON object on stdout, with a `next` hint) and has fixed exit codes: 0 ok, 2 user or
config error, 3 remote failure, 4 audit FAIL. Spending money and overriding an audit still need a person's yes.

## Cost and time

- **One LoRA:** $0.12–$1.81 of GPU time on one RTX PRO 6000 at about $1 per hour (under 2 GPU-hours). Training runs at
  about 5,000 tokens/s.
- **A full run** has cost $0.25–$2.15 on our public tasks. That covers the baseline reads, one to three training sizes,
  eval and the report. Jev reads are billed by Jev and cached, so re-runs do not pay for them again.

## Example (preview syntax)

A task is one `task.yaml`:

```yaml
name: banking77
title: Bank customer intent routing
source: {hf: legacy-datasets/banking77, revision: <commit>, license: CC-BY-4.0, splits: {train: train, test: test}}
fields: {text: text, label: label}
labels: {from_features: true}
state_template: "Customer message: {text}"
question: Which intent does this bank customer's message express?
metric: macro_f1
sizes: {calib: 300, test: 500}
arms: [n300, n1000, all]
```

```sh
vertical init banking77 --hf legacy-datasets/banking77   # scaffold task.yaml
vertical inspect banking77                               # rendered records, labels, lengths
vertical validate banking77
vertical prep banking77                                  # download, split, dedupe, overlap checks
vertical audit banking77 --stage data
vertical run banking77 --dry-run                         # cost of every paid stage; nothing runs
vertical run banking77 --max-usd 3                       # baseline → train → eval → calibrate → report
vertical report banking77                                # report.md + report.json
vertical serve banking77 --arm all --yes                 # the adapter behind /v1/systemone
vertical try banking77 --arm all --text "My card still hasn't arrived"
```

Other commands: `list`, `status`, `stop`, `backends --check`, `compare`, `latency`, `note`, `import-read` (adds a read
made outside the CLI to a run's report).
