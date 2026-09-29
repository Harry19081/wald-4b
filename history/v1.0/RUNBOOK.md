# Runbook for the Decision Index maintainers

These steps serve Wald-4B on one NVIDIA RTX PRO 6000 (96 GB) so that the kit's `http` engine can run the full 0.2.1
suite against it.

## 1. Install

The weights are in the Hub repo `Harry19081/Wald-4B` (bf16 safetensors, tokenizer, config, `temperature.json`,
`serving.json`, `MANIFEST.json`). The code is in this repository, pinned by commit.

```sh
huggingface-cli download Harry19081/Wald-4B --local-dir /models/Wald-4B
cd /models/Wald-4B && python - <<'EOF'   # optional: verify every file against MANIFEST.json
import hashlib, json; m = json.load(open("MANIFEST.json"))
for f, v in m.items():
    h = hashlib.sha256(open(f, "rb").read()).hexdigest(); assert h == v["sha256"], f
print("ok", len(m))
EOF
```

Choose one of the following.

- **Docker:**
  ```sh
  docker build -t wald-serve .
  docker run --gpus '"device=0"' -v /models/Wald-4B:/model:ro -p 8000:8000 wald-serve
  ```
- **No Docker** (Python 3.12 and `uv`):
  ```sh
  ./run.sh /models/Wald-4B
  ```
- **By hand:**
  ```sh
  uv venv -p 3.12 .venv && VIRTUAL_ENV=.venv uv pip install "vllm==0.30.0" ./server
  VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/wald-serve --model /models/Wald-4B --port 8000
  ```
  `VLLM_USE_FLASHINFER_SAMPLER=0` is needed because FlashInfer's sampler refuses sm_120 in vLLM 0.30.0.

In every case, `wald-serve` starts vLLM 0.30.0 on the weights: bf16, loopback only, `--max-model-len 131072`,
`--gpu-memory-utilization 0.90`, `--max-num-seqs 256`, `--seed 0`. It then serves `POST /v1/systemone` on port 8000. The
service is ready when `curl -s localhost:8000/health` returns `{"ok": true, ...}` with the effective configuration. The
first start takes about 2–4 minutes (weights load and CUDA graph capture).

## 2. Run the suite

```sh
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=wald-4b \
    --option timeout=3600 ...
```

- **Send requests concurrently.** We ran 16–32 kit processes over group-balanced shards against one server, and vLLM
  batches them. A single sequential client would be several times slower (section 4).
- **Use a long timeout (≥ 3,600 s).** Under load, the p95 request latency was several minutes, from the requests with
  many long questions (BRIGHT / ToolRet chunks, ContractNLI, ACOS).
- **A 422 means a declared capacity limit**, and the body contains `maximum context length`. Count it as unsupported,
  not as an error.
- **Smoke test:**
  ```sh
  curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{"state":"I was charged twice.","questions":{"q":{"type":"noul","instructions":"Is this a billing problem?"}}}'
  ```

## 3. Declared configuration and limits

| item | value |
|---|---|
| declared policy | **effort `medium`**: one pass; if the top probability is below 0.7, a thought of at most 512 tokens (temperature 0.6, top-p 0.95, top-k 20, seed = hash of the request and question id), then a second read |
| prompt format | `repeat_state_plain` (the state written twice), from `serving.json`; the same for every benchmark |
| readout | option-letter logits at the last position; tempered with `temperature.json` (A table for one-pass answers, B512 after a thought) |
| > 26 options | knockout: ⌈n / 26⌉ consecutive chunks, then a final over the chunk winners; one pass; up to 676 options (the wire allows 255) |
| context | 131,072 tokens per question prompt; longer → HTTP 422 `maximum context length`, never truncated |
| thought that does not fit | if prompt + 512 + 64 tokens exceeds the context, the one-pass answer is returned |
| questions per request | no fixed limit; read concurrently (8 at a time per request); tested up to 64 |
| determinism | same request → same thought seed → same answer, up to bf16 batch-order noise (argmax agreement 99.7 % between two reads of 36k questions) |

Nothing in the server depends on the benchmark: there is no per-benchmark prompt, option filtering, retry or truncation.

## 4. Expected runtime of the full suite on one RTX PRO 6000

The full suite has about 150,500 scoreable requests, including the display-only MMLU / ARC and RouterBench (not scored
in 0.2.1). HLE is not counted here.

**How we estimated.** Our basis is the measured wall time of our 6,948-request sample reads:
- one pass: 875 s;
- medium: 1,702 s. Both were measured on one card shared with a second engine, with 16 kit processes each.
- always-think: 5,520 s, on a shared card, including a k = 4 read of the knowledge and language rows.

We scaled these by a full-to-sample work ratio of 12–15×. That ratio weights each benchmark's per-request latency by its
full size; the sample over-represents long-prompt benchmarks. A dedicated card is assumed to be 1.25–1.7× faster than
our shared runs. `repeat_state_plain` writes the state twice, which adds about 30–60 % to prefill: this is an estimate,
not measured on the full suite. The in-request question concurrency of this server shortens the tail of requests with
many questions, and the table does not credit it.

| policy | thinks on | plain prompt | repeat_state_plain (declared) |
|---|---|---|---|
| none | 0 % | ≈ 2–3 h | ≈ 2.5–4.5 h |
| low | a few % | ≈ 2.5–4 h | ≈ 3–5.5 h |
| **medium (declared)** | ≈ 12 % of questions | **≈ 3.5–6 h** | **≈ 4.5–9 h** |
| high | every 2–26-option question | ≈ 7–12 h | ≈ 9–16 h |
| high-k4 | every question, 4 thoughts | ≈ 12–20 h | ≈ 15–25 h |

These are estimates, and our own full-suite read with the packaged server will replace them (CHECKLIST.md). We declare
`medium`. `high-k4` would take most of a day on one card: prefill for every question would be about 2.3× that of `high`,
and decoding 4×. Our measured `high-k4` gain over `high` was not significant (+0.48 [−0.4, +1.5] on the parent
checkpoint), and that measurement applied k = 4 only to the knowledge and language benchmarks.

## 5. Throughput and latency (for reference)

- One question, one pass, serial: p50 26 ms on one RTX PRO 6000; 51 ms on an H100.
- `medium`, serial: p50 ≈ 52 ms, p95 ≈ 1.1 s (the questions that think).
- Batched one-pass throughput: about 115 decisions/s on one RTX PRO 6000 (fp8). bf16 throughput was not measured separately.
