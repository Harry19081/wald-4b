# Runbook

This revision (`v1.2`) holds **Wald-Q4B v1.2**, checkpoint `02600-f19`. Section 1 serves it and reproduces its JevBench public read. Section 2 is the unchanged v1.1 runbook for the complete Decision Index run: that run belongs to **v1.1** (checkpoint `022D0-f7`), so download `--revision v1.1` for it.

The server code, prompt format, tokenizer and temperature table are byte-identical in v1.1 and v1.2. The two weight shards differ, and `serving.json` declares effort `none` in v1.2 (`high` in v1.1). MANIFEST.json lists the exact model, tokenizer, temperature and code hashes of this revision.

## 1. Wald-Q4B v1.2

```sh
hf download org2ai/Wald-4B --revision v1.2 --local-dir ./Wald-Q4B-v1.2
cd Wald-Q4B-v1.2
./run.sh "$PWD"          # effort none (declared in serving.json), repeat_state_plain, context 131072
```

`GET /health` must report `"effort": "none"`. v1.2's JevBench and JevAdvBench results were measured with effort `none` and `repeat_state_plain` on one NVIDIA RTX 5090 32 GB (vLLM 0.30.0, BF16); the JevAdvBench read used a 32,768-token context limit.

JevBench public set (204/231), with `fstandhartinger/jevbench` at `9ec6f15a`:

```sh
cd jevbench          # a checkout of fstandhartinger/jevbench at 9ec6f15a
for T in easy original hard; do
  python -m jevbench.cli run --tasks datasets/public/$T.jsonl --adapter typesafe --endpoint http://127.0.0.1:8000 \
    --model jev-latest --key-env '' --reserve-usd 0 --cost-basis self_hosted_loopback_no_tariff \
    --results out/$T/results.jsonl --raw-dir out/raw-$T --ledger out/$T/ledger.jsonl --manifest out/$T/manifest.json \
    --run-label wald-q4b-v1.2-pub-$T
done
cat out/easy/results.jsonl out/original/results.jsonl out/hard/results.jsonl > out/results-all.jsonl
python -m jevbench.cli summarize --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --results out/results-all.jsonl --public-export out/summary-all.json
```

JevAdvBench: send the benchmark's requests (`JevAdvBench/JevAdvBench` at `3218e05`, one question per request) to the same endpoint and score the answers with the benchmark's own analysis code. The benchmark data is CC BY-NC 4.0 and is not redistributed here.

## 2. Wald-Q4B v1.1: complete Decision Index run

### Reproduce Wald-Q4B 22D0-f7

Release: **v1.1** · checkpoint `022D0-f7`. Previous release: **v1.0**.

Download the immutable HF tag 22D0-f7 (or the full HF commit in the submission) into a fresh directory. Do not mix old v1.0 single-file weights with the new shards. MANIFEST.json lists the exact model, tokenizer, temperature and code hashes.

### Convenient packaged server

```sh
hf download org2ai/Wald-4B --revision v1.1 --local-dir ./Wald-Q4B-22D
cd Wald-Q4B-22D
./run.sh "$PWD"
```

Default policy is high, repeat_state_plain, context 131072. Lower effort modes are alternative configurations with no 54.59 claim. GPU requirements: vLLM 0.30.0, BF16, NVIDIA RTX PRO 6000 96GB for comparison. The packaged server has parity tests against the reference below for prompts and answer probabilities. Scheduling may differ; latency is not established by those tests.

### Frozen reference protocol

The full run used eval.systemone_vllm from the private development checkout; that checkout was not committed at launch. The included reference/ files freeze the release-time implementation, with hashes in MANIFEST.json. This code provenance limitation is disclosed rather than presenting a later public commit as the original launch commit.

```sh
python -m vllm.entrypoints.openai.api_server --model "$PWD" --served-model-name Wald-Q4B-022D0-f7-full021 --host 127.0.0.1 --port 8321 --max-model-len 131072 --gpu-memory-utilization 0.60 --max-num-seqs 128 --seed 0
# Separate terminal, same environment:
PYTHONPATH="$PWD/reference" python -m eval.systemone_vllm --vllm http://127.0.0.1:8321 --served Wald-Q4B-022D0-f7-full021 --gate 1.01 --budget 512 --temperature "$PWD/temperature.json" --wide knockout --template paren --max-model-len 131072 --return-raw --prompt-format repeat_state_plain --port 8421 --model-name Wald-Q4B-022D0-f7-full021
```

Install the pinned kit `apolinario/decision-index@87d4650b42b377c0291a89c1f1a879f9b31082bf`, rebuild and verify its complete suite locally, then run its http engine against port 8421. Suite payloads are not redistributed. No request or option pruning/truncation. Complete saved responses are untouched and suitable for maintainer rescoring.

Capacity: 131,072 tokens. Wide questions use ordered knockout over all supplied options. A thought that cannot fit falls back to the one-pass read. The full run had 0 unsupported and 0 errors. Concurrency ranged 16–40 request runners for throughput; timing is not the maintainer's serial admission gate. The 32-request preflight median 821.3 ms does not replace 750 private serial requests after warm-up.

Training/data/calibration limitations are in CONTAMINATION.md and PROVENANCE.md. Official admission remains pending.
