# Reproduce Wald-Q4B 22D0-f7

Release: **v1.1** · checkpoint `022D0-f7`. Previous release: **v1.0**.

Download the immutable HF tag 22D0-f7 (or the full HF commit in the submission) into a fresh directory. Do not mix old v1.0 single-file weights with the new shards. MANIFEST.json lists the exact model, tokenizer, temperature and code hashes.

## Convenient packaged server

```sh
hf download Harry19081/Wald-4B --revision v1.1 --local-dir ./Wald-Q4B-22D
cd Wald-Q4B-22D
./run.sh "$PWD"
```

Default policy is high, repeat_state_plain, context 131072. Lower effort modes are alternative configurations with no 54.59 claim. GPU requirements: vLLM 0.30.0, BF16, NVIDIA RTX PRO 6000 96GB for comparison. The packaged server has parity tests against the reference below for prompts and answer probabilities. Scheduling may differ; latency is not established by those tests.

## Frozen reference protocol

The full run used eval.systemone_vllm from the private development checkout; that checkout was not committed at launch. The included reference/ files freeze the release-time implementation, with hashes in MANIFEST.json. This code provenance limitation is disclosed rather than presenting a later public commit as the original launch commit.

```sh
python -m vllm.entrypoints.openai.api_server --model "$PWD" --served-model-name Wald-Q4B-022D0-f7-full021 --host 127.0.0.1 --port 8321 --max-model-len 131072 --gpu-memory-utilization 0.60 --max-num-seqs 128 --seed 0
# Separate terminal, same environment:
PYTHONPATH="$PWD/reference" python -m eval.systemone_vllm --vllm http://127.0.0.1:8321 --served Wald-Q4B-022D0-f7-full021 --gate 1.01 --budget 512 --temperature "$PWD/temperature.json" --wide knockout --template paren --max-model-len 131072 --return-raw --prompt-format repeat_state_plain --port 8421 --model-name Wald-Q4B-022D0-f7-full021
```

Install the pinned kit `apolinario/decision-index@87d4650b42b377c0291a89c1f1a879f9b31082bf`, rebuild and verify its complete suite locally, then run its http engine against port 8421. Suite payloads are not redistributed. No request or option pruning/truncation. Complete saved responses are untouched and suitable for maintainer rescoring.

Capacity: 131,072 tokens. Wide questions use ordered knockout over all supplied options. A thought that cannot fit falls back to the one-pass read. The full run had 0 unsupported and 0 errors. Concurrency ranged 16–40 request runners for throughput; timing is not the maintainer's serial admission gate. The 32-request preflight median 821.3 ms does not replace 750 private serial requests after warm-up.

Training/data/calibration limitations are in CONTAMINATION.md and PROVENANCE.md. Official admission remains pending.
