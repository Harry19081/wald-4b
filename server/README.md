# wald-serve

The TypeSafe `POST /v1/systemone` server for Wald-4B: the option-letter readout over a vLLM OpenAI-compatible server or
llama.cpp's `llama-server`, the effort gate (none / low / medium / high / high-k), the knockout for more than 26
options, and the temperature table. It needs only public packages: `pydantic`, plus `vllm==0.30.0` or llama.cpp on the
machine that runs the weights.

```sh
pip install ".[vllm]"                              # on a Linux CUDA machine
wald-serve --model /path/to/Wald-4B --port 8000    # starts vLLM on the weights and serves /v1/systemone
```

GGUF weights (CPU, Apple Silicon or any GPU llama.cpp supports):

```sh
pip install .
wald-serve --gguf ./Wald-4B-Q8_0.gguf --port 8000                     # starts llama-server (on PATH) on the file
wald-serve --llamacpp http://127.0.0.1:8080 --temperature temperature.json \
  --prompt-format repeat_state_plain --effort none --port 8000        # or attach to a running llama-server
```

With `--gguf`, `serving.json` and `temperature.json` are read from the GGUF's folder. The llama.cpp backend sends token
ids to `/completion` and reads the letters from its top 100 next-token logprobs (`n_probs`); a letter outside them gets
the same floor as in vLLM. llama.cpp's context is `--max-model-len` × `--llama-parallel` (default 4 slots).

Tests (no model, no GPU; a fake server stands in for vLLM and llama-server): `pip install ".[test]" && cd tests && pytest -q`.
`tests/test_parity.py` compares prompts and answers byte for byte with the reference implementation the published
numbers were measured with; it is skipped unless that implementation is importable (WALD_PARITY_SRC set).
