# wald-serve

The TypeSafe `POST /v1/systemone` server for Wald-4B: the option-letter readout over a vLLM OpenAI-compatible server,
the effort gate (none / low / medium / high / high-k), the knockout for more than 26 options, and the temperature table.
It needs only public packages: `pydantic`, plus `vllm==0.30.0` on the GPU machine.

```sh
pip install ".[vllm]"                              # on a Linux CUDA machine
wald-serve --model /path/to/Wald-4B --port 8000    # starts vLLM on the weights and serves /v1/systemone
```

Tests (no model, no GPU; a fake vLLM stands in): `pip install ".[test]" && cd tests && pytest -q`.
`tests/test_parity.py` compares prompts and answers byte for byte with the reference implementation the published
numbers were measured with; it is skipped unless that implementation is importable (WALD_PARITY_SRC set).
