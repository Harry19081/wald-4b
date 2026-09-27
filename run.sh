#!/usr/bin/env bash
# One command, no Docker: create a Python 3.12 environment with vLLM 0.30.0 and wald-serve, then serve the weights on
# 0.0.0.0:${PORT:-8000} (POST /v1/systemone, GET /health). Needs a Linux CUDA machine and `uv`.
#
#   ./run.sh /path/to/Wald-4B                  # declared policy from serving.json (effort medium)
#   EFFORT=high ./run.sh /path/to/Wald-4B      # another policy: none | low | medium | high | high-k<k>
set -euo pipefail
MODEL=${1:?usage: run.sh <weights dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$HERE/.venv}
[ -x "$VENV/bin/python" ] || { uv venv -p 3.12 "$VENV" && VIRTUAL_ENV="$VENV" uv pip install "vllm==0.30.0" "$HERE/server"; }
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
exec "$VENV/bin/wald-serve" --model "$MODEL" --port "${PORT:-8000}" ${EFFORT:+--effort "$EFFORT"} --vllm-args "${VLLM_ARGS:-}"
