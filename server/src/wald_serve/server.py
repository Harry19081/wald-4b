"""`wald-serve`: the TypeSafe `POST /v1/systemone` server for Wald-4B.

One command starts vLLM on the weights (a loopback-only sidecar) and this front-end in front of it:

    wald-serve --model /path/to/Wald-4B --port 8000

or attaches to a vLLM server that is already running:

    wald-serve --vllm http://127.0.0.1:8011 --served wald --temperature /path/to/temperature.json --port 8000

Defaults come from `<model>/serving.json` when present (`effort`, `prompt_format`, `max_model_len`), else the built-in
ones below; command-line flags override both. The response carries TypeSafe's answer keys plus `mode` (A = one pass,
B = after a thought, K = knockout) and `usage`.

Endpoints: POST /v1/systemone (also POST /), GET /health, GET /v1/models.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .engine import Capacity, Client, answer, load_tables, policy
from .prompt import PROMPT_FORMATS

DEFAULTS = {"effort": "medium", "prompt_format": "plain", "max_model_len": 131072}


def read_serving(model_dir) -> dict:
    p = Path(model_dir) / "serving.json" if model_dir else None
    if p and p.is_file():
        return {k: v for k, v in json.loads(p.read_text()).items() if k in DEFAULTS or k == "temperature"}
    return {}


def launch_vllm(model: str, served: str, port: int, max_len: int, mem: float, extra: str) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", served,
           "--host", "127.0.0.1", "--port", str(port), "--max-model-len", str(max_len), "--gpu-memory-utilization",
           str(mem), "--max-num-seqs", "256", "--seed", "0", *shlex.split(extra)]
    print(json.dumps({"launching": cmd}), flush=True)
    proc = subprocess.Popen(cmd, start_new_session=True)

    def stop():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    atexit.register(stop)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(1200):
        if proc.poll() is not None:
            raise SystemExit(f"vLLM exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5):
                return proc
        except OSError:
            time.sleep(2)
    raise SystemExit("vLLM did not become healthy within 40 minutes")


def make_handler(cl: Client, pol: dict, tables: dict, info: dict, workers: int):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                return self.send(200, {"object": "list", "data": [{"id": info["model"], "object": "model"}]})
            self.send(200, {"ok": True, **info})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                p = policy(req["effort"]) if isinstance(req, dict) and req.get("effort") else pol
            except Exception as e:   # noqa: BLE001
                return self.send(400, {"error": f"{type(e).__name__}: {e}"[:400]})
            t0 = time.perf_counter()
            try:
                ans, usage = answer(cl, req, p, tables, workers)
            except Capacity as e:
                return self.send(422, {"error": str(e)[:400]})
            except Exception as e:   # noqa: BLE001
                return self.send(400 if isinstance(e, ValueError) else 500, {"error": f"{type(e).__name__}: {e}"[:400]})
            self.send(200, {"model": info["model"], "answers": ans, "usage": usage, "latency_s": time.perf_counter() - t0})
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wald-serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="weights directory: start vLLM on it (serving.json / temperature.json read from it)")
    src.add_argument("--vllm", help="URL of a running vLLM OpenAI-compatible server")
    ap.add_argument("--served", default="wald", help="vLLM served model name")
    ap.add_argument("--model-name", default="wald-4b", help="name reported in responses")
    ap.add_argument("--effort", default=None, help="none | low | medium | high | high-k<k> (default: serving.json, else medium)")
    ap.add_argument("--prompt-format", choices=PROMPT_FORMATS, default=None, help="default: serving.json, else plain")
    ap.add_argument("--temperature", default=None, help="temperature table (default: <model>/temperature.json)")
    ap.add_argument("--max-model-len", type=int, default=None, help="context limit in tokens (default: serving.json, else 131072)")
    ap.add_argument("--question-workers", type=int, default=8, help="questions of one request read concurrently (1 = serial)")
    ap.add_argument("--vllm-port", type=int, default=8011)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--vllm-args", default="", help="extra `vllm serve` arguments, e.g. \"--quantization fp8\"")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    cfg = {**DEFAULTS, **read_serving(a.model)}
    effort = a.effort or cfg["effort"]
    fmt = a.prompt_format or cfg["prompt_format"]
    max_len = a.max_model_len or int(cfg["max_model_len"])
    pol = policy(effort)
    temps = a.temperature or (str(Path(a.model) / cfg.get("temperature", "temperature.json")) if a.model else None)
    if temps and not Path(temps).is_file():
        raise SystemExit(f"temperature table not found: {temps}")
    tables = load_tables(temps, pol["budget"])

    if a.model:
        launch_vllm(a.model, a.served, a.vllm_port, max_len, a.gpu_memory_utilization, a.vllm_args)
        endpoint = f"http://127.0.0.1:{a.vllm_port}"
    else:
        endpoint = a.vllm
    cl = Client(endpoint, a.served, max_len, prompt_format=fmt)
    info = {"model": a.model_name, "version": __version__, "effort": pol["name"], "gate": pol["gate"],
            "budget": pol["budget"], "think_k": pol["k"], "prompt_format": fmt, "max_model_len": max_len,
            "wide": "knockout", "max_options": 676, "temperature": bool(tables)}
    ThreadingHTTPServer.daemon_threads = True
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(cl, pol, tables, info, a.question_workers))
    print(json.dumps({"serving": info, "port": a.port}), flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
