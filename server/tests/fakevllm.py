"""A stand-in for vLLM's OpenAI-compatible server, for tests without a model or a GPU.

Tokenizer: one token per character (so "A" is one token and " A" is two). Completions: the logprob of every requested
token id is a deterministic function of the prompt ids, so a read is reproducible; generation returns a short fixed
thought per seed. A prompt longer than `max_len` answers 400 "maximum context length", as vLLM does.

The same server also answers llama.cpp's llama-server shapes (`/tokenize` with `content`, `/completion` with `n_probs`),
reporting the same logprobs for the printable-ASCII token ids, so both clients see one model.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _lp(prompt, tid):
    h = hashlib.sha256(json.dumps([prompt[-64:], len(prompt), tid]).encode()).digest()
    return -(int.from_bytes(h[:4], "big") % 4000) / 1000.0


class FakeVLLM:
    def __init__(self, max_len=100_000):
        self.max_len = max_len
        self.calls = {"tokenize": 0, "read": 0, "generate": 0}
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self.send(200, {"ok": True})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                if self.path == "/tokenize":
                    fake.calls["tokenize"] += 1
                    return self.send(200, {"tokens": [ord(c) for c in body.get("prompt", body.get("content"))]})
                if self.path == "/completion":
                    return self.llama(body)
                prompt = body["prompt"]
                if len(prompt) + body["max_tokens"] > fake.max_len:
                    return self.send(400, {"message": f"This model's maximum context length is {fake.max_len} tokens."})
                if body["max_tokens"] == 1:
                    fake.calls["read"] += 1
                    top = {f"token_id:{t}": _lp(prompt, t) for t in body["logprob_token_ids"]}
                    return self.send(200, {"choices": [{"logprobs": {"top_logprobs": [top]}}]})
                fake.calls["generate"] += 1
                n = body.get("n", 1)
                texts = [f" thought {body['seed'] % 997} #{i}: the evidence points one way." for i in range(n)]
                return self.send(200, {"choices": [{"text": t} for t in texts], "usage": {"completion_tokens": 9 * n}})

            def llama(self, body):
                prompt = body["prompt"]
                if len(prompt) + body["n_predict"] > fake.max_len:
                    return self.send(400, {"error": {"message": "the request exceeds the available context size"}})
                if body["n_predict"] == 1:
                    fake.calls["read"] += 1
                    top = sorted(({"id": t, "token": chr(t), "logprob": _lp(prompt, t)} for t in range(32, 127)),
                                 key=lambda x: -x["logprob"])[: body["n_probs"]]
                    return self.send(200, {"content": top[0]["token"],
                                           "completion_probabilities": [{**top[0], "top_logprobs": top}]})
                fake.calls["generate"] += 1
                return self.send(200, {"content": f" thought {body['seed'] % 997}: the evidence points one way.",
                                       "tokens_predicted": 9})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
