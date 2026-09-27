"""Reading one request: a letter read per question over a vLLM OpenAI-compatible server, the effort gate, the knockout
for more than 26 options, and the temperature table.

Per question (2..26 options):
  1. one-pass read: the prompt's next-token distribution over the option letters, z_L = log(P("L") + P(" L")),
     softmax over the question's letters (mode `A`);
  2. if the effort policy's gate g > 0 and the one-pass max p < g: generate a thought after `Reasoning:` (at most
     `budget` tokens, temperature 0.6 / top-p 0.95 / top-k 20, stop at `\\nAnswer`, seed = hash of the request and the
     question id), then read the letters again after `\\nAnswer: (` (mode `B`). With k > 1 thoughts the answer is the
     mean of the k post-thought distributions; thought 0 is the k = 1 thought;
  3. the answer is tempered with the checkpoint's temperature table (bucket = question type x option count; table `A`
     for one-pass answers, `B<budget>` for thought answers). Tempering changes confidence, never the argmax.

More than 26 options (knockout, mode `K`, one pass, no thought): the options are split, in their given order, into
ceil(n / 26) near-equal consecutive chunks read with the same prompt; the chunk winners are then read as one final
question; P(option) = P_final(its chunk) x P_chunk(option). This is one rule for every wide question.

Declared capacity limits: a prompt longer than max_model_len is refused with HTTP 422 "maximum context length" (never
truncated); a thought that would not fit falls back to the one-pass answer; at most 26 x 26 = 676 options per question
(the wire format allows 255).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .prompt import CLOSE, CUE, LETTERS, STOP, THINK, question_prompt
from .wire import SystemOneRequest, to_record

BUCKETS = ((2, "2"), (4, "3-4"), (8, "5-8"), (math.inf, "9+"))


class Capacity(ValueError):
    """A declared capacity limit: answered with HTTP 422."""


# --- effort policies --------------------------------------------------------------------------------------------------
# gate: think when the one-pass max p < gate (0 = never, 1.01 = always); budget: max thought tokens; k: thoughts averaged.
POLICIES = {
    "none": {"gate": 0.0, "budget": 512, "k": 1},
    "low": {"gate": 0.5, "budget": 512, "k": 1},
    "medium": {"gate": 0.7, "budget": 512, "k": 1},
    "high": {"gate": 1.01, "budget": 512, "k": 1},
}


def policy(name: str) -> dict:
    """none / low / medium / high, or high-k<k> (always think, mean of k thoughts, 2 <= k <= 8)."""
    if name in POLICIES:
        return dict(POLICIES[name], name=name)
    if name.startswith("high-k") and name[6:].isdigit() and 2 <= int(name[6:]) <= 8:
        return {"gate": 1.01, "budget": 512, "k": int(name[6:]), "name": name}
    raise ValueError(f"unknown effort {name!r}: none, low, medium, high or high-k<2..8>")


# --- temperature ------------------------------------------------------------------------------------------------------
def bucket_key(qtype: str, n: int) -> str:
    return f"{qtype}|" + next(name for upper, name in BUCKETS if n <= upper)


def temperature(table, qtype: str, n: int) -> float:
    if not table:
        return 1.0
    b = (table.get("buckets") or {}).get(bucket_key(qtype, n))
    return float(b["temperature"]) if b else float(table.get("single", 1.0))


def temper(p, t: float):
    if t == 1.0:
        return list(p)
    z = [math.log(max(x, 1e-12)) / t for x in p]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def load_tables(path, budget: int = 512) -> dict:
    """-> {"A": table, "B": table, "K": table}. A plain bucket table serves every mode; a gated table's thought entry is
    B<budget> when present, else B."""
    if not path:
        return {}
    t = json.loads(Path(path).read_text())
    if "buckets" in t or "single" in t:
        return {"A": t, "B": t, "K": t}
    return {"A": t.get("A"), "B": t.get(f"B{budget}") or t.get("B"), "K": t.get("A")}


def hash_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# --- vLLM client (stdlib; thread-safe, no shared mutable state after __init__) ----------------------------------------
class Client:
    def __init__(self, endpoint: str, served: str, max_len: int, prompt_format: str = "plain", timeout: float = 900):
        self.url, self.served, self.max_len, self.timeout = endpoint.rstrip("/"), served, max_len, timeout
        self.prompt_format = prompt_format
        self.letter_ids = []
        for L in LETTERS:
            ids = []
            for v in (L, " " + L):
                t = self.ids(v)
                if len(t) == 1 and t[0] not in ids:
                    ids.append(t[0])
            if not ids:
                raise RuntimeError(f"no single-token encoding of letter {L!r}")
            self.letter_ids.append(ids)
        self.close = self.ids(CLOSE)

    def post(self, path: str, body: dict, retries: int = 4):
        last = None
        for a in range(retries):
            try:
                req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST",
                                             headers={"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read()[:400].decode("utf-8", "replace")
                if e.code in (400, 413, 422):
                    if "maximum context length" in detail or "max_model_len" in detail or "too long" in detail:
                        raise Capacity(f"maximum context length: {detail[:300]}") from None
                    raise ValueError(f"{e.code}: {detail}") from None
                last = RuntimeError(f"{e.code}: {detail}")
            except (urllib.error.URLError, OSError) as e:
                last = e
            time.sleep(2 ** a)
        raise RuntimeError(f"{path}: {last}")

    def ids(self, text: str) -> list[int]:
        return self.post("/tokenize", {"model": self.served, "prompt": text, "add_special_tokens": False})["tokens"]

    def readout(self, ids: list[int], n: int):
        """-> (softmax over the first n letters, total letter mass)."""
        if len(ids) + 1 > self.max_len:
            raise Capacity(f"prompt of {len(ids)} tokens is longer than the maximum context length {self.max_len}")
        want = [t for L in self.letter_ids[:n] for t in L]
        body = {"model": self.served, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
                "logprob_token_ids": want, "return_tokens_as_token_ids": True}
        r = self.post("/v1/completions", body)
        top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        lp = {int(k.split(":", 1)[1]): v for k, v in top.items() if k.startswith("token_id:") and v is not None and v > -9999}
        floor = min(lp.values()) - 2.0 if lp else -30.0
        z = [math.log(sum(math.exp(lp.get(t, floor)) for t in L)) for L in self.letter_ids[:n]]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [x / s for x in e], sum(math.exp(v) for v in z)

    def generate(self, ids: list[int], max_tokens: int, seed: int, n: int = 1):
        """-> (text, completion tokens) for n = 1; (list of texts, total tokens) for n > 1."""
        body = {"model": self.served, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
                "top_k": 20, "seed": seed, "stop": [STOP], "include_stop_str_in_output": False}
        if n > 1:
            body["n"] = n
        r = self.post("/v1/completions", body)
        ntok = (r.get("usage") or {}).get("completion_tokens", 0)
        if n > 1:
            return [c.get("text") or "" for c in r["choices"]], ntok
        c = r["choices"][0]
        return c.get("text") or "", ntok


# --- one question -----------------------------------------------------------------------------------------------------
def chunks(n: int, size: int = 26) -> list[list[int]]:
    k = math.ceil(n / size)
    base, extra = divmod(n, k)
    out, i = [], 0
    for c in range(k):
        m = base + (1 if c < extra else 0)
        out.append(list(range(i, i + m)))
        i += m
    return out


def subset(q: dict, idx: list[int]) -> dict:
    return {**q, "options": [q["options"][i] for i in idx], "option_texts": [q["option_texts"][i] for i in idx]}


def wide_read(n: int, read):
    """Knockout over n > 26 options, given read(option indices) -> probabilities over them."""
    groups = chunks(n)
    within, winners = [], []
    for g in groups:
        p = read(g)
        within.append(p)
        winners.append(g[max(range(len(g)), key=lambda j: p[j])])
    top = read(winners)
    out = [0.0] * n
    for c, g in enumerate(groups):
        for j, i in enumerate(g):
            out[i] = top[c] * within[c][j]
    s = sum(out)
    return [x / s for x in out]


def read_question(cl: Client, state: str, q: dict, pol: dict, seed_text: str, usage: dict):
    """-> (probabilities in the question's option order, mode A | B | K)."""
    n = len(q["options"])
    if n <= len(LETTERS):
        prompt = question_prompt(state, q, cl.prompt_format)
        ida = cl.ids(prompt)
        usage["input_tokens"] += len(ida)
        p, _ = cl.readout(ida, n)
        gate, budget, k = pol["gate"], pol["budget"], pol["k"]
        if gate > 0 and n > 1 and max(p) < gate:
            pre = cl.ids(prompt[: -len(CUE)] + THINK)
            if len(pre) + budget + 64 <= cl.max_len:   # else the one-pass answer stands
                text, ntok = cl.generate(pre, budget, hash_seed(seed_text))
                texts = [text]
                usage["output_tokens"] += ntok
                if k > 1:
                    more, ntok = cl.generate(pre, budget, hash_seed(seed_text + "#k"), n=k - 1)
                    texts += more
                    usage["output_tokens"] += ntok
                reads = []
                for text in texts:
                    body = text.strip()
                    ids = pre + (cl.ids(" " + body) if body else []) + cl.close
                    pb, _ = cl.readout(ids, n)
                    usage["input_tokens"] += len(ids)
                    reads.append(pb)
                return [sum(r[i] for r in reads) / len(reads) for i in range(n)], "B"
        return p, "A"
    if len(chunks(n)) > len(LETTERS):
        raise Capacity(f"{n} options: at most {len(LETTERS) ** 2} options per choice")

    def read(idx):
        ids = cl.ids(question_prompt(state, subset(q, idx), cl.prompt_format))
        usage["input_tokens"] += len(ids)
        return cl.readout(ids, len(idx))[0]
    return wide_read(n, read), "K"


def seed_base(req: dict) -> str:
    """The thought seed covers the TypeSafe fields only (a per-request `effort` override does not change it)."""
    return json.dumps({k: v for k, v in req.items() if k != "effort"}, sort_keys=True)


def answer(cl: Client, req: dict, pol: dict, tables: dict, workers: int = 8):
    """-> (answers in TypeSafe's wire format, usage). Questions are read concurrently (each question's read depends only
    on the state, the question and its seed, so the order of reads does not change an answer)."""
    rec, meta = to_record(SystemOneRequest.model_validate({k: v for k, v in req.items() if k != "effort"}))
    base = seed_base(req)
    jobs = []
    for rq, m in zip(rec["questions"], meta):
        q = {"id": m["id"], "type": rq["qtype"], "instructions": rq["instr"], "options": rq["options"],
             "option_texts": rq["options"]}
        jobs.append((q, m))

    def one(job):
        q, m = job
        usage = {"input_tokens": 0, "output_tokens": 0}
        p, mode = read_question(cl, rec["state"], q, pol, base + m["id"], usage)
        return p, mode, usage

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            results = list(ex.map(one, jobs))
    else:
        results = [one(j) for j in jobs]

    answers, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    for (q, m), (p, mode, u) in zip(jobs, results):
        usage["input_tokens"] += u["input_tokens"]
        usage["output_tokens"] += u["output_tokens"]
        keys = m["keys"]
        p = temper(p, temperature(tables.get(mode), m["type"], len(keys)))
        j = max(range(len(p)), key=lambda i: p[i])
        if m["type"] == "noul":
            answers[m["id"]] = {"type": "noul", "noul": p[keys.index("true")], "mode": mode}
        elif m["type"] == "choice":
            K = len(p)
            answers[m["id"]] = {"type": "choice", "choice": keys[j], "probabilities": dict(zip(keys, p)),
                                "confidence": 1.0 if K == 1 else (p[j] - 1 / K) / (1 - 1 / K), "mode": mode}
        else:
            answers[m["id"]] = {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
                                "probabilities": {str(i): v for i, v in enumerate(p)}, "confidence": p[j], "mode": mode}
    return answers, usage
