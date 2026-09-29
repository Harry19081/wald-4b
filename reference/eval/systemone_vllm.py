"""A concurrent POST /v1/systemone front-end over a vLLM server, for our Base-lineage letter models (015A0-f3 one-pass,
015D0-f4 gated think) in TypeSafe's wire format (the Decision Index kit's `http` engine and JevBench both read it).

Per question the bytes are eval.think_paren's (kev.api.to_record -> midtrain.letter.question_prompt, paren layout):

    State:\\n{state}\\n\\nQuestion: {instructions}\\n(A) {option 1}\\n(B) {option 2}\\nAnswer: (      <- one-pass read

the letters A..Z read at the next position as z_L = log(P("L") + P(" L")), softmax over the question's options. With
--gate g > 0 a question whose one-pass max p < g is re-read after a short thought (think_paren's think layout:
`Reasoning: {thought}\\nAnswer: (`, ≤ --budget tokens, temperature 0.6 / top-p 0.95 / top-k 20, seed = hash of the
request and question id, as think_paren serve); --gate 0.5 / --budget 256 is eval.effort's `low`, 0.7 / 512 its `medium`. Answers are tempered with the --temperature bucket table (the `A` / `B` tables of a gated
table, or one plain table for both).

Differences from `think_paren serve` (which stays as it is):
  - no global lock: requests are answered concurrently (vLLM batches them); the client is stateless stdlib urllib;
  - answers carry the TypeSafe wire keys (`type`, `choice` + `probabilities` + `confidence`, or `noul`), as
    kev.api.to_answers writes them, plus `mode` (A / B / K);
  - all 26 letters are read (think_paren's client reads A..P, so 17-26-option questions came back short);
  - more than 26 options ("wide" questions, e.g. BANKING77's 77 intents): --wide knockout reads the options in
    ceil(n / 26) near-equal consecutive chunks with the same prompt, then one final question over the chunk winners;
    P(option) = P_final(its chunk) x P_chunk(option); one-pass only (no thought). --wide refuse answers 422
    "at most 26 options per choice" instead. --topk K (> 0, <= 26) adds a third round: the K options with the highest
    knockout probability, in their original order, are re-read as one K-way question and the top K's knockout mass is
    spread by that read (mode `T`; the other options keep their knockout probability). Default 0 = plain knockout;
  - a prompt longer than --max-model-len answers 422 with "maximum context length" (the kit records it as unsupported;
    nothing is truncated); a thought that would not fit falls back to the one-pass read;
  - --think-k k (test-time scaling): a question that thinks samples k thoughts and answers with the mean of the k
    post-thought letter distributions (before tempering). Thought 0 is the k = 1 thought (same seed), thoughts 1..k-1
    come from one n = k-1 request seeded from the same text + "#k"; so k = 1 answers exactly as before;
  - --return-raw adds `raw` to every answer: the untempered one-pass distribution (`A`), each thought's untempered
    post-thought distribution (`B`, in sample order) and the mode, so one always-think read (--gate 1.01) can be
    re-scored offline as one-pass, any gate, k = 1 or the mean of the first j thoughts (trainer/eval/scripts/di_policies.py);
  - --prompt-format (default `plain`: the bytes above, unchanged) applies one general rewrite to every question's prompt,
    never keyed to a benchmark; adapted from featherless-ai/simple-jev @ dae340e (Apache-2.0,
    hf-server/hf_prompt_policies.py; the STRICT / FULL_EXAMPLES / STATE_REPEAT / INPUT_REPEAT strings are copied
    verbatim). Our base-lineage models read a plain letter layout, not a chat template, so the policies' system text
    becomes a preamble before `State:` and there is no `[thinking]` prefill / JSON answer / nine-bin Noul:
      repeat_state          STRICT + FULL_EXAMPLES preamble; the state written twice, joined by STATE_REPEAT
      repeat_state_plain    the state written twice (STATE_REPEAT), no preamble (the repetition alone)
      strict_mix_repeat2    STRICT preamble; the whole input (state + question + options) twice, joined by INPUT_REPEAT

    python -m eval.systemone_vllm --vllm http://127.0.0.1:8011 --served 015A0-f3 --gate 0 --port 8100 \\
        --temperature results/v2/015A0-f3-cal/temperature.json

Needs trainer/kev and trainer/midtrain importable (kev.api, midtrain.letter).
"""
import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CUE = "Answer: ("
THINK = "Reasoning:"
CLOSE = "\nAnswer: ("
STOP = "\nAnswer"
BUCKETS = ((2, "2"), (4, "3-4"), (8, "5-8"), (math.inf, "9+"))   # t2m_kev.temperature.BUCKETS


class Capacity(ValueError):
    """A declared capacity limit: answered 422 with a marker the kit's http engine maps to `unsupported`."""


# --- temperature (serve_gated's table format) ---------------------------------------------------------------------------
def bucket_key(qtype, n):
    return f"{qtype}|" + next(name for upper, name in BUCKETS if n <= upper)


def temperature(table, qtype, n):
    if not table: return 1.0
    b = (table.get("buckets") or {}).get(bucket_key(qtype, n))
    return float(b["temperature"]) if b else float(table.get("single", 1.0))


def temper(p, t):
    if t == 1.0: return list(p)
    z = [math.log(max(x, 1e-12)) / t for x in p]
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]


def load_tables(path, budget=512):
    """-> {"A": table, "B": table, "K": table}; a plain bucket table serves every mode; a gated table's think entry is
    B<budget> when fitted for this budget, else B."""
    if not path: return {}
    t = json.loads(Path(path).read_text())
    if "buckets" in t or "single" in t: return {"A": t, "B": t, "K": t}
    return {"A": t.get("A"), "B": t.get(f"B{budget}") or t.get("B"), "K": t.get("A")}


# --- simple-jev prompt formats (Apache-2.0, featherless-ai/simple-jev @ dae340e, hf-server/hf_prompt_policies.py) ------
SJ_STRICT = 'Use the question and rubric as the decision rule. For truth or probability, assess the exact proposition, including its conditions: relevance is not truth, lack of mention is not falsity, and a plausible inference is not an explicit fact. For numerical probability, count or derive the favorable outcomes and divide by the total; evaluate the requested event rather than its complement. For ordered levels, choose the most specific level justified by the evidence, without escalating beyond what its definition requires.'
SJ_FULL_EXAMPLES = 'Independent worked examples below are not facts about the actual state. Transfer only the reasoning rules, not their entities, numbers, conclusions or answers.\nA general rule requires P and Q. A more specific applicable exception requires only P for certified repairs. The case is a certified repair; P is satisfied and Q is not. The exception replaces the general prerequisite list, so the action is permitted.\nA 72-hour window starts March 3 at 06:00 and ends March 6 at 06:00. An event exactly at the endpoint satisfies "by the deadline" but not "before the deadline". Adding a full extra calendar day would be incorrect.\nA policy requires a supervisor when total exposure is at most 50 and a director above 50. Existing exposure is 45 and the new commitment is 10, so total exposure is 55 and the director is required. Testing only the new commitment would apply the wrong quantity.\nItem K maps to unit B; unit B maps to zone 4; zone 4 maps to contact P. An override substitutes contact Q only on holidays. Today is not a holiday. Follow all mappings, then test the override condition: the operative contact is P.\nCause X has prior probability 0.2 and triggers a signal with probability 0.8. Cause Y has prior probability 0.8 and triggers it with probability 0.1. Given the signal, the probability of X is (0.2*0.8)/(0.2*0.8+0.8*0.1)=2/3, not 0.8 and not 0.2.\nLevel 0 means no evidence, level 1 requires condition P, and level 2 requires P and Q. If P holds but Q does not, the matching level is 1. Select the defined category rather than averaging the two nearby categories.\nA requested function must return the maximum of any nonempty numeric list, including negative values. A proposed implementation starts best=0 and only replaces best when a value is larger. It works on positive examples but returns 0 for [-5,-2], whose correct maximum is -2. The implementation does not meet the full requirement.\nFor the actual task, use its own evidence, definitions, exceptions and output labels. Return only the required answer; do not reproduce these explanations.'
SJ_STATE_REPEAT = '\n\nRead the same context again before answering. This is a repeated copy, not additional events or independent evidence:\n'
SJ_INPUT_REPEAT = '\n\nRead the same input again before answering:\n'
PROMPT_FORMATS = ("plain", "repeat_state", "repeat_state_plain", "strict_mix_repeat2")


def format_prompt(prompt, fmt):
    """One question's plain letter prompt (`State:\\n{state}\\n\\nQuestion: ... Answer: (`) rewritten by --prompt-format."""
    if fmt == "plain": return prompt
    if not prompt.endswith(CUE): raise ValueError("prompt does not end with the answer cue")
    if fmt in ("repeat_state", "repeat_state_plain"):
        pre = SJ_STRICT + "\n\n" + SJ_FULL_EXAMPLES + "\n\n" if fmt == "repeat_state" else ""
        if prompt.startswith("State:\n") and "\n\nQuestion:" in prompt:
            i = prompt.index("\n\nQuestion:")
            state = prompt[len("State:\n"):i]
            return pre + "State:\n" + state + SJ_STATE_REPEAT + state + prompt[i:]
        return pre + prompt   # blank state: nothing to repeat
    if fmt == "strict_mix_repeat2":
        body = prompt[: -len(CUE)].rstrip("\n")
        return SJ_STRICT + "\n\n" + body + SJ_INPUT_REPEAT + body + "\n" + CUE
    raise ValueError(f"unknown prompt format {fmt!r}")


def hash_seed(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# --- vLLM client (stdlib, thread-safe: no shared mutable state after __init__) --------------------------------------------
class Client:
    def __init__(self, endpoint, served, max_len, timeout=900):
        self.url, self.served, self.max_len, self.timeout = endpoint.rstrip("/"), served, max_len, timeout
        self.letter_ids = []
        for L in LETTERS:
            ids = []
            for v in (L, " " + L):
                t = self.ids(v)
                if len(t) == 1 and t[0] not in ids: ids.append(t[0])
            if not ids: raise RuntimeError(f"no single-token encoding of letter {L!r}")
            self.letter_ids.append(ids)
        self.close = self.ids(CLOSE)

    def post(self, path, body, retries=4):
        last = None
        for a in range(retries):
            try:
                req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST", headers={"content-type": "application/json"})
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

    def ids(self, text, special=False):
        return self.post("/tokenize", {"model": self.served, "prompt": text, "add_special_tokens": special})["tokens"]

    def readout(self, ids, n):
        if len(ids) + 1 > self.max_len:
            raise Capacity(f"prompt of {len(ids)} tokens is longer than the maximum context length {self.max_len}")
        want = [t for L in self.letter_ids[:n] for t in L]
        body = {"model": self.served, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "logprobs": 20, "logprob_token_ids": want,
                "return_tokens_as_token_ids": True}
        r = self.post("/v1/completions", body)
        top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        lp = {int(k.split(":", 1)[1]): v for k, v in top.items() if k.startswith("token_id:") and v is not None and v > -9999}
        floor = min(lp.values()) - 2.0 if lp else -30.0
        z = [math.log(sum(math.exp(lp.get(t, floor)) for t in L)) for L in self.letter_ids[:n]]
        m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
        return [x / s for x in e], sum(math.exp(v) for v in z)

    def generate(self, ids, max_tokens, seed, n=1):
        """-> (text, completion tokens) for n = 1 (as before); a list of texts and the total tokens for n > 1."""
        body = {"model": self.served, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "seed": seed, "stop": [STOP], "include_stop_str_in_output": False}
        if n > 1: body["n"] = n
        r = self.post("/v1/completions", body)
        ntok = (r.get("usage") or {}).get("completion_tokens", 0)
        if n > 1: return [c.get("text") or "" for c in r["choices"]], ntok
        c = r["choices"][0]
        return c.get("text") or "", ntok


# --- one question ---------------------------------------------------------------------------------------------------------
def chunks(n, size=26):
    k = math.ceil(n / size); base, extra = divmod(n, k)
    out, i = [], 0
    for c in range(k):
        m = base + (1 if c < extra else 0); out.append(list(range(i, i + m))); i += m
    return out


def prompt_ids(cl, state, q):
    """Token ids of one question's one-pass prompt: `paren` = midtrain.letter's plain letter layout (our Base-lineage
    models), `chat` = eval.vllm_letter's chat template (system + user message through the model's own chat template,
    add_generation_prompt, enable_thinking False; the chat-start models, e.g. 013A0-f1)."""
    if cl.template == "chat":
        from eval.vllm_letter import DEFAULT_CHAT_KWARGS, chat_messages, user_message
        st, instr, opts = q["chat"]
        body = {"model": cl.served, "messages": chat_messages(user_message(st, instr, opts)), "add_generation_prompt": True,
                "add_special_tokens": False, "chat_template_kwargs": DEFAULT_CHAT_KWARGS}
        return cl.post("/tokenize", body)["tokens"]
    from midtrain.letter import question_prompt
    if cl.template == "rawlogit":
        # Match the pretrained-base RawLogitPredictor's A./B. ... Answer: read,
        # including its tokenizer's default special prefix when present.
        return cl.ids(question_prompt(state, q, "rawlogit", "letter"), special=True)
    return cl.ids(format_prompt(question_prompt(state, q, "paren", "letter"), getattr(cl, "prompt_format", "plain")))


def subset(q, idx):
    sub = {**q, "options": [q["options"][i] for i in idx], "option_texts": [q["option_texts"][i] for i in idx]}
    if "chat" in q: sub["chat"] = (q["chat"][0], q["chat"][1], [q["chat"][2][i] for i in idx])
    return sub


def read_question(cl, state, q, gate, budget, seed_text, usage, k=1, raw=None):
    """-> (probabilities in the question's option order, mode, the one-pass probabilities when it thought else None).
    k > 1: the mean of k post-thought distributions; `raw` (a dict) receives the untempered A and per-thought B reads."""
    from midtrain.letter import question_prompt
    n = len(q["options"])
    if n <= len(LETTERS):
        ida = prompt_ids(cl, state, q); usage["input_tokens"] += len(ida)
        p, _ = cl.readout(ida, n)
        if raw is not None: raw["A"] = p
        if gate > 0 and n > 1 and max(p) < gate and cl.template == "paren":
            prompt = format_prompt(question_prompt(state, q, "paren", "letter"), getattr(cl, "prompt_format", "plain"))
            if not prompt.endswith(CUE): raise ValueError("one-pass prompt does not end with the answer cue")
            pre = cl.ids(prompt[: -len(CUE)] + THINK)
            if len(pre) + budget + 64 <= cl.max_len:   # as think_paren serve: else the one-pass answer stands
                text, ntok = cl.generate(pre, budget, hash_seed(seed_text))
                texts = [text]; usage["output_tokens"] += ntok
                if k > 1:
                    more, ntok = cl.generate(pre, budget, hash_seed(seed_text + "#k"), n=k - 1)
                    texts += more; usage["output_tokens"] += ntok
                reads = []
                for text in texts:
                    body = text.strip()
                    ids = pre + (cl.ids(" " + body) if body else []) + cl.close
                    pb, _ = cl.readout(ids, n); usage["input_tokens"] += len(ids); reads.append(pb)
                if raw is not None: raw["B"] = reads
                pb = [sum(r[i] for r in reads) / len(reads) for i in range(n)]
                return pb, "B", p
        return p, "A", None
    if cl.wide != "knockout":
        raise Capacity(f"{n} options: at most {len(LETTERS)} options per choice")
    if len(chunks(n)) > len(LETTERS):
        raise Capacity(f"{n} options: at most {len(LETTERS) ** 2} options per choice")

    def read(idx):
        ids = prompt_ids(cl, state, subset(q, idx)); usage["input_tokens"] += len(ids)
        return cl.readout(ids, len(idx))[0]
    return wide_read(n, read, int(getattr(cl, "topk", 0) or 0))


def wide_read(n, read, topk=0):
    """More than 26 options, given `read(option indices) -> probabilities over them` (one letter read of that subset):
    knockout (consecutive chunks, then a final over the chunk winners; P(option) = P_final(chunk) x P_chunk(option)),
    then with topk > 0 one K-way re-read of the K options with the highest knockout probability (original order; ties
    to the lower index), whose knockout mass it redistributes. -> (probabilities, mode "K" | "T", knockout
    probabilities when topk ran else None). Shared by this reader and the vertical harness's HF parity read."""
    groups = chunks(n)
    within, winners = [], []
    for g in groups:
        p = read(g)
        within.append(p); winners.append(g[max(range(len(g)), key=lambda j: p[j])])
    top = read(winners)
    out = [0.0] * n
    for c, g in enumerate(groups):
        for j, i in enumerate(g): out[i] = top[c] * within[c][j]
    s = sum(out)
    out = [x / s for x in out]
    k = min(int(topk or 0), len(LETTERS), n)
    if k < 2:
        return out, "K", None
    idx = sorted(sorted(range(n), key=lambda i: (-out[i], i))[:k])
    pk = read(idx)
    mass = sum(out[i] for i in idx)
    res = list(out)
    for j, i in enumerate(idx): res[i] = mass * pk[j]
    s = sum(res)
    return [x / s for x in res], "T", out


def answer(cl, req, gate, budget, tables, k=1, return_raw=False):
    from kev.api import SystemOneRequest, to_record
    rec, meta = to_record(SystemOneRequest.model_validate(req))
    answers, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    seed_base = json.dumps(req, sort_keys=True)   # think_paren serve's seed: hash of the request + question id
    chat = None
    if cl.template == "chat":   # = eval.vllm_letter.question_parts (kev.api rendering, positional `a: ` prefixes dropped),
        from midtrain.rawprompt import raw_options   # inlined: its import chain (eval.predictors -> kev.data) needs `datasets`
        chat = {}
        for rq, m in zip(rec["questions"], meta):
            src = req["questions"][m["id"]]
            chat[m["id"]] = (rec["state"], rq["instr"], raw_options(m["type"], list(src["criteria"]) if m["type"] == "choice" else None, rq["options"]))
    for rq, m in zip(rec["questions"], meta):
        q = {"id": m["id"], "type": rq["qtype"], "instructions": rq["instr"], "options": rq["options"], "option_texts": rq["options"]}
        if chat is not None: q["chat"] = chat[m["id"]]
        raw = {} if return_raw else None
        p, mode, pa = read_question(cl, rec["state"], q, gate, budget, seed_base + m["id"], usage, k, raw)
        keys = m["keys"]
        p = temper(p, temperature(tables.get(mode), m["type"], len(keys)))
        one_pass = None
        if pa is not None and mode == "B":   # the one-pass read this thought replaced, tempered as mode A (so one read yields both policies)
            pa = temper(pa, temperature(tables.get("A"), m["type"], len(keys)))
            one_pass = {"probabilities": dict(zip(keys, pa))}
        j = max(range(len(p)), key=lambda i: p[i])
        if m["type"] == "noul":
            answers[m["id"]] = {"type": "noul", "noul": p[keys.index("true")], "mode": mode}
        elif m["type"] == "choice":
            K = len(p)
            answers[m["id"]] = {"type": "choice", "choice": keys[j], "probabilities": dict(zip(keys, p)),
                                "confidence": 1.0 if K == 1 else (p[j] - 1 / K) / (1 - 1 / K), "mode": mode}
        else:
            answers[m["id"]] = {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "probabilities": {str(i): v for i, v in enumerate(p)},
                                "confidence": p[j], "mode": mode}
        if one_pass is not None:
            answers[m["id"]]["one_pass"] = one_pass
        if raw is not None:
            answers[m["id"]]["raw"] = {"mode": mode, "keys": keys, "A": raw.get("A"), "B": raw.get("B")}
    return answers, usage


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm", required=True); ap.add_argument("--served", required=True)
    ap.add_argument("--gate", type=float, default=0.0, help="think when the one-pass max p < gate (0: one-pass only)")
    ap.add_argument("--budget", type=int, default=512); ap.add_argument("--temperature", default=None)
    ap.add_argument("--wide", choices=("knockout", "refuse"), default="knockout")
    ap.add_argument("--topk", type=int, default=0, help="> 26 options: re-read the knockout's top K (<= 26) as one K-way question (0: off)")
    ap.add_argument("--template", choices=("paren", "chat", "rawlogit"), default="paren", help="paren: midtrain.letter layout; chat: eval.vllm_letter's chat prompt (no thinking); rawlogit: pretrained-base A./B. plain read")
    ap.add_argument("--max-model-len", type=int, default=16384); ap.add_argument("--model-name", default=None)
    ap.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="plain", help="general prompt rewrite adapted from simple-jev (module docstring); plain = unchanged bytes")
    ap.add_argument("--think-k", type=int, default=1, help="thoughts per thinking question; the answer is their mean post-thought distribution")
    ap.add_argument("--return-raw", action="store_true", help="add untempered one-pass / per-thought reads to every answer (offline policy re-scoring)")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args(argv)
    cl = Client(a.vllm, a.served, a.max_model_len); cl.wide = a.wide; cl.template = a.template; cl.prompt_format = a.prompt_format
    if a.prompt_format != "plain" and a.template != "paren": raise SystemExit("--prompt-format needs --template paren")
    cl.topk = a.topk   # its own statement: on the line above it only ran when the SystemExit was raised (never)
    tables = load_tables(a.temperature, a.budget)
    name = a.model_name or a.served

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args): pass

        def send(self, code, obj):
            b = json.dumps(obj).encode(); self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self.send(200, {"ok": True, "model": name, "gate": a.gate, "budget": a.budget, "wide": a.wide, "topk": a.topk, "max_model_len": a.max_model_len,
                            "think_k": a.think_k, "return_raw": a.return_raw, "prompt_format": a.prompt_format})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            t0 = time.perf_counter()
            try:
                ans, usage = answer(cl, req, a.gate, a.budget, tables, a.think_k, a.return_raw)
            except Capacity as e:
                return self.send(422, {"error": str(e)[:400]})
            except Exception as e:   # noqa: BLE001
                return self.send(400 if isinstance(e, ValueError) else 500, {"error": f"{type(e).__name__}: {e}"[:400]})
            self.send(200, {"model": name, "answers": ans, "usage": usage, "latency_s": time.perf_counter() - t0})

    ThreadingHTTPServer.daemon_threads = True
    print(json.dumps({"serving": name, "gate": a.gate, "budget": a.budget, "wide": a.wide, "port": a.port}), flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
