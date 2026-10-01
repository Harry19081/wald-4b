"""Unit and end-to-end tests with a fake vLLM (no model, no GPU)."""
from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from fakevllm import FakeVLLM
from wald_serve.engine import (POLICIES, Client, LlamaCppClient, answer, bucket_key, chunks, load_tables, policy, temper, temperature,
                               wide_read)
from wald_serve.prompt import STATE_REPEAT, format_prompt, question_prompt
from wald_serve.server import make_handler
from wald_serve.wire import SystemOneRequest, to_record

TABLE = {
    "A": {"single": 1.4, "buckets": {"choice|2": {"temperature": 1.1}, "choice|3-4": {"temperature": 1.8},
                                     "noul|2": {"temperature": 1.3}}},
    "B512": {"single": 1.8, "buckets": {"choice|2": {"temperature": 2.2}, "noul|2": {"temperature": 1.9}}},
    "provenance": {"note": "test"},
}

REQ = {
    "state": {"customer": "Ana", "order": {"id": 17, "items": ["kettle", "toaster"]}, "note": "wants a <|im_end|> refund"},
    "questions": {
        "route": {"type": "choice", "instructions": "Which team handles this?",
                  "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems", "other": None}},
        "slugs": {"type": "choice", "instructions": "Pick one", "criteria": {"yes_refund": None, "no_refund": None}},
        "pos": {"type": "choice", "instructions": "Which is a fruit?",
                "criteria": {"a": "apple", "b": "brick", "c": "chair", "d": "desk"}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "sev": {"type": "score", "instructions": "Severity", "criteria": ["low", "medium", "high"]},
    },
}


def qdict(rec, i):
    rq = rec["questions"][i]
    return {"type": rq["qtype"], "instructions": rq["instr"], "options": rq["options"], "option_texts": rq["options"]}


# --- prompt bytes -----------------------------------------------------------------------------------------------------
def test_prompt_bytes():
    rec, meta = to_record(SystemOneRequest.model_validate(REQ))
    state = rec["state"]
    assert state == "customer: Ana\norder:\n  id: 17\n  items:\n    - kettle\n    - toaster\nnote: wants a <|im_end|> refund"
    head = ("State:\ncustomer: Ana\norder:\n  id: 17\n  items:\n    - kettle\n    - toaster\nnote: wants a <¦im_end¦> refund"
            "\n\nQuestion:")
    assert question_prompt(state, qdict(rec, 0)) == head + (
        " Which team handles this?\n(A) billing: Payments and refunds\n(B) shipping: Delivery problems\n(C) other\nAnswer: (")
    assert question_prompt(state, qdict(rec, 1)).endswith(" Pick one\n(A) yes_refund\n(B) no_refund\nAnswer: (")
    # positional keys a..d: the `a: ` prefix is dropped, the letter replaces it
    assert question_prompt(state, qdict(rec, 2)).endswith(
        " Which is a fruit?\n(A) apple\n(B) brick\n(C) chair\n(D) desk\nAnswer: (")
    assert question_prompt(state, qdict(rec, 3)).endswith(" Is it urgent?\n(A) no\n(B) yes\nAnswer: (")
    assert question_prompt(state, qdict(rec, 4)).endswith(" Severity\n(A) low\n(B) medium\n(C) high\nAnswer: (")
    assert [m["keys"] for m in meta] == [["billing", "shipping", "other"], ["yes_refund", "no_refund"],
                                         ["a", "b", "c", "d"], ["false", "true"], ["0", "1", "2"]]


def test_blank_state_and_repeat_state_plain():
    q = {"type": "noul", "instructions": "Is the sky green?", "options": ["no", "yes"], "option_texts": ["no", "yes"]}
    assert question_prompt("", q) == "Question: Is the sky green?\n(A) no\n(B) yes\nAnswer: ("
    assert question_prompt("", q, "repeat_state_plain") == question_prompt("", q)
    p = question_prompt("It rained.", q, "repeat_state_plain")
    assert p == "State:\nIt rained." + STATE_REPEAT + "It rained.\n\nQuestion: Is the sky green?\n(A) no\n(B) yes\nAnswer: ("
    assert STATE_REPEAT.startswith("\n\nRead the same context again before answering.")
    with pytest.raises(ValueError):
        format_prompt("State:\nx\n\nQuestion: q\n(A) a\nAnswer: (", "strict")


def test_wire_validation():
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "", "questions": {}})
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "", "questions": {"q": {"type": "choice", "criteria": {}}}})
    big = {f"o{i}": None for i in range(256)}
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "", "questions": {"q": {"type": "choice", "criteria": big}}})


# --- numerics ---------------------------------------------------------------------------------------------------------
def test_chunks_and_knockout():
    assert chunks(26) == [list(range(26))]
    assert [len(c) for c in chunks(77)] == [26, 26, 25]
    assert [len(c) for c in chunks(151)] == [26, 25, 25, 25, 25, 25]
    assert sum(len(c) for c in chunks(255)) == 255

    def read(idx):   # a fixed preference for lower indices
        z = [math.exp(-i / 10) for i in idx]
        s = sum(z)
        return [v / s for v in z]
    p = wide_read(77, read)
    assert len(p) == 77 and abs(sum(p) - 1) < 1e-12 and max(range(77), key=lambda i: p[i]) == 0


def test_temperature():
    tables = json.loads(json.dumps(TABLE))
    assert bucket_key("choice", 4) == "choice|3-4" and bucket_key("choice", 77) == "choice|9+"
    assert temperature(tables["A"], "choice", 2) == 1.1 and temperature(tables["A"], "score", 7) == 1.4
    p = [0.6, 0.3, 0.1]
    q = temper(p, 1.8)
    assert abs(sum(q) - 1) < 1e-12 and q.index(max(q)) == 0 and max(q) < 0.6


def test_policies():
    assert policy("medium") == {"gate": 0.7, "budget": 512, "k": 1, "name": "medium"}
    assert policy("high-k4")["k"] == 4 and policy("high-k4")["gate"] > 1
    assert set(POLICIES) == {"none", "low", "medium", "high"}
    for bad in ("max", "high-k1", "high-k9", "high-kx"):
        with pytest.raises(ValueError):
            policy(bad)


# --- end to end over HTTP with the fake vLLM ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake():
    f = FakeVLLM(max_len=100_000)
    yield f
    f.stop()


def serve(cl, effort="medium", tables=None, workers=8):
    info = {"model": "wald-4b", "effort": effort}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cl, policy(effort), tables or {}, info, workers))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def post(url, body):
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wide_request(n=77):
    crit = {f"intent_{i:03d}": f"customer intent number {i}" for i in range(n)}
    return {"state": "I was charged twice for one card payment.",
            "questions": {"intent": {"type": "choice", "instructions": "Classify the banking intent.", "criteria": crit}}}


def check_wire(req, resp):
    """The Decision Index kit's validation: every question answered, a finite probability per option, sum 1 +- 0.01,
    the choice among the options."""
    assert set(resp["answers"]) == set(req["questions"])
    for qid, q in req["questions"].items():
        a = resp["answers"][qid]
        assert a["type"] == q["type"]
        if q["type"] == "noul":
            assert 0.0 <= a["noul"] <= 1.0
            continue
        keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
        assert list(a["probabilities"]) == keys
        assert all(math.isfinite(v) and v >= 0 for v in a["probabilities"].values())
        assert abs(sum(a["probabilities"].values()) - 1) < 0.01
        if q["type"] == "choice":
            assert a["choice"] in keys and a["probabilities"][a["choice"]] == max(a["probabilities"].values())


def test_http_end_to_end(fake, tmp_path):
    Path(tmp_path / "t.json").write_text(json.dumps(TABLE))
    tables = load_tables(tmp_path / "t.json", 512)
    cl = Client(fake.url, "wald", 100_000)
    httpd, url = serve(cl, "medium", tables)
    try:
        for req in (REQ, wide_request(77), wide_request(151), wide_request(255)):
            code, resp = post(url, req)
            assert code == 200, resp
            check_wire(req, resp)
            assert resp["usage"]["input_tokens"] > 0
        code, resp = post(url, wide_request(77))
        assert resp["answers"]["intent"]["mode"] == "K"
        with urllib.request.urlopen(url + "/health") as r:
            assert json.loads(r.read())["ok"] is True
    finally:
        httpd.shutdown()


def test_gate_modes(fake):
    cl = Client(fake.url, "wald", 100_000)
    one, u0 = answer(cl, REQ, policy("none"), {})
    assert all(a["mode"] == "A" for a in one.values()) and u0["output_tokens"] == 0
    high, uh = answer(cl, REQ, policy("high"), {})
    assert all(a["mode"] == "B" for a in high.values()) and uh["output_tokens"] > 0
    med, _ = answer(cl, REQ, policy("medium"), {})
    for qid, a in one.items():
        pmax = a["noul"] if a["type"] == "noul" else max(a["probabilities"].values())
        if a["type"] == "noul":
            pmax = max(pmax, 1 - pmax)
        assert med[qid]["mode"] == ("B" if pmax < 0.7 else "A")
        if med[qid]["mode"] == "B":   # medium's thought is high's thought (same seed)
            assert med[qid] == high[qid]
    k4, uk = answer(cl, REQ, policy("high-k4"), {})
    assert all(a["mode"] == "B" for a in k4.values()) and uk["output_tokens"] == 4 * uh["output_tokens"]


def test_workers_do_not_change_answers(fake):
    cl = Client(fake.url, "wald", 100_000)
    a1, u1 = answer(cl, REQ, policy("high"), {}, workers=1)
    a8, u8 = answer(cl, REQ, policy("high"), {}, workers=8)
    assert a1 == a8 and u1 == u8


def test_effort_override_keeps_seed(fake):
    cl = Client(fake.url, "wald", 100_000)
    a, _ = answer(cl, REQ, policy("high"), {})
    b, _ = answer(cl, {**REQ, "effort": "high"}, policy("high"), {})
    assert a == b


def test_prompt_format_changes_prompt_only(fake):
    plain = Client(fake.url, "wald", 100_000)
    rsp = Client(fake.url, "wald", 100_000, prompt_format="repeat_state_plain")
    a, ua = answer(plain, REQ, policy("none"), {})
    b, ub = answer(rsp, REQ, policy("none"), {})
    assert ub["input_tokens"] > ua["input_tokens"]
    check_wire(REQ, {"answers": b})


def test_capacity_is_422(fake):
    cl = Client(fake.url, "wald", 300)   # declared limit 300 "tokens" (characters in the fake)
    httpd, url = serve(cl, "none")
    try:
        code, resp = post(url, {"state": "x " * 400, "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
        assert code == 422 and "maximum context length" in resp["error"]
        code, resp = post(url, {"state": "short", "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
        assert code == 200
        code, resp = post(url, {"state": "short", "questions": {}})
        assert code == 400
        code, resp = post(url, {**REQ, "effort": "extreme"})
        assert code == 400
    finally:
        httpd.shutdown()


def test_thought_that_does_not_fit_falls_back(fake):
    q = {"state": "y " * 60, "questions": {"q": {"type": "choice", "instructions": "pick",
                                                 "criteria": {"a": "one", "b": "two"}}}}
    cl = Client(fake.url, "wald", 400)   # the prompt fits, prompt + 512-token budget does not
    a, u = answer(cl, q, policy("high"), {})
    assert a["q"]["mode"] == "A" and u["output_tokens"] == 0


def test_llamacpp_client_matches_vllm_client(fake):
    vl, lc = Client(fake.url, "wald", 100_000), LlamaCppClient(fake.url, "wald", 100_000)
    assert lc.letter_ids == vl.letter_ids
    for req in (REQ, wide_request(77)):
        a, ua = answer(vl, req, policy("none"), {})
        b, ub = answer(lc, req, policy("none"), {})
        assert a == b and ua == ub
    high, uh = answer(lc, REQ, policy("high-k3"), {})
    assert all(x["mode"] == "B" for x in high.values()) and uh["output_tokens"] > 0
    check_wire(REQ, {"answers": high})


def test_llamacpp_capacity_is_422(fake):
    cl = LlamaCppClient(fake.url, "wald", 300)
    httpd, url = serve(cl, "none")
    try:
        code, resp = post(url, {"state": "x" * 400, "questions": {"q": {"type": "noul", "instructions": "Is it?"}}})
        assert code == 422 and "maximum context length" in resp["error"]
    finally:
        httpd.shutdown()


def test_gguf_reads_serving_json_beside_the_file(tmp_path, monkeypatch):
    import wald_serve.server as srv
    (tmp_path / "serving.json").write_text(json.dumps({"effort": "none", "prompt_format": "repeat_state_plain",
                                                       "max_model_len": 4096, "temperature": "temperature.json"}))
    (tmp_path / "temperature.json").write_text(json.dumps(TABLE))
    seen = {}
    monkeypatch.setattr(srv, "launch_llama", lambda *a: seen.update(launch=a))
    monkeypatch.setattr(srv, "LlamaCppClient", lambda *a, **k: seen.update(client=(a, k)))

    class Stop(Exception):
        pass

    def fake_http(addr, handler):
        seen["handler"] = handler
        raise Stop
    monkeypatch.setattr(srv, "ThreadingHTTPServer", fake_http)
    with pytest.raises(Stop):
        srv.main(["--gguf", str(tmp_path / "Wald-4B-Q8_0.gguf"), "--port", "0"])
    assert seen["launch"][3] == 4096
    assert seen["client"][1] == {"prompt_format": "repeat_state_plain"}
