"""Parity with the reference implementation the Decision Index numbers were measured with.

Skipped unless WALD_PARITY_SRC points at a checkout that provides `eval.systemone_vllm` (with its `kev` and `midtrain`
imports on PYTHONPATH). Checks, on generated requests, that this package builds byte-identical prompts and returns
identical answers against the same (fake) vLLM for every effort policy and both prompt formats.
"""
from __future__ import annotations

import os
import random

import pytest

from fakevllm import FakeVLLM
from wald_serve import engine
from wald_serve.prompt import question_prompt
from wald_serve.wire import SystemOneRequest, to_record

pytestmark = pytest.mark.skipif(not os.environ.get("WALD_PARITY_SRC"), reason="reference implementation not available")

WORDS = "the a order refund card payment late missing charged twice store policy item box user tool call ask".split()


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def gen_request(rng):
    kind = rng.random()
    if kind < 0.25:
        state = ""
    elif kind < 0.6:
        state = text(rng, rng.randint(3, 60))
    else:
        state = {"ticket": text(rng, 20), "meta": {"n": rng.randint(1, 9), "tags": [text(rng, 2), text(rng, 3)]}}
    qs = {}
    for i in range(rng.randint(1, 5)):
        t = rng.choice(["choice", "choice", "noul", "score"])
        if t == "noul":
            qs[f"q{i}"] = {"type": "noul", "instructions": text(rng, 6)}
        elif t == "score":
            qs[f"q{i}"] = {"type": "score", "instructions": text(rng, 5), "criteria": [text(rng, 2) for _ in range(rng.randint(2, 7))]}
        else:
            n = rng.choice([2, 3, 4, 9, 26, 27, 53, 77])
            style = rng.random()
            if style < 0.33:
                crit = {chr(97 + j) if n <= 26 else f"o{j + 1}": text(rng, 3) for j in range(n)}
            elif style < 0.66:
                crit = {f"label_{j}": None for j in range(n)}
            else:
                crit = {f"Option {j}": text(rng, 4) for j in range(n)}
            qs[f"q{i}"] = {"type": "choice", "instructions": text(rng, 7), "criteria": crit}
    return {"state": state, "questions": qs}


@pytest.fixture(scope="module")
def fake():
    f = FakeVLLM()
    yield f
    f.stop()


def test_prompts_identical():
    from eval.systemone_vllm import format_prompt as ref_format
    from kev.api import SystemOneRequest as RefReq, to_record as ref_record
    from midtrain.letter import question_prompt as ref_prompt
    rng = random.Random(7)
    n = 0
    for _ in range(300):
        req = gen_request(rng)
        rec, _ = to_record(SystemOneRequest.model_validate(req))
        rrec, _ = ref_record(RefReq.model_validate(req))
        assert rec["state"] == rrec["state"]
        for rq, rrq in zip(rec["questions"], rrec["questions"]):
            q = {"type": rq["qtype"], "instructions": rq["instr"], "options": rq["options"], "option_texts": rq["options"]}
            if len(q["options"]) > 26:
                continue
            for fmt in ("plain", "repeat_state_plain"):
                assert question_prompt(rec["state"], q, fmt) == ref_format(ref_prompt(rrec["state"], q, "paren", "letter"), fmt)
                n += 1
    assert n > 500


@pytest.mark.parametrize("effort,gate,k", [("none", 0.0, 1), ("low", 0.5, 1), ("medium", 0.7, 1), ("high", 1.01, 1),
                                           ("high-k4", 1.01, 4)])
@pytest.mark.parametrize("fmt", ["plain", "repeat_state_plain"])
def test_answers_identical(fake, effort, gate, k, fmt):
    from eval import systemone_vllm as ref
    table = {"A": {"single": 1.4, "buckets": {"choice|2": {"temperature": 1.1}, "noul|2": {"temperature": 1.3}}},
             "B512": {"single": 1.8, "buckets": {"choice|3-4": {"temperature": 2.2}}}}
    tables = {"A": table["A"], "B": table["B512"], "K": table["A"]}
    rcl = ref.Client(fake.url, "wald", 100_000)
    rcl.wide, rcl.template, rcl.prompt_format, rcl.topk = "knockout", "paren", fmt, 0
    cl = engine.Client(fake.url, "wald", 100_000, prompt_format=fmt)
    rng = random.Random(11)
    for _ in range(25):
        req = gen_request(rng)
        want, wu = ref.answer(rcl, req, gate, 512, tables, k)
        got, gu = engine.answer(cl, req, engine.policy(effort), tables)
        for a in want.values():
            a.pop("one_pass", None)
        assert got == want and gu == wu
