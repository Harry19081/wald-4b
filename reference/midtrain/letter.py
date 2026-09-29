"""Letter layout: decision records as the plain prompt the base reads (model.readout letter / mix, docs/letter-readout.md).

The pointer layout (encode.encode_decision) wraps state, question and options in delimiter tokens the base never saw and
reads a head trained from scratch. The letter layout keeps the base's own input and output: every question is the plain
prompt an untuned base is scored with, and the answer is read from the LM head's option-letter logits at the prompt's
last token, so at step 0 a letter model scores exactly what the base scores zero-shot on the same prompt.

    layout "paren" (default, decider's style)        layout "rawlogit" (= midtrain.rawprompt, byte for byte)

    State:                                           State:
    {state}                                          {state}

    Question: {instructions}                         Question: {instructions}
    (A) {option 1}                                   A. {option 1}
    (B) {option 2}                                   B. {option 2}
    Answer: (                                        Answer:

The state block is left out when the state is blank; option strings are rawprompt.raw_options' (the rendered option
texts, a positional `a: ` prefix dropped). readout: z_j = log sum_{v in ids(L_j)} exp(W_v . h) at the last token, ids(L) =
the single-token "L" and " L" (rawprompt.letter_token_ids); the softmax over the question's options cancels log Z.
noul "yesno" (model.letter_noul): a noul question whose options are plain no / yes is asked as
`Question: {instructions}\\nAnswer (yes or no):` and read at the no / yes word tokens (yesno_token_ids) instead.

Encoding: the same dict as encode_decision (so every isolation form, packing, the batcher, the decision anchor and the
readout hooks take it unchanged): the state part is `State:\\n{state}\\n\\nQuestion:` (the tokens every question of the
record shares; with the tokenizer's special prefix, e.g. BOS), branch k is the rest of question k's prompt, positions
restart after the state. Each branch is the joint tokenization of the full prompt minus the state part whenever the
state part is a token prefix of it (the usual case; so state + branch == prompt_ids(tok, prompt) exactly), else the
branch's own tokenization (`split_mismatch` counts it). No delimiter tokens, no pauses, no LM labels. decide_idx: the
prompt's last token; opt_idx: the last token of each option line (the mix readout's pointer term reads them; a yes/no
branch has no option lines and points both at the last token, where the pointer term is a constant that cancels).
"""
from .encode import IGNORE, OPT_DECIDE, OPT_NONE, ContextOverflow, option_texts, render
from .rawprompt import LETTERS, choice_keys, plain, raw_options

LAYOUTS = ("paren", "rawlogit")
# Not yet a layout here: "semif_chat" (JevK5 / Jobe's prompt; rawprompt.semif_prompt, read by eval's RawLogitPredictor
# with --raw-layout semif_chat, docs/hard-tier-leaders.md technique 1). To train on it: state_text becomes the chat
# prefix up to the evidence (`<|im_start|>system\n{SEMIF_SYSTEM}<|im_end|>\n<|im_start|>user\n{"evidence": <state JSON>,`
# -- evidence comes first in their JSON, so questions still share the state part), branch_text the rest of the JSON
# (`"criterion": ..., "options": [...]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`) with option j's span
# on its "description" value; decide_idx the prompt's last token; the readout the bare letters A-P only (no " L"
# variant: rawprompt.semif_letter_ids), noul asked true-first; tokens with add_special_tokens=False and no `<|name|>`
# rewrite (the template's own specials are real). The state is the record's state as JSON, not encode.render's text.
NOUL_READOUTS = ("letter", "yesno")
_CUE = {"paren": "Answer: (", "rawlogit": "Answer:"}
YESNO_CUE = "Answer (yes or no):"
STATE_TAIL = "Question:"


def option_line(layout, j, text):
    return f"({LETTERS[j]}) {plain(text)}" if layout == "paren" else f"{LETTERS[j]}. {plain(text)}"


def is_yesno(q, noul):
    """Whether question q is read at the no / yes words (model.letter_noul yesno, plain no/yes options only)."""
    return noul == "yesno" and q["type"] == "noul" and option_texts(q) == ["no", "yes"]


def state_text(state):
    """The shared state part: `State:\\n{state}\\n\\nQuestion:` (just `Question:` for a blank state)."""
    s = render(state)
    return (f"State:\n{plain(s)}\n\n" if str(s).strip() else "") + STATE_TAIL


def branch_text(q, layout="paren", noul="letter"):
    """(text after the state part, [(start, end) char span of option j's text within it]); ValueError past 26 options."""
    if is_yesno(q, noul):
        return f" {plain(q['instructions'])}\n{YESNO_CUE}", []
    texts = option_texts(q)
    if len(texts) > len(LETTERS):
        raise ValueError(f"{len(texts)} options: the letter readout has {len(LETTERS)} letters")
    keys = choice_keys(q["options"]) if q["type"] == "choice" else None
    out, spans = f" {plain(q['instructions'])}", []
    for j, t in enumerate(raw_options(q["type"], keys, texts)):
        line = option_line(layout, j, t)
        out += "\n"
        spans.append((len(out), len(out) + len(line)))
        out += line
    return out + "\n" + _CUE[layout], spans


def question_prompt(state, q, layout="paren", noul="letter"):
    """The full prompt of one question (the text a base is scored on; tests and docs)."""
    return state_text(state) + branch_text(q, layout, noul)[0]


def yesno_token_ids(tok):
    """[[no ids], [yes ids]]: the single-token encodings of no / No / " no" / " No" (and yes), deduplicated."""
    out = []
    for w in ("no", "yes"):
        ids = []
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize()):
            enc = tok(v, add_special_tokens=False).input_ids
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        if not ids:
            raise ValueError(f"tokenizer has no single-token encoding of {w!r}")
        out.append(ids)
    return out


def _tokens(tok, text, special):
    enc = tok(text, add_special_tokens=special, return_offsets_mapping=bool(getattr(tok, "is_fast", False)))
    return list(enc.input_ids), (list(enc.offset_mapping) if "offset_mapping" in enc else None)


def _ends(offsets, spans, shift, n, last):
    """Token index (within the branch) of each option span's last token: the last token starting before the span's end."""
    if offsets is None:
        return [last] * len(spans)
    out = []
    for a, b in spans:
        idx = [t for t in range(n) if offsets[t][1] > offsets[t][0] and offsets[t][0] - shift < b]
        out.append(idx[-1] if idx else last)
    return out


def encode_letter(tok, rec, layout="paren", noul="letter", max_state=1024, max_branch=1024, max_packed=4096, strict=False,
                  drop_many=False, stats=None, **_):
    """Encode one decision record in the letter layout (module docstring); returns a list of packed encodings like
    encode_decision. A question past 26 options raises ContextOverflow, or is left out with drop_many (training; `stats`
    counts it under "letter_dropped"); a record left without questions raises ContextOverflow. A state over max_state
    tokens raises with strict, else its text is cut (the state part keeps its `\\n\\nQuestion:` tail). Extra keyword
    arguments (n_pause, echo, rationale) are ignored: the layout has none of them."""
    if layout not in LAYOUTS:
        raise ValueError(f"letter layout must be one of {LAYOUTS}, not {layout!r}")
    if noul not in NOUL_READOUTS:
        raise ValueError(f"letter_noul must be one of {NOUL_READOUTS}, not {noul!r}")
    stats = stats if stats is not None else {}
    st = state_text(rec["state"])
    S, _ = _tokens(tok, st, True)
    truncated = len(S) > max_state
    if truncated:
        if strict:
            raise ContextOverflow(f"state exceeds {max_state} tokens: {len(S)}")
        tail, _ = _tokens(tok, "\n\n" + STATE_TAIL, False)
        head, _ = _tokens(tok, st[: -len(STATE_TAIL)].rstrip("\n"), True)
        S = head[: max(1, max_state - len(tail))] + tail
    branches = []   # (ids, opt, readout offsets, question index, yes/no)
    for qi, q in enumerate(rec["questions"]):
        try:
            bt, spans = branch_text(q, layout, noul)
        except ValueError as err:
            if not drop_many:
                raise ContextOverflow(str(err)) from None
            stats["letter_dropped"] = stats.get("letter_dropped", 0) + 1
            continue
        ids, offs, shift = None, None, 0
        if not truncated:
            full, foffs = _tokens(tok, st + bt, True)
            if full[: len(S)] == S:
                ids, offs, shift = full[len(S):], (foffs[len(S):] if foffs else None), len(st)
        if ids is None:
            stats["split_mismatch"] = stats.get("split_mismatch", 0) + int(not truncated)
            ids, offs = _tokens(tok, bt, False)
        if not ids:
            raise ContextOverflow("empty question prompt")
        n = len(ids)
        ends = _ends(offs, spans, shift, n, n - 1)
        opt = [OPT_NONE] * n
        if offs is not None:
            for t in range(n):
                a = offs[t][0] - shift
                for j, (s0, s1) in enumerate(spans):
                    if s0 <= a < s1:
                        opt[t] = j
        opt[-1] = OPT_DECIDE
        if n > max_branch:
            raise ContextOverflow(f"branch too long: {n} tokens (limit {max_branch})")
        if len(S) + n > max_packed:
            raise ContextOverflow(f"state + branch exceeds {max_packed} packed tokens")
        k = len(q["options"]) if not spans else len(spans)
        readout = ends if spans else [n - 1] * k
        branches.append((ids, opt, readout, q, qi, bool(not spans)))
    if not branches:
        raise ContextOverflow("no question the letter readout can score")
    out, i = [], 0
    while i < len(branches):
        ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
        decide_idx, opt_idx, targets, qtypes, qids, qidx, yesno = [], [], [], [], [], [], []
        k = 0
        while i < len(branches) and len(ids) + len(branches[i][0]) <= max_packed:
            br, bopt, readout, q, qi, yn = branches[i]; k += 1
            base = len(ids)
            ids += br; seg += [k] * len(br); pos += [len(S) + o for o in range(len(br))]; opt += bopt
            decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in readout])
            targets.append(q.get("target", 0)); qtypes.append(q["type"]); qids.append(q.get("id", f"q{qi + 1}")); qidx.append(qi)
            yesno.append(yn)
            i += 1
        L = len(ids)
        out.append({"kind": "decision", "ids": ids, "seg": seg, "pos": pos, "opt": opt, "sib": [0] * L, "decide_idx": decide_idx,
                    "opt_idx": opt_idx, "targets": targets, "qtypes": qtypes, "qids": qids, "qidx": qidx, "tiers": [0] * k,
                    "rat_corrupt": [False] * k, "lm_labels": [IGNORE] * L, "rat_targets": 0, "n_pause": 0, "echo": False,
                    "tier2_dropped": 0, "state_truncated": truncated, "layout": layout, "yesno": yesno})
    return out


def attach_letter_raw_ids(encs, max_tokens):
    """Reference prompts for the decision anchor / KL-only rows under the letter layout: branch k's prompt is the
    encoding's own state + branch ids, exactly the tokens the model reads, so KL(p_ref || p_model) compares the two
    models on one prompt (rawprompt.attach_raw_ids renders the rawlogit prompt instead). Sets raw_ids / raw_skip as
    attach_raw_ids does: "length" past max_tokens, "yesno" for a no / yes-word branch (its readout is not the letters)."""
    for e in encs:
        Ls = e["seg"].count(0)
        S, start = e["ids"][:Ls], Ls
        ids, skip = [], []
        for d, yn in zip(e["decide_idx"], e.get("yesno") or [False] * len(e["decide_idx"])):
            end = d + 1
            p = S + e["ids"][start:end]
            start = end
            if yn:
                ids.append(None); skip.append("yesno")
            elif len(p) > int(max_tokens):
                ids.append(None); skip.append("length")
            else:
                ids.append(p); skip.append(None)
        e["raw_ids"], e["raw_skip"] = ids, skip
    return encs


def letter_kwargs(cfg):
    """encode_letter keyword arguments for a run config (model.readout letter / mix), or None for the pointer layout."""
    m = cfg.get("model") or {}
    if m.get("readout", "pointer") == "pointer":
        return None
    return {"layout": m.get("letter_layout", "paren"), "noul": m.get("letter_noul", "letter")}
