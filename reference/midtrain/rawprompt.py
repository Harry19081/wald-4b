"""The plain lettered prompt an untuned base is read through as a direct-logit classifier (JevBench's raw-base rows).

One module for both readers, so the text is the same byte for byte:
  eval.predictors.RawLogitPredictor   scores Kev-format records (it renders them with kev.api.to_record, then uses
                                      raw_options / raw_prompt / prompt_ids / letter_token_ids from here)
  midtrain decision anchor            (train.decision_anchor_weight) renders traj2model training records with
                                      record_prompts, which gives the same text as RawLogitPredictor on
                                      t2m_kev.convert.to_kev_request(record) (eval/tests/test_eval_rawlogit.py checks it)

Template (RAW_PROMPT_VERSION):

    State:
    {state}                         (the rendered state; the block is left out when it is blank)

    Question: {instructions}
    A. {option 1}
    B. {option 2}
    ...
    Answer:

`<|name|>` in caller text becomes `<¦name¦>` (Kev's rule). No chat template, no trailing space after "Answer:"; the ids
are tokenizer(prompt, add_special_tokens=True). Readout: z_L = log(P("L") + P(" L")) over the full next-token
distribution, p = softmax over the present options' z. No torch or kev import here: eval imports it at module load.

A second layout, SEMIF_CHAT (semif_prompt, "semif_chat/1"), is JevK5's / Jobe's prompt byte for byte (SemIf's protocol,
TheoLeeCJ/SemIf, MIT): a system instruction, the question as one JSON object {evidence, criterion, options: [{letter,
description}]} in the user turn, the Qwen chat template with thinking off, answer letters A-P. See the section below.
"""
import json
import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RAW_PROMPT_VERSION = "rawlogit/1"
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")     # Kev's rule (midtrain.encode.user_tokens): caller text never forms <|special|>
POSITIONAL = re.compile(r"[a-z]|o[0-9]+")            # choice keys that only number the options (kev.api / encode.option_texts)
_SLUG = re.compile(r"[a-z0-9][a-z0-9_.-]{0,39}")     # t2m_kev.convert.SLUG / encode._SLUG


class TooManyOptions(ValueError):
    """More options than letters: the letter readout cannot score the question."""


def plain(text):
    return _SPECIAL.sub(r"<¦\1¦>", str(text))


def raw_options(qtype, keys, texts):
    """The option strings the base sees after its letter: the rendered option texts (kev.api.to_record's, the strings
    midtrain.encode.option_texts gives for a traj2model question), except that a choice question whose keys are all
    positional (a, b, c, ... / o27) drops the `a: ` prefix, since the letter replaces it."""
    if qtype == "choice":
        keys = list(keys)
        if all(POSITIONAL.fullmatch(k) for k in keys):
            return [t[len(k) + 2:] if t.startswith(f"{k}: ") else t for k, t in zip(keys, texts)]
    return list(texts)


def raw_prompt(state, instructions, options):
    """The plain prompt (module docstring). `state` and `instructions` are already rendered text."""
    if len(options) > len(LETTERS):
        raise TooManyOptions(f"{len(options)} options: the letter readout has {len(LETTERS)} letters")
    head = f"State:\n{plain(state)}\n\n" if str(state).strip() else ""
    lines = [f"{LETTERS[j]}. {plain(o)}" for j, o in enumerate(options)]
    return head + f"Question: {plain(instructions)}\n" + "\n".join(lines) + "\nAnswer:"


def prompt_ids(tok, prompt):
    """The token ids RawLogitPredictor feeds the base for a prompt."""
    return tok(prompt, add_special_tokens=True).input_ids


def letter_token_ids(tok, letters=LETTERS):
    """{letter: [token ids]}: the single-token encodings of "A" and " A" (both, deduplicated; a variant that is not one
    token is left out). A letter with neither is an error: the readout would have nothing to read."""
    out = {}
    for L in letters:
        ids = []
        for v in (L, " " + L):
            enc = tok(v, add_special_tokens=False).input_ids
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        if not ids:
            raise ValueError(f"tokenizer has no single-token encoding of {L!r} or {' ' + L!r}")
        out[L] = ids
    return out


def choice_keys(options):
    """A traj2model choice question's Kev keys (t2m_kev.convert.option_keys): the options themselves when they are all
    distinct short slugs, else a, b, c, ... (o1, o2, ... past 26)."""
    opts = [str(o) for o in options]
    if opts and len(set(opts)) == len(opts) and all(isinstance(o, str) and _SLUG.fullmatch(o) for o in options):
        return opts
    return [chr(ord("a") + i) if len(opts) <= 26 else f"o{i + 1}" for i in range(len(opts))]


def question_prompt(state_text, q):
    """The raw prompt of one traj2model question under an already rendered state; TooManyOptions past 26 options."""
    from .encode import option_texts   # torch-importing module; only the training side calls this
    texts = option_texts(q)
    keys = choice_keys(q["options"]) if q["type"] == "choice" else None
    return raw_prompt(state_text, str(q["instructions"]), raw_options(q["type"], keys, texts))


def record_prompts(rec):
    """traj2model decision record -> [prompt or None] per question, in question order: the text RawLogitPredictor scores
    for t2m_kev.convert.to_kev_request(rec). None where the question has more options than letters."""
    from .encode import render
    state = render(rec["state"])
    out = []
    for q in rec["questions"]:
        try:
            out.append(question_prompt(state, q))
        except TooManyOptions:
            out.append(None)
    return out


def attach_raw_ids(encs, rec, tok, max_tokens):
    """Decision anchor, data side (the prefetch thread): give each decision encoding of `rec` two per-branch lists,
    `raw_ids` (the prompt ids under `tok`, the reference model's tokenizer, or None) and `raw_skip` (None when the
    branch is anchored; "tier2" for a rationale branch, which repeats its question and is not a candidate; "length" when
    the prompt exceeds max_tokens; "options" past 26 options). Each question is rendered and tokenized once."""
    prompts, cache = None, {}
    for e in encs:
        ids, skip = [], []
        for qi, tier in zip(e["qidx"], e["tiers"]):
            if tier == 2:
                ids.append(None); skip.append("tier2")
                continue
            if qi not in cache:
                if prompts is None:
                    prompts = record_prompts(rec)
                p = prompts[qi]
                if p is None:
                    cache[qi] = (None, "options")
                else:
                    x = prompt_ids(tok, p)
                    cache[qi] = (x, None) if len(x) <= int(max_tokens) else (None, "length")
            ids.append(cache[qi][0]); skip.append(cache[qi][1])
        e["raw_ids"], e["raw_skip"] = ids, skip
    return encs


# --- semif_chat: JevK5 / Jobe's prompt (docs/hard-tier-leaders.md technique 1) ----------------------------------------
# Reproduced from, byte for byte (trainer/eval/tests/test_eval_semif_chat.py checks it against prompts rendered by their
# code at these commits, trainer/eval/tests/fixtures/semif_chat_prompts.json):
#   allebee/jevk5 @7b97499 jevk5/prompt.py:13 (LETTERS), 21-24 (SYSTEM), 28-32 (CHAT_TEMPLATE, "verified against
#     apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)"), 35-50 (messages / prompt_text),
#     53-65 (decision_options)
#   MantisShrimpdev/jobe @0382f20 src/jobe/prompt.py:20-23, 95-109, 142-166; src/jobe/server.py:99-117 (options_for),
#     199-201 (Decision(evidence=state, criterion=instructions)); src/jobe/slots.py:83-128 (letter ids resolved in context)
# Jobe's server additionally describes index-only choice options from a jev-browser page state (server.py:159-175,
# describe_indices); it only fires on choice criteria with EMPTY descriptions and a state carrying page.elements, which
# no JevBench / transfer / sealed-like item has, so it is left out (JevK5 does not have it either).
# Unlike the plain layout: no `<|name|>` rewrite (their JSON carries caller text as is), the state is the request's JSON
# value (not kev.api.render's text), noul reads A = true, B = false, and at most 16 options (letters A-P).

SEMIF_PROMPT_VERSION = "semif_chat/1"
SEMIF_LETTERS = "ABCDEFGHIJKLMNOP"
SEMIF_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
                "Respond with only its uppercase letter, with no explanation or reasoning.")
SEMIF_CHAT_TEMPLATE = ("<|im_start|>system\n{system}<|im_end|>\n"
                       "<|im_start|>user\n{user}<|im_end|>\n"
                       "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def semif_options(qtype, criteria):
    """[(option id, description)] of a Kev/TypeSafe question (type, criteria as sent on the wire), in the order the letters
    are given: noul true, false (the criteria text, else "The proposition is <id>."); choice the criteria keys (the value,
    else the key); score the level indices. Descriptions are "<id>: <text>" (JevK5 decision_options, Jobe options_for)."""
    if qtype == "noul":
        c = criteria or {}
        pairs = [(k, c.get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif qtype == "choice":
        c = dict.fromkeys(criteria) if isinstance(criteria, list) else (criteria or {})
        pairs = [(str(k), v or str(k)) for k, v in c.items()]
    elif qtype == "score":
        pairs = [(str(i), level) for i, level in enumerate(criteria or [])]
    else:
        raise ValueError(f"unknown question type {qtype!r}")
    return [(k, f"{k}: {d}") for k, d in pairs]


def semif_messages(state, criterion, descriptions):
    """The two chat messages (system, user JSON) for one question; `state` is the request's state value (any JSON)."""
    if len(descriptions) > len(SEMIF_LETTERS):
        raise TooManyOptions(f"{len(descriptions)} options: the semif_chat readout has {len(SEMIF_LETTERS)} letters")
    payload = {"evidence": state, "criterion": criterion,
               "options": [{"letter": SEMIF_LETTERS[i], "description": d} for i, d in enumerate(descriptions)]}
    return [{"role": "system", "content": SEMIF_SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def semif_prompt(state, criterion, descriptions):
    """The full semif_chat prompt text, chat template included and ending after `<think>\n\n</think>\n\n`, where the answer
    letter goes. Tokenized with add_special_tokens=False (the template carries its own special tokens)."""
    system, user = semif_messages(state, criterion, descriptions)
    return SEMIF_CHAT_TEMPLATE.format(system=system["content"], user=user["content"])


def semif_question_prompt(state, question):
    """(prompt, [option id in letter order]) for one Kev/TypeSafe question dict (type, instructions, criteria)."""
    opts = semif_options(question["type"], question.get("criteria"))
    return semif_prompt(state, question.get("instructions", ""), [d for _, d in opts]), [k for k, _ in opts]


def semif_letter_ids(tok, prompt, ids, count):
    """The token id of each of the first `count` answer letters, resolved in context as Jobe does (slots.py:83-128):
    encode(prompt + L)[-1], checking that the letter is one token and does not re-tokenize the prompt's tail."""
    out = []
    for L in SEMIF_LETTERS[:count]:
        merged = tok(prompt + L, add_special_tokens=False).input_ids
        if len(merged) != len(ids) + 1 or merged[:-1] != ids:
            raise ValueError(f"answer letter {L!r} is not one clean token after the semif_chat prompt")
        out.append(merged[-1])
    if len(set(out)) != len(out):
        raise ValueError("answer letters collide in context")
    return out
