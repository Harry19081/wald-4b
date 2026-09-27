"""The plain letter prompt the model reads, one prompt per question. No chat template, no special tokens:

    State:
    {state}

    Question: {instructions}
    (A) {option 1}
    (B) {option 2}
    Answer: (

The answer is read at the last position from the LM head's option-letter logits (engine.Client.readout). When the model
thinks, the final `Answer: (` is replaced by `Reasoning:`, a short thought is generated, and the letters are read again
after `\\nAnswer: (`.

Prompt formats (one general rewrite applied to every question, never chosen per benchmark):
    plain                the layout above
    repeat_state_plain   the state written twice, the second copy introduced by STATE_REPEAT. The sentence is taken
                         verbatim from featherless-ai/simple-jev @ dae340e (Apache-2.0; see NOTICE).
"""
from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CUE = "Answer: ("
THINK = "Reasoning:"
CLOSE = "\nAnswer: ("
STOP = "\nAnswer"
STATE_TAIL = "Question:"

# featherless-ai/simple-jev @ dae340e, hf-server/hf_prompt_policies.py (Apache-2.0), verbatim.
STATE_REPEAT = ("\n\nRead the same context again before answering. This is a repeated copy, not additional events or "
                "independent evidence:\n")
PROMPT_FORMATS = ("plain", "repeat_state_plain")

_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")     # caller text never forms a <|special|> token
_POSITIONAL = re.compile(r"[a-z]|o[0-9]+")          # choice keys that only number the options
_SLUG = re.compile(r"[a-z0-9][a-z0-9_.-]{0,39}")


def plain(text) -> str:
    return _SPECIAL.sub(r"<¦\1¦>", str(text))


def choice_keys(options) -> list[str]:
    """The options themselves when they are all distinct short slugs, else a, b, c, ... (o1, o2, ... past 26)."""
    opts = [str(o) for o in options]
    if opts and len(set(opts)) == len(opts) and all(isinstance(o, str) and _SLUG.fullmatch(o) for o in options):
        return opts
    return [chr(ord("a") + i) if len(opts) <= 26 else f"o{i + 1}" for i in range(len(opts))]


def raw_options(qtype: str, keys, texts) -> list[str]:
    """The option strings after their letter: a choice question whose keys are all positional (a, b, ... / o27) drops
    the `a: ` prefix, since the letter replaces it."""
    if qtype == "choice":
        keys = list(keys)
        if all(_POSITIONAL.fullmatch(k) for k in keys):
            return [t[len(k) + 2:] if t.startswith(f"{k}: ") else t for k, t in zip(keys, texts)]
    return list(texts)


def state_text(state: str) -> str:
    """`State:\\n{state}\\n\\nQuestion:` (just `Question:` for a blank state)."""
    return (f"State:\n{plain(state)}\n\n" if str(state).strip() else "") + STATE_TAIL


def branch_text(q: dict) -> str:
    """The rest of one question's prompt after `Question:`. q = {"type", "instructions", "options", "option_texts"}."""
    texts = [str(o) for o in q["option_texts"]]
    if len(texts) > len(LETTERS):
        raise ValueError(f"{len(texts)} options: one letter read handles at most {len(LETTERS)}")
    keys = choice_keys(q["options"]) if q["type"] == "choice" else None
    out = f" {plain(q['instructions'])}"
    for j, t in enumerate(raw_options(q["type"], keys, texts)):
        out += f"\n({LETTERS[j]}) {plain(t)}"
    return out + "\n" + CUE


def format_prompt(prompt: str, fmt: str) -> str:
    """Apply a prompt format to one question's plain prompt."""
    if fmt == "plain":
        return prompt
    if not prompt.endswith(CUE):
        raise ValueError("prompt does not end with the answer cue")
    if fmt == "repeat_state_plain":
        if prompt.startswith("State:\n") and "\n\nQuestion:" in prompt:
            i = prompt.index("\n\nQuestion:")
            state = prompt[len("State:\n"):i]
            return "State:\n" + state + STATE_REPEAT + state + prompt[i:]
        return prompt   # blank state: nothing to repeat
    raise ValueError(f"unknown prompt format {fmt!r}; one of {PROMPT_FORMATS}")


def question_prompt(state: str, q: dict, fmt: str = "plain") -> str:
    """The one-pass prompt of one question under a rendered state."""
    return format_prompt(state_text(state) + branch_text(q), fmt)
