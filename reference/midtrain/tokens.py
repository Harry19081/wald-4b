"""Delimiter and pause tokens.

Every delimiter is exactly one token of the tokenizer. Where the tokenizer already defines a rarely used reserved token
(Qwen: <|fim_prefix|> <|fim_middle|> <|box_start|> <|box_end|> <|fim_suffix|>, the same choice as Kev) it is reused and
its embedding is simply trained along with everything else (full-parameter). Where a tokenizer lacks one (MiniCPM5 has
no box tokens; SmolLM2 has none of them) the missing names are added as special tokens and the embedding matrix grows;
the new rows start at the mean embedding. `<pause>` is always added: no base ships it.
"""
from dataclasses import dataclass, field

ROLES = ("state", "q", "opt", "opt_end", "decide")
# the Qwen names, reused verbatim where they exist and added under the same name where they do not, so one checkpoint
# format serves every base
PREFERRED = {"state": "<|fim_prefix|>", "q": "<|fim_middle|>", "opt": "<|box_start|>", "opt_end": "<|box_end|>", "decide": "<|fim_suffix|>"}
PAUSE = "<pause>"
# optional roles, installed only when named (so a run without them keeps its vocabulary): `think` opens a tier-2
# rationale (docs/reasoning-architecture.md); Qwen ships <think> as one token, other bases get it added
OPTIONAL_ROLES = ("think",)
THINK = "<think>"


@dataclass
class Delimiters:
    ids: dict[str, int]                       # role -> token id (roles above plus "pause")
    names: dict[str, str]                     # role -> token string
    added: list[str] = field(default_factory=list)   # tokens that were added to the tokenizer (need fresh embeddings)

    def __getitem__(self, role):
        return self.ids[role]

    @property
    def all_ids(self):
        return set(self.ids.values())

    def to_dict(self):
        return {"ids": self.ids, "names": self.names, "added": self.added}


def _is_single_token(tok, text):
    ids = tok(text, add_special_tokens=False).input_ids
    return len(ids) == 1 and tok.convert_tokens_to_ids(text) == ids[0] and ids[0] != tok.unk_token_id


def install_delimiters(tok, names=None):
    """Resolve (and add where missing) the delimiter and pause tokens on a tokenizer, plus any optional role (`think`)
    that `names` gives. Returns Delimiters. The model's
    embeddings must be resized afterwards when `added` is non-empty (see model.MidtrainModel)."""
    names = {**PREFERRED, **(names or {})}
    names["pause"] = names.get("pause", PAUSE)
    roles = (*ROLES, "pause", *(r for r in OPTIONAL_ROLES if names.get(r)))
    existing = set(tok.get_vocab())
    missing = [names[r] for r in roles if names[r] not in existing]
    if missing:
        tok.add_tokens(missing, special_tokens=True)
    ids = {r: tok.convert_tokens_to_ids(names[r]) for r in roles}
    for r, t in names.items():
        if r not in ids:
            continue
        if ids[r] is None or ids[r] == tok.unk_token_id:
            raise ValueError(f"delimiter {r}={t!r} is not in the vocabulary after adding it")
        if not _is_single_token(tok, t):
            raise ValueError(f"delimiter {r}={t!r} does not tokenize to exactly one token")
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"delimiter tokens collide: {ids}")
    return Delimiters(ids=ids, names={r: names[r] for r in ids}, added=missing)


def pad_id(tok):
    """Right-padding id (never attended to)."""
    if tok.pad_token_id is not None:
        return tok.pad_token_id
    if tok.eos_token_id is not None:
        return tok.eos_token_id
    return 0
