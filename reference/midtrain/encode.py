"""Record encoding, block-causal mask and row form. Ported from Kev's kev/model.py (Apache-2.0, see NOTICE) and
extended with pause tokens, option echo, rationale (tier-2) branches and plain-text rows
(docs/reasoning-architecture.md, section 1).

Decision record -> one packed sequence; every (question, tier) branch has its own seg and restarts after the state:

    <state> state tokens | <q> instr <opt> o1 </opt> ... <opt> oK </opt>  SUFFIX  | <q> ... | ...

    SUFFIX, tier 0/1           <pause>*N <decide>                                   readout: </opt> of o1..oK
    SUFFIX, tier 0/1 + echo    <pause>*N <opt> k1 </opt> ... <opt> kK </opt> <decide>      echoed </opt>
    SUFFIX, tier 2             <think> r1..rm <decide>                              readout: </opt> of o1..oK
    SUFFIX, tier 2 + echo      <think> r1..rm <opt> k1 </opt> ... <opt> kK </opt> <decide> echoed </opt>

k_j is option j's short key (`echo_key`). Two echo modes (the doc's table, section 1):
  siblings    attention-only bases: the spans share position ids (each starts right after the last pause / rationale
              token), each sees everything before the echo block plus itself (`sib`, enforced by the packed mask and by
              the row and fork forms' 4D masks), <decide> sits after the longest span and sees all of them, so the block
              is order-invariant
  sequential  hybrids (recurrent layers cannot be masked): consecutive positions, the spans see each other in option
              order; the same layout as Kev's echo without option isolation, which fine-tunes the Qwen3.5 checkpoints
A question with a rationale emits its tier-0/1 branch always and a tier-2 branch with probability p_tier2.

seg[i]      : 0 for state tokens, k for the tokens of branch k (k >= 1)
pos[i]      : position ids; every branch restarts right after the state, so branches are interchangeable
sib[i]      : j + 1 on the tokens of echo span j in siblings mode, else 0
decide_idx  : index of each branch's <decide>; opt_idx: index of each option's scored </opt> (echoed when echo is on)
lm_labels   : next-token targets (the model shifts them): text rows everywhere; decision rows only on the rationale
              (r1..rm, and <decide> as the stop target after rm, placed on the first echo token in echo mode); a
              corrupted rationale (or a self-sampled one marked correct: false) gets none. Never on state, instructions,
              options, pauses or echo.
qidx        : index in rec["questions"] of each branch's question
tiers, rat_corrupt, rat_targets : per branch 0/1/2 and "rationale without LM targets"; LM targets in the encoding

Text record -> one causal row with seg all 0 and lm_labels = ids (the model shifts them).
"""
import hashlib
import re
import torch

from .tokens import Delimiters

OPT_NONE, OPT_DECIDE, OPT_PAUSE, OPT_THINK, OPT_ECHO = -1, -2, -3, -4, -5
IGNORE = -100


class ContextOverflow(ValueError):
    """A record does not fit the context (state, branch or packed limit)."""


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, delims: Delimiters, text):
    """Tokenize caller text so it can never produce a delimiter or pause token (option boundaries are unforgeable).
    `<|name|>` is rewritten to `<¦name¦>` (Kev) and any literal delimiter/pause string gets the same treatment."""
    text = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    for name in delims.names.values():
        if name in text:
            text = text.replace(name, "<¦" + name[1:-1] + "¦>")
    return tok(text, add_special_tokens=False).input_ids


def render(v, indent=0):
    """Flatten a JSON state (str | object | array) into the text the model sees, exactly as Kev's serving path does
    (kev.api.render), so a mid-trained base sees the same state text the fine-tune and the server will give it.
    trainer/data wraps ~30 % of hf states as {"document": ...}, {"ticket": ...} or a chat list."""
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


_SLUG = re.compile(r"[a-z0-9][a-z0-9_.-]{0,39}")


def option_texts(q):
    """The option strings exactly as the fine-tune and the server render them (t2m_kev.convert.to_kev_question ->
    kev.api.to_record): choice options that are all distinct short slugs stay as they are, anything else becomes
    `a: text`, `b: text`, ... (`o27: ...` past 26); noul is `no` / `yes` (`no: text` when the options are not literally
    no/yes); score levels are the level texts. Mid-training on the same surface form is what lets the head and the
    delimiter embeddings carry over into the fine-tune.
    A question may carry `option_texts`, the strings already rendered (eval.predictors.MidtrainPredictor passes
    kev.api.to_record's, so Kev-format keys such as `c: ...` or `none_of_these: ...` reach the model unchanged)."""
    if q.get("option_texts") is not None:
        return [str(o) for o in q["option_texts"]]
    opts = [str(o) for o in q["options"]]
    if q["type"] == "choice":
        if len(set(opts)) == len(opts) and all(_SLUG.fullmatch(o) for o in opts):
            return opts
        keys = [chr(ord("a") + i) if len(opts) <= 26 else f"o{i + 1}" for i in range(len(opts))]
        return [f"{k}: {o}" if o else k for k, o in zip(keys, opts)]
    if q["type"] == "noul" and opts != ["no", "yes"]:
        return [f"no: {opts[0]}" if opts[0] else "no", f"yes: {opts[1]}" if opts[1] else "yes"]
    return opts


def echo_key(q, j, text):
    """The short key option j is echoed by (reasoning-architecture.md, token layouts): the slug or letter `option_texts`
    puts before the colon, `no`/`yes` for noul, the level index for score."""
    if q["type"] == "score":
        return str(j)
    if q["type"] == "noul":
        return ("no", "yes")[j]
    return text.split(": ", 1)[0]       # as kev.model.echo_keys falls back to


def rationale_of(q):
    """(text, lm) of a question's rationale, or None. Accepts the trainer-JSONL string form (`rationale` plus an optional
    `rationale_corrupt: true`) and the object form of reasoning-architecture.md section 3 ({"text", "corrupt", "correct",
    ...}). lm is False for a corrupted rationale and for a self-sampled one marked `correct: false`: those train the head
    only. An object carrying `ids` (a decoded rationale, see rationale_ids) counts even when empty (an immediate stop)."""
    r = q.get("rationale")
    if isinstance(r, dict):
        text, corrupt, correct = r.get("text"), r.get("corrupt", q.get("rationale_corrupt", False)), r.get("correct", True)
        if isinstance(r.get("ids"), list):   # decoded token ids (predict, tier 2): used as they are, possibly none
            return (text if isinstance(text, str) else ""), not (bool(corrupt) or correct is False)
    elif isinstance(r, str):
        text, corrupt, correct = r, q.get("rationale_corrupt", False), True
    else:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return text, not (bool(corrupt) or correct is False)


def rationale_ids(tok, delims, q, text):
    """The rationale's token ids: `rationale.ids` when the object form carries them (a rationale decoded by the model at
    tier 2, midtrain.predict: re-tokenizing its text need not give back the decoded ids), else user_tokens(text)."""
    r = q.get("rationale")
    if isinstance(r, dict) and isinstance(r.get("ids"), list):
        return [int(t) for t in r["ids"]]
    return user_tokens(tok, delims, text)


def _draw(key, p):
    """A deterministic Bernoulli(p) draw from a string key (stable across processes, unlike hash())."""
    if p >= 1:
        return True
    if p <= 0:
        return False
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") / 2 ** 64 < p


def echo_mode(hybrid):
    """The echo layout a base supports: sibling spans need a maskable (attention-only) backbone."""
    return "sequential" if hybrid else "siblings"


def decision_kwargs(cfg, hybrid=False):
    """encode_decision keyword arguments for the config's echo / rationale switches; {} when both are off (the pilot
    configs), so their encodings are unchanged."""
    kw = {}
    if (cfg.get("pause") or {}).get("echo_options"):
        kw["echo"] = echo_mode(hybrid)
    r = cfg.get("rationale") or {}
    if r.get("enabled"):
        kw["rationale"] = {"p_tier2": float(r.get("p_tier2", 0.3)), "max_tokens": int(r.get("max_tokens", 512)),
                           "seed": str((cfg.get("train") or {}).get("seed", 0))}
    return kw


def encode_decision(tok, delims: Delimiters, rec, n_pause=0, max_state=1024, max_branch=1024, max_packed=4096, strict=False,
                    echo=False, rationale=None, rng_key=""):
    """Encode one decision record. Returns a list of packed encodings: usually one; several when the branches do not fit
    one packed sequence (each chunk repeats the state). Raises ContextOverflow when a tier-0/1 branch does not fit; a
    tier-2 branch that does not fit (or whose rationale exceeds rationale["max_tokens"]) is dropped and counted.

    echo: False, or repeat each option's key after the pauses / rationale and score the echoed </opt> (option echo,
      arm C2) in mode "siblings" (True) or "sequential" (see the module docstring; `echo_mode(model.hybrid)` picks).
    rationale: None (questions' rationales are ignored) or {"p_tier2", "max_tokens", "seed"}: a question with a
      rationale also emits a tier-2 branch with probability p_tier2, drawn deterministically from (seed, rng_key, the
      question); needs the `think` delimiter role."""
    state_tokens = user_tokens(tok, delims, render(rec["state"]))
    if strict and len(state_tokens) + 1 > max_state:
        raise ContextOverflow(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    S = [delims["state"]] + state_tokens[: max_state - 1]
    q_id, o_id, c_id, d_id, p_id = (delims[r] for r in ("q", "opt", "opt_end", "decide", "pause"))
    if echo not in (False, None, True, "siblings", "sequential"):
        raise ValueError(f"echo must be False, True, 'siblings' or 'sequential', not {echo!r}")
    sibling = echo in (True, "siblings")
    if rationale is not None and "think" not in delims.ids:
        raise ValueError("rationale training needs the `think` delimiter (install_delimiters(tok, {'think': '<think>'}))")
    branches, dropped = [], 0      # (ids, opt, pos offsets, sib, labels, readout offsets, question, its index, tier, corrupt)
    for qi, q in enumerate(rec["questions"]):
        instr = [q_id] + user_tokens(tok, delims, q["instructions"])
        texts = option_texts(q)
        spans = [[o_id] + user_tokens(tok, delims, o) + [c_id] for o in texts]
        prefix = instr + [t for sp in spans for t in sp]
        prefix_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp]
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        echo_spans = [[o_id] + user_tokens(tok, delims, echo_key(q, j, t)) + [c_id] for j, t in enumerate(texts)] if echo else None
        suffixes = [([p_id] * n_pause, [OPT_PAUSE] * n_pause, [IGNORE] * n_pause, 1 if n_pause else 0, False, False)]
        rat = rationale_of(q) if rationale is not None else None
        if rat is not None and _draw(f"{rationale.get('seed', '')}:{rng_key}:{qi}:{q.get('id')}:{q['instructions']}:{rat[0]}", rationale["p_tier2"]):
            r_ids = rationale_ids(tok, delims, q, rat[0])
            if len(r_ids) > int(rationale.get("max_tokens", 512)):
                dropped += 1
            else:
                mid = [delims["think"]] + r_ids
                suffixes.append((mid, [OPT_THINK] * len(mid), [IGNORE] + (r_ids if rat[1] else [IGNORE] * len(r_ids)), 2, rat[1], not rat[1]))
        for mid, mid_opt, mid_lab, tier, lm, corrupt in suffixes:
            br, bopt, blab = prefix + mid, prefix_opt + mid_opt, [IGNORE] * len(prefix) + mid_lab
            boff, bsib = list(range(len(br))), [0] * len(br)
            if echo:
                e0, readout = len(br), []
                for j, sp in enumerate(echo_spans):
                    start = e0 if sibling else len(br)
                    boff += range(start, start + len(sp)); bsib += [j + 1 if sibling else 0] * len(sp)
                    br += sp; bopt += [OPT_ECHO] * len(sp); blab += [IGNORE] * len(sp)
                    readout.append(len(br) - 1)
                d_off = e0 + max(len(sp) for sp in echo_spans) if sibling else len(br)
                if lm:
                    blab[e0] = d_id    # the stop is learned, the echo is not: h[rm] predicts <decide> though <opt> follows
            else:
                readout, d_off = list(ends), len(br)
            br.append(d_id); bopt.append(OPT_DECIDE); boff.append(d_off); bsib.append(0)
            blab.append(d_id if lm and not echo else IGNORE)
            if len(br) > max_branch or len(S) + len(br) > max_packed:
                if tier == 2:
                    dropped += 1
                    continue
                if len(br) > max_branch:
                    raise ContextOverflow(f"branch too long: {len(br)} tokens (limit {max_branch})")
                raise ContextOverflow(f"state + branch exceeds {max_packed} packed tokens")
            branches.append((br, bopt, boff, bsib, blab, readout, q, qi, tier, corrupt))
    out, i = [], 0
    while i < len(branches):
        ids, seg, pos, opt, sib, lab = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S), [0] * len(S), [IGNORE] * len(S)
        decide_idx, opt_idx, targets, qtypes, qids, qidx, tiers, corrupt = [], [], [], [], [], [], [], []
        k = 0
        while i < len(branches) and len(ids) + len(branches[i][0]) <= max_packed:
            br, bopt, boff, bsib, blab, readout, q, qi, tier, bad = branches[i]; k += 1
            base, p0 = len(ids), len(S)
            ids += br; seg += [k] * len(br); pos += [p0 + o for o in boff]; opt += bopt; sib += bsib; lab += blab
            decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in readout])
            targets.append(q["target"]); qtypes.append(q["type"]); qids.append(q.get("id", f"q{qi + 1}")); qidx.append(qi)
            tiers.append(tier); corrupt.append(bad)
            i += 1
        out.append({"kind": "decision", "ids": ids, "seg": seg, "pos": pos, "opt": opt, "sib": sib, "decide_idx": decide_idx, "opt_idx": opt_idx,
                    "targets": targets, "qtypes": qtypes, "qids": qids, "qidx": qidx, "tiers": tiers, "rat_corrupt": corrupt, "lm_labels": lab,
                    "rat_targets": sum(l != IGNORE for l in lab), "n_pause": n_pause, "echo": ("siblings" if sibling else "sequential") if echo else False,
                    "tier2_dropped": dropped if not out else 0, "state_truncated": len(state_tokens) + 1 > max_state})
    return out


def encode_text(tok, delims: Delimiters, text, max_len=2048):
    """One causal row for the plain LM loss. Documents end with EOS so the model learns boundaries; BOS is added when the
    tokenizer has one (Llama-style bases)."""
    ids = user_tokens(tok, delims, text)
    if tok.bos_token_id is not None and getattr(tok, "add_bos_token", False):
        ids = [tok.bos_token_id] + ids
    ids = ids[: max_len - 1]
    if tok.eos_token_id is not None:
        ids = ids + [tok.eos_token_id]
    L = len(ids)
    return {"kind": "text", "ids": ids, "seg": [0] * L, "pos": list(range(L)), "opt": [OPT_NONE] * L, "sib": [0] * L, "decide_idx": [], "opt_idx": [],
            "targets": [], "qtypes": [], "qids": [], "qidx": [], "tiers": [], "rat_corrupt": [], "lm_labels": list(ids), "rat_targets": 0, "n_pause": 0,
            "echo": False, "tier2_dropped": 0, "state_truncated": False}


def branch_mask_batch(segs, device, dtype=torch.float32, length=None, sibs=None):
    """Batched block-causal mask, additive [B,1,L,L], right-padded to the longest sequence (Kev).
    attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]); pads (-1) are masked keys; the diagonal is always kept
    so no row is fully masked. A row whose seg is all zero gets the plain causal mask (text rows).
    sibs (option echo): tokens of two different echo spans (sib > 0, unequal) never see each other."""
    L = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if sibs is not None:
        g = torch.zeros((len(segs), L), dtype=torch.long, device=device)
        for b, sb in enumerate(sibs):
            g[b, : len(sb)] = torch.tensor(sb, device=device)
        allow = allow & ~((g[:, :, None] > 0) & (g[:, None, :] > 0) & (g[:, :, None] != g[:, None, :]))
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def rows_of(enc):
    """Split a packed decision encoding into (state_ids, state_pos, rows); rows[k] = {"ids", "pos", "labels", "sib",
    "decide", "opts"} holds branch k's tokens (state-continuing positions), its LM labels, echo siblings and readout
    offsets within the branch. state + rows[k] as one causal row equals the packed block-causal form for that branch
    on any architecture (Kev); with echo siblings, on attention-only bases, given the sibling mask."""
    seg = enc["seg"]; Ls = seg.count(0)
    labels, sib = enc["lm_labels"], enc.get("sib") or [0] * len(seg)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "labels": labels[start:end], "sib": sib[start:end],
                     "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


def as_rows(enc):
    """The row form of any encoding: a text row is itself; a decision encoding becomes one causal row per branch,
    each = state + branch, with absolute readout indices. Returns list of (ids, pos, lm_labels, decide, opts, sib)."""
    if enc["kind"] == "text":
        return [(enc["ids"], enc["pos"], enc["lm_labels"], None, None, enc.get("sib") or [0] * len(enc["ids"]))]
    S, Sp, rows = rows_of(enc)
    return [(S + r["ids"], Sp + r["pos"], [IGNORE] * len(S) + r["labels"], len(S) + r["decide"], [len(S) + o for o in r["opts"]], [0] * len(S) + r["sib"])
            for r in rows]
