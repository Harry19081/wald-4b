"""Render figures/*.svg from figures/data.json (written by tools/export_figure_data.py). Standard library only:

    python tools/make_figures.py

Every SVG carries light and dark colors (prefers-color-scheme), a <title> tooltip on each mark and direct labels, so
identity never depends on color alone. This script only draws the numbers in data.json.
"""
from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "figures/data.json").read_text())
OUT = ROOT / "figures"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
AREAS = [("knowledge", "Knowledge"), ("language", "Language"), ("retrieval", "Retrieval"), ("tools", "Tools"), ("arts", "Arts")]

STYLE = """
<style>
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --grid:#e6e5e1; --line:#c9c8c2;
        --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#8b6fd6; --laya:#c2369b; --clm:#9a7b2f; --dot:#b9b8b2; --hl:#e8f0fb; }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#8f8e86; --grid:#2e2e2b; --line:#4a4a45;
          --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9d86e8; --laya:#e05cb8; --clm:#c9a24a; --dot:#5c5b56; --hl:#1d2a3b; }
  }
  text { font-family: FONT; fill: var(--ink); }
  .t2 { fill: var(--ink2); } .mu { fill: var(--muted); }
  .h1 { font-size: 16px; font-weight: 600; } .sm { font-size: 11px; } .xs { font-size: 10px; } .b { font-weight: 600; }
  .grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--line); stroke-width: 1; }
</style>""".replace("FONT", FONT)


def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-labelledby="t d">\n<title id="t">{escape(title)}</title>\n<desc id="d">{escape(desc)}</desc>\n'
            f'{STYLE}\n<rect width="{w}" height="{h}" fill="var(--surface)"/>\n' + "\n".join(body) + "\n</svg>\n")


def text(x, y, s, cls="sm", anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(str(s))}</text>'


FAINT = ("--laya", "--clm")   # weak external baselines: shown, but pale so they do not draw the eye


def op(var):
    return ' fill-opacity="0.38"' if var in FAINT else ""


def legend(body, x, y, items, step=190):
    for name, var in items:
        body.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="2" fill="var({var})"{op(var)}/>')
        body.append(text(x + 18, y, name, "sm mu" if var in FAINT else "sm"))
        x += step


# --- 1. Decision Index vs model size ----------------------------------------------------------------------------------
def di_vs_size():
    W, H, L, T, PW, PH = 1000, 520, 70, 90, 640, 360
    xmin, xmax, ymin, ymax = math.log10(1.5), math.log10(45), 20, 60
    sx = lambda b: L + PW * (math.log10(b) - xmin) / (xmax - xmin)
    sy = lambda v: T + PH - PH * (v - ymin) / (ymax - ymin)
    body = [text(24, 30, "Decision Index 0.2.1 vs model size: a 4B model among 12–35B entries", "h1"),
            text(24, 50, "Grey: every board entry with a parameter count in its base model name (official full-suite scores). "
                 "Blue: Wald-4B v1.0 on our 6,948-request stratified sample", "xs t2"),
            text(24, 63, "(Jev scores 57.19 on that sample vs 57.89 on the board, so the sample reads about 0.7 low). "
                 "Hollow = reported from a run whose result file is not committed yet.", "xs t2")]
    for v in range(ymin, ymax + 1, 10):
        body.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{L + PW}" y2="{sy(v):.1f}" class="grid"/>')
        body.append(text(L - 8, sy(v) + 4, v, "xs mu", "end"))
    for b in (2, 4, 8, 12, 27, 35):
        body.append(f'<line x1="{sx(b):.1f}" y1="{T}" x2="{sx(b):.1f}" y2="{T + PH}" class="grid"/>')
        body.append(text(sx(b), T + PH + 16, f"{b}B", "xs mu", "middle"))
    body.append(text(L + PW / 2, T + PH + 34, "parameters of the base model (log scale; MoE = total)", "xs mu", "middle"))
    body.append(text(L - 50, T - 10, "index", "xs mu"))
    jev = D["board"]["jev"]["index"]
    body.append(f'<g><title>Jev (hosted API, size undisclosed): {jev}</title><line x1="{L}" y1="{sy(jev):.1f}" x2="{L + PW}" '
                f'y2="{sy(jev):.1f}" stroke="var(--s2)" stroke-width="1.5" stroke-dasharray="6 4"/></g>')
    body.append(text(L + 6, sy(jev) - 5, f"Jev {jev} (hosted API, size undisclosed)", "xs"))
    named = {e["name"] for e in D["board"]["entries"][:5]} | {"Decider 4B", "Decider 35B-A3B", "Hopper", "JevK5"}
    labels = []
    for e in D["board"]["entries"]:
        if not e["size_b"]:
            continue
        x, y = sx(e["size_b"]) + (sum(map(ord, e["name"])) % 7 - 3), sy(e["index"])
        body.append(f'<g><title>#{e["rank"]} {escape(e["name"])} ({escape(e["base_model"] or "")}): {e["index"]}</title>'
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--dot)"/></g>')
        if e["name"] in named:
            labels.append([y, x, f"#{e['rank']} {e['name']} {e['index']}"])
    labels.sort()
    for i in range(1, len(labels)):   # keep labels in the same column at least 12 px apart
        if abs(labels[i][1] - labels[i - 1][1]) < 80:
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 12)
    for y, x, s_ in labels:
        body.append(text(x + 8, y + 3, s_, "xs t2"))
    for i, w in enumerate(D["wald_di"]):
        x, y = sx(4) + 14, sy(w["index"])
        filled = w["status"] == "committed"
        body.append(f'<g><title>{escape(w["label"])}: {w["index"]} ({w["status"]})</title><circle cx="{x:.1f}" cy="{y:.1f}" r="6" '
                    f'fill="{"var(--s1)" if filled else "var(--surface)"}" stroke="var(--s1)" stroke-width="2.5"/></g>')
        body.append(text(x + 10, y + 4, f"{w['label']} {w['index']}", "sm b" if i == 0 else "xs"))
    body.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + PH}" class="axis"/>')
    return svg(W, H, body, "Decision Index vs model size", "Wald-4B against every sized board entry; Jev as a dashed line.")


# --- 2. Decision Index areas --------------------------------------------------------------------------------------------
def di_areas():
    w = D["wald_di"][0]
    rows = [("Jev (board)", D["board"]["jev"]["areas"], "--s2")]
    if w.get("areas"):
        rows.insert(0, (w["label"] + ("" if w["status"] == "committed" else " (reported)"), w["areas"], "--s1"))
    for e, var in zip(D["board_4_5"], ("--s3", "--s4")):
        rows.append((f"#{e['rank']} {e['name']} (board)", e["areas"], var))
    W, L, PW, GH, BAR = 900, 110, 600, 78, 13
    T = 146
    H = T + len(AREAS) * GH + 40
    sx = lambda v: L + PW * v
    body = [text(24, 30, "Decision Index 0.2.1 by area (chance-corrected skill, 0–1)", "h1"),
            text(24, 50, "Wald-4B v1.0 (4B) vs Jev and the board's #4 / #5 entries (27B). Board rows are official full-suite "
                 "numbers; the Wald row is our stratified-sample read." if w.get("areas") else
                 "Wald-4B v1.0 areas: pending its committed read. Board rows are official full-suite numbers.", "xs t2")]
    for j, r in enumerate(rows):
        body.append(f'<rect x="24" y="{66 + j * 16}" width="12" height="12" rx="2" fill="var({r[2]})"/>')
        body.append(text(42, 76 + j * 16, r[0]))
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        body.append(f'<line x1="{sx(v):.1f}" y1="{T - 6}" x2="{sx(v):.1f}" y2="{H - 30}" class="grid"/>')
        body.append(text(sx(v), H - 16, f"{v:.1f}", "xs mu", "middle"))
    for i, (aid, aname) in enumerate(AREAS):
        y = T + i * GH
        body.append(text(L - 10, y + 30, aname, "sm", "end"))
        for j, (name, areas, var) in enumerate(rows):
            v = areas[aid]
            yy = y + j * (BAR + 3)
            body.append(f'<g><title>{escape(name)} · {aname}: {v:.3f}</title><rect x="{L}" y="{yy}" width="{max(sx(v) - L, 1):.1f}" '
                        f'height="{BAR}" rx="2" fill="var({var})"/></g>')
            body.append(text(sx(v) + 4, yy + 10, f"{v:.2f}", "xs"))
    body.append(f'<line x1="{L}" y1="{T - 6}" x2="{L}" y2="{H - 30}" class="axis"/>')
    return svg(W, H, body, "Decision Index by area", "Five area skills for Wald-4B, Jev and the board's #4 and #5.")


# --- 3. verticals --------------------------------------------------------------------------------------------------------
def verticals():
    rows = sorted(D["verticals"], key=lambda r: -(r["ours_lora"] - r["jev"]))
    L, PW, GH, BAR = 230, 440, 80, 10
    W, T = L + PW + 330, 132
    H = T + len(rows) * GH + 50
    sx = lambda v: L + PW * v / 100
    wins = sum(bool(r["beats_jev"]) for r in rows)
    lo_c, hi_c = min(r["train_usd"] for r in rows), max(r["train_usd"] for r in rows)
    body = [text(24, 30, f"Quick LoRA (< $2, < 2 GPU-h per task) on Wald-4B beats Jev on {wins} of {len(rows)} tasks", "h1"),
            text(24, 50, f"Same fixed test items and byte-identical requests for every system. Ours = one LoRA on the task's "
                 f"labels, ${lo_c:.2f}–${hi_c:.2f} of GPU time each; the others are zero-shot. Jev cannot be fine-tuned.", "xs t2")]
    series = [("ours_lora", "Wald-4B v0.9 + LoRA (< $2)", "--s1"), ("jev", "Jev (API, zero-shot)", "--s2"),
              ("wald_0shot", "Wald-4B v0.9 zero-shot", "--s3"), ("raw_qwen_0shot", "raw Qwen3.5-4B-Base zero-shot", "--dot"),
              ("laya_0shot", "Laya 421M (zero-shot, Jev's wire request)", "--laya"),
              ("clm_0shot", "CLM-8B (zero-shot, deployment verified)", "--clm")]
    legend(body, 24, 76, [(n, v) for _, n, v in series[:3]], step=330)
    legend(body, 24, 94, [(n, v) for _, n, v in series[3:]], step=330)
    missing = [n.split(" (")[0] for k, n, _ in series if all(r[k] is None for r in rows)]
    if missing:
        body.append(text(24, 112, f"Not shown (no committed read yet): {', '.join(missing)}. A task without a bar for a system "
                         "has no read of it.", "xs mu"))
    for v in range(0, 101, 20):
        body.append(f'<line x1="{sx(v):.1f}" y1="{T - 8}" x2="{sx(v):.1f}" y2="{H - 40}" class="grid"/>')
        body.append(text(sx(v), H - 26, v, "xs mu", "middle"))
    body.append(text(L + PW / 2, H - 10, "task metric (%)", "xs mu", "middle"))
    body.append(text(W - 270, T - 14, "ours − Jev [95 % CI] · LoRA cost", "xs t2"))
    for i, r in enumerate(rows):
        y = T + i * GH
        body.append(text(L - 10, y + 16, r["task"], "sm", "end"))
        body.append(text(L - 10, y + 30, f"{r['metric']} · n = {r['n_test']:,}", "xs mu", "end"))
        for j, (key, name, var) in enumerate(series):
            v = r[key]
            if v is None:
                continue
            yy = y + j * (BAR + 2)
            body.append(f'<g><title>{escape(r["task"])}: {escape(name)} {v:.1f}</title><rect x="{L}" y="{yy}" '
                        f'width="{max(sx(v) - L, 1.5):.1f}" height="{BAR}" rx="2" fill="var({var})"{op(var)}/></g>')
            if j == 0:
                body.append(text(sx(v) + 4, yy + 9, f"{v:.1f}", "xs"))
        lo, hi = r["ours_minus_jev_ci95"]
        d = r["ours_lora"] - r["jev"]
        mark = "▲" if r["beats_jev"] else ("▼" if hi < 0 else "n.s.")
        body.append(text(W - 270, y + 18, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}] {mark} · ${r['train_usd']:.2f}", "sm"))
    body.append(f'<line x1="{L}" y1="{T - 8}" x2="{L}" y2="{H - 40}" class="axis"/>')
    return svg(W, H, body, "Verticals: quick LoRA vs Jev vs zero-shot", "Grouped bars for 12 tasks with paired differences to Jev.")


# --- 4. effort -----------------------------------------------------------------------------------------------------------
def effort():
    E, DI, api = D["effort_serving"], D["effort_di"], D["reference_api"]
    W, H = 1060, 420
    body = [text(24, 30, "Effort: think only when unsure", "h1"),
            text(24, 50, "Left: Decision Index 0.2.1 (stratified sample, 95 % CI) of one always-think read re-scored per "
                 "policy (the next-generation base). Right: accuracy on our eval banks and serial latency per decision "
                 "(Wald-4B, one H100).", "xs t2")]
    L, T, PW, PH = 70, 90, 360, 250
    keys = list(DI)
    xs = {k: L + 30 + i * (PW - 60) / (len(keys) - 1) for i, k in enumerate(keys)}
    lo, hi = 38, 54
    sy = lambda v: T + PH - PH * (v - lo) / (hi - lo)
    for v in range(lo, hi + 1, 4):
        body.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{L + PW}" y2="{sy(v):.1f}" class="grid"/>')
        body.append(text(L - 8, sy(v) + 4, v, "xs mu", "end"))
    body.append(text(L - 50, T - 10, "Decision Index", "xs mu"))
    pts = []
    for k in keys:
        r = DI[k]
        if not r:
            continue
        x, (a, b) = xs[k], r["ci95"]
        body.append(f'<line x1="{x:.1f}" y1="{sy(a):.1f}" x2="{x:.1f}" y2="{sy(b):.1f}" stroke="var(--s1)" stroke-width="2"/>')
        body.append(f'<g><title>effort {k}: {r["index"]} [{a}, {b}]</title><circle cx="{x:.1f}" cy="{sy(r["index"]):.1f}" r="5" '
                    f'fill="{"var(--s1)" if k != "medium" else "var(--s2)"}" stroke="var(--surface)" stroke-width="2"/></g>')
        body.append(text(x + 8, sy(r["index"]) + 4, f"{r['index']:.1f}", "xs"))
        body.append(text(x, T + PH + 18, k, "sm b" if k == "medium" else "sm", "middle"))
        pts.append((x, sy(r["index"])))
    body.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="var(--s1)" '
                f'stroke-width="1" stroke-dasharray="3 3"/>')
    body.append(text(L, T + PH + 36, "declared policy: medium (orange) · high-k4 = mean of 4 thoughts", "xs mu"))
    # right panel: accuracy (XL dev) vs latency, one point per effort
    R0, RT, RW, RH = 580, 90, 380, 250
    lmin, lmax = math.log10(0.03), math.log10(3.0)
    lx = lambda s: R0 + RW * (math.log10(s) - lmin) / (lmax - lmin)
    alo, ahi = 70, 90
    ay = lambda v: RT + RH - RH * (v - alo) / (ahi - alo)
    for s in (0.03, 0.1, 0.3, 1, 3):
        body.append(f'<line x1="{lx(s):.1f}" y1="{RT}" x2="{lx(s):.1f}" y2="{RT + RH}" class="grid"/>')
        body.append(text(lx(s), RT + RH + 16, f"{s:g} s", "xs mu", "middle"))
    for v in range(alo, ahi + 1, 5):
        body.append(f'<line x1="{R0}" y1="{ay(v):.1f}" x2="{R0 + RW}" y2="{ay(v):.1f}" class="grid"/>')
        body.append(text(R0 - 8, ay(v) + 4, v, "xs mu", "end"))
    body.append(text(R0 - 40, RT - 10, "XL dev accuracy %", "xs mu"))
    body.append(text(R0 + RW / 2, RT + RH + 34, "serial latency per decision, p50 (dot) to p95 (ring), log scale", "xs mu", "middle"))
    for e in E:
        y, x1, x2 = ay(e["xl_dev"]), lx(e["p50_s"]), lx(e["p95_s"])
        body.append(f'<g><title>effort {e["effort"]}: XL dev {e["xl_dev"]}, p50 {e["p50_s"]} s, p95 {e["p95_s"]} s, thinks on '
                    f'{e["thinks"]} %</title><line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="var(--s1)" stroke-width="2"/>'
                    f'<circle cx="{x1:.1f}" cy="{y:.1f}" r="5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/>'
                    f'<circle cx="{x2:.1f}" cy="{y:.1f}" r="5" fill="var(--surface)" stroke="var(--s1)" stroke-width="2"/></g>')
        body.append(text(x1 - 8, y + 4, f"{e['effort']} ({e['thinks']:g} % think)", "xs", "end"))
    xa = lx(api["p50_s"])
    body.append(f'<line x1="{xa:.1f}" y1="{RT}" x2="{xa:.1f}" y2="{RT + RH}" stroke="var(--s2)" stroke-width="1.5" stroke-dasharray="5 4"/>')
    body.append(text(xa + 5, RT + 12, f"Jev API p50 {api['p50_s']:g} s", "xs t2"))
    return svg(W, H, body, "Effort vs accuracy vs latency", "Index rises with effort; median latency stays near one pass up to medium.")


# --- 5. latency and cost vs Jev ----------------------------------------------------------------------------------------
def latency_cost():
    S, api = D["serving"], D["reference_api"]
    W, H = 1000, 380
    body = [text(24, 30, "Latency and cost per decision vs Jev", "h1"),
            text(24, 50, "Left: median serial latency (ms). Right: $ per 1,000 decisions. Wald-4B on one RTX PRO 6000 "
                 "(self-hosted GPU-hour cost at measured throughput); Jev at API list price.", "xs t2")]
    vt = {r["task"]: r for r in D["verticals"]}
    lat = [("one question, one pass (c = 1)", S["rtxpro6000_c1_p50_ms"], None),
           ("When2Call request", S["vertical_when2call_p50_ms"], S["jev_when2call_p50_ms"]),
           ("BANKING77 request (77 options)", S["vertical_banking77_p50_ms"], S["jev_banking77_p50_ms"])]
    L, T, PW = 220, 100, 250
    mx = 1100
    for i, (name, w, j) in enumerate(lat):
        y = T + i * 70
        body.append(text(L - 10, y + 14, name, "sm", "end"))
        key = "When2Call" if "When2Call" in name else ("BANKING77" if "BANKING77" in name else None)
        laya = vt[key]["laya_p50_ms"] if key else None
        clm = vt[key]["clm_p50_ms"] if key else None
        for k, (v, var, who) in enumerate(((w, "--s1", "Wald-4B"), (j, "--s2", "Jev"), (laya and round(laya), "--laya", "Laya"),
                                           (clm and round(clm), "--clm", "CLM-8B"))):
            if v is None:
                continue
            yy = y + k * 16
            body.append(f'<g><title>{escape(name)}: {escape(str(who))} {v} ms</title><rect x="{L}" y="{yy}" width="{max(PW * v / mx, 1.5):.1f}" '
                        f'height="13" rx="2" fill="var({var})"{op(var)}/></g>')
            body.append(text(L + PW * v / mx + 4, yy + 10, f"{who} {v:g} ms", "xs mu" if var in FAINT else "xs"))
    R0 = 740
    costs = [("Wald-4B fp8, batched", S["rtxpro6000_fp8_usd_per_1k"], "--s1"),
             ("Wald-4B multi-tenant, c = 32", S["multitenant_c32_usd_per_1k"], "--s1"),
             ("Jev API", api["usd_per_1k"], "--s2")]
    for i, (name, v, var) in enumerate(costs):
        y = T + i * 40
        body.append(text(R0 - 10, y + 11, name, "sm", "end"))
        body.append(f'<g><title>{escape(name)}: ${v} per 1,000 decisions</title><rect x="{R0}" y="{y}" width="{200 * v / 0.045:.1f}" '
                    f'height="14" rx="2" fill="var({var})"/></g>')
        body.append(text(R0 + 200 * v / 0.045 + 4, y + 11, f"${v:.4f}", "xs"))
    ratio = api["usd_per_1k"] / S["rtxpro6000_fp8_usd_per_1k"]
    body.append(text(R0 - 150, T + 140, f"Jev costs about {ratio:.1f}× more per decision.", "sm b"))
    body.append(text(24, H - 14, "Laya: its own server, as measured in the same vertical runs. CLM-8B: latency and cost not "
                     "committed yet. Not every system was measured on the same GPU; see the model card.", "xs mu"))
    return svg(W, H, body, "Latency and cost vs Jev", "Median latency and dollars per 1,000 decisions, Wald-4B vs Jev.")


if __name__ == "__main__":
    for name, fn in (("di-vs-size", di_vs_size), ("di-areas", di_areas), ("verticals", verticals),
                     ("latency-cost", latency_cost)):
        (OUT / f"{name}.svg").write_text(fn())
        print("wrote", OUT / f"{name}.svg")
