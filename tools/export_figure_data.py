"""Export the numbers behind figures/*.svg and the comparison table into figures/data.json.

    python tools/export_figure_data.py --src <research checkout> --summary <exported summary.json>

Reads only committed result files of the research checkout (paths below are relative to --src):
  results/decision-index/board-live.json            the live Decision Index 0.2.1 board (official full-suite numbers)
  results/decision-index/boot021-*.json             our paired-bootstrap reads on the 6,948-request stratified sample
  verticals/<task>/runs/*/report.json               raw Qwen3.5-4B-Base zero-shot on each vertical's fixed test set
and the exported summary (verticals: ours + LoRA / Jev / Wald zero-shot; effort table; serving cost and latency).

A read that is not committed yet is taken from PENDING (a number relayed from the run, marked "reported") and is
replaced automatically once its bootstrap file exists. Nothing here is recomputed: the script copies and labels.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Our Decision Index rows. boot = (bootstrap file, model key) once committed; reported = relayed number until then.
WALD_ROWS = [  # one row: v1.0 = 021A0-f10 at the declared policy (medium, prompt format repeat_state_plain)
    {"id": "v1.0", "label": "Wald-4B v1.0 · effort medium", "boot": ("boot021-0927-021A0-f10.json", "arrmed"),
     "reported": {"index": 53.91, "blocklist_adjusted": 53.62, "board_rank": 6},
     "reported_areas_file": "figures/reported-areas-v1.0.json"},
]
EFFORT_KEYS = {"none": "g4one", "low": "g4low", "medium": "g4med", "high": "g4high", "high-k4": "g4k4"}
QWEN27 = ("boot021-bigbase-0927.json", "q27paren")
JEV_SAMPLE = ("boot021-0927.json", "jev")
BOARD_NAMES = {"decider_4b": "Decider 4B", "decider_35b": "Decider 35B-A3B"}
SUITE = [  # JevBench public 231 (accuracy %) and XL hidden split (XL-Int), zero-shot, same requests
    {"system": "Wald-4B v1.0 · medium (packaged server)", "public231": round(100 * 204 / 231, 1), "xl_hidden": 63.2,
     "source": "public 231: packaged wald-serve smoke read of 021A0-f10 09-27 (ledger 197); XL: xl-bench leaderboard.md (hidden split)"},
    {"system": "Jev (API, jev-1.13.0)", "public231": 86.6, "xl_hidden": 73.0, "source": "results/external/jev-1.13.0; xl-bench #1"},
    {"system": "Cygnet (gemma-4-12B-it + shim)", "public231": None, "xl_hidden": 59.8, "source": "xl-bench #4"},
    {"system": "Laya (zero-shot, Jev's wire request)", "public231": 58.0, "xl_hidden": 13.4, "source": "results/external/laya; xl-bench #24"},
    {"system": "CLM-8B (zero-shot, same requests, deployment verified)", "public231": 39.0, "xl_hidden": 12.7,
     "source": "ledger 190 read (results/external/clm-8b); xl-bench #25"},
    {"system": "raw Qwen3.5-4B-Base (zero-shot, our readout)", "public231": round(100 * 156 / 231, 1), "xl_hidden": None,
     "source": "exported lineage summary (public one-pass 156 / 231)"},
]
NC_TASKS = {"WebLINX intent"}
COLD_PAIRED_CI = [7.7, 10.0]   # paired bootstrap, docs: buyer-demo COLD v3 full test (+8.8)
COLD_USD = 1.0                 # three COLD arms shared about 1 GPU-hour (about $1) on one RTX PRO 6000
VERTICAL_TASKS = {  # summary task name -> vertical directory
    "When2Call": "when2call", "BANKING77": "banking77", "SGD intent": "sgd-intent", "ToxicChat": "toxicchat",
    "MetaTool": "metatool", "AndroidControl": "androidcontrol", "WebLINX intent": "weblinx-intent",
    "RouteLLM routing": "model-routing", "Agent-trajectory safety (hard)": "agent-trajectory-safety-hard",
    "Mind2Web": "mind2web", "Prompt injection": "prompt-injection", "RewardBench": "rewardbench",
}


def size_b(base_model: str | None):
    """Total parameters in billions read from a base model name (35B-A3B -> 35, E4B -> 4); None when absent."""
    if not base_model:
        return None
    m = re.search(r"(?<![A-Za-z0-9.])E?(\d+(?:\.\d+)?)B(?:-A\d+B)?", base_model.split("/")[-1])
    return float(m.group(1)) if m else None


def boot(src: Path, f: str, key: str):
    p = src / "results/decision-index" / f
    if not p.is_file():
        return None
    m = json.loads(p.read_text())["models"].get(key)
    if not m:
        return None
    return {"index": m["balanced_skill"], "ci95": m["ci"]["balanced_skill"], "areas": m["areas"],
            "source": f"results/decision-index/{f}#{key}"}


def external_vertical(src: Path, task_dir: str, prefix: str):
    """A zero-shot external system's read on a vertical (system id starting with `prefix`), with its p50 latency."""
    for rep in sorted(glob.glob(str(src / "verticals" / task_dir / "runs/*/report.json"))):
        for s in json.loads(Path(rep).read_text())["systems"]:
            if s["id"] == prefix or (s["id"].startswith(prefix) and "multilingual" not in s["id"]):
                return {"value": round(100 * s["metric"], 1), "p50_ms": s.get("p50_ms"), "source": str(Path(rep).relative_to(src))}
    return None


def raw_qwen_vertical(src: Path, task_dir: str):
    for rep in sorted(glob.glob(str(src / "verticals" / task_dir / "runs/*/report.json"))):
        for s in json.loads(Path(rep).read_text())["systems"]:
            if s["id"] == "hosted-qwen35-4b-raw" or (s["kind"] == "base" and "(raw)" in s["label"] and "chat" not in s["label"]):
                return {"value": round(100 * s["metric"], 1), "source": str(Path(rep).relative_to(src))}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    src = Path(a.src)
    summ = json.loads(Path(a.summary).read_text())
    board = json.loads((src / "results/decision-index/board-live.json").read_text())

    entries = sorted(board["entries"], key=lambda e: -e["index"])
    out = {"board": {"edition": board["edition"], "generated_utc": board["generated_utc"],
                     "jev": {"index": board["jev"]["index"], "areas": board["jev"]["areas"]},
                     "entries": [{"rank": i + 1, "name": e["name"], "size_b": size_b(e["base_model"]),
                                  "base_model": e["base_model"] if size_b(e["base_model"]) else None,
                                  "kind": e["kind"], "index": e["index"], "areas": e["areas"]} for i, e in enumerate(entries)]}}
    wald = []
    for r in WALD_ROWS:
        got = boot(src, *r["boot"])
        row = {"id": r["id"], "label": r["label"], "size_b": 4.0}
        if got:
            row.update(got, status="committed")
        elif r.get("reported"):
            row.update(r["reported"], status="reported (bootstrap not committed yet)")
            ra = ROOT / r["reported_areas_file"]
            if ra.is_file():
                row["areas"] = json.loads(ra.read_text())["areas"]
        else:
            continue
        wald.append(row)
    out["wald_di"] = wald
    out["effort_di"] = {k: boot(src, "boot021-0927-015G0-f4.json", v) for k, v in EFFORT_KEYS.items()}
    out["qwen27_paren"] = boot(src, *QWEN27)
    out["jev_sample"] = boot(src, *JEV_SAMPLE)
    out["board_named"] = {k: next(({"rank": e["rank"], "index": e["index"], "base_model": e["base_model"]}
                                   for e in out["board"]["entries"] if e["name"] == n), None) for k, n in BOARD_NAMES.items()}
    out["board_4_5"] = [out["board"]["entries"][3], out["board"]["entries"][4]]

    verts = []
    for v in summ["verticals"]:
        d = VERTICAL_TASKS.get(v["task"])
        raw = raw_qwen_vertical(src, d) if d else None
        laya = external_vertical(src, d, "hosted-laya") if d else None
        clm = external_vertical(src, d, "hosted-clm") if d else None
        verts.append({"laya_0shot": laya and laya["value"], "laya_p50_ms": laya and laya["p50_ms"],
                      "clm_0shot": clm and clm["value"], "clm_p50_ms": clm and clm["p50_ms"],
                      "task": v["task"], "metric": v["metric"], "n_test": v["n_test"], "ours_lora": v["lora_all"],
                      "ours_ci95": v.get("lora_all_ci95"), "jev": v["jev"], "wald_0shot": v["base"],
                      "raw_qwen_0shot": raw["value"] if raw else None, "raw_source": raw["source"] if raw else None,
                      "ours_minus_jev_ci95": v.get("all_minus_jev_ci95"), "train_usd": v.get("train_usd"),
                      "beats_jev": v.get("beats_jev")})
    # NC-licensed task data: internal only, not on the public card
    verts = [v for v in verts if v["task"] not in NC_TASKS]
    # COLD (Chinese offensive language), full test set, from the committed buyer-demo read
    cold = json.loads((src / "results/buyer/cold-v3/full-test/summary.json").read_text())["systems"]
    f1 = lambda k: round(100 * cold[k]["full5323"]["macro_f1"]["value"], 1)
    ci = [round(100 * x, 1) for x in cold["01900-f8+base"]["full5323"]["macro_f1"]["ci95"]]
    verts.append({"task": "COLD (Chinese)", "metric": "macro-F1", "n_test": 5323, "ours_lora": f1("01900-f8+base"),
                  "ours_ci95": ci, "jev": f1("jev"), "wald_0shot": f1("015D0-f4"), "raw_qwen_0shot": None, "raw_source": None,
                  "laya_0shot": None, "laya_p50_ms": None, "clm_0shot": None, "clm_p50_ms": None,
                  "ours_minus_jev_ci95": COLD_PAIRED_CI, "train_usd": COLD_USD, "beats_jev": True,
                  "note": "LoRA on all 25,382 train labels averaged with the zero-shot base; ties the best published 83.7"})
    out["verticals"] = verts
    # same-request suite reads (JevBench public 231 accuracy %, XL-Intelligence) from committed reports / notes
    out["suite"] = SUITE
    out["effort_serving"] = summ["effort"]
    out["reference_api"] = summ["reference_api"]
    out["serving"] = summ["serving"]
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "figures/data.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"wald_di": [(w["id"], w["index"], w["status"]) for w in wald],
                      "verticals_with_raw": sum(v["raw_qwen_0shot"] is not None for v in verts)}))


if __name__ == "__main__":
    main()
