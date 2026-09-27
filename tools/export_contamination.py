"""Export contamination/trained-on-di-ids.json: every Decision Index item id found in the training data of the
submitted checkpoint's whole lineage, with the training component(s) it was found in and why it is there.

    python tools/export_contamination.py --src <research checkout>

Inputs (relative to --src; committed scan outputs, ids and counts only, no item text):
  results/decision-index/contamination-verified.json      strict-verified hits of the first scan (stage-1 corpus)
  trainer/data/blocklists/filters/<dataset>.di-scan.json  full-blocklist scans of each training dataset
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# lineage component -> the scan that covers it (the stage-1 corpus scan covers the verified stage-1 hits as well)
COMPONENTS = {
    "stage 1 (full-parameter decision training corpus)": "corpus-c11-train",
    "stage 2 (LoRA refinement data)": "finetune-f3-stage2",
    "stage 3 (short-thought distillation data)": "think-paren-v1",
}
VERIFIED_STAGE1 = "c9"
# Stage 4 (the DI-targeted LoRA) and the RL arm's data were filtered with the full blocklist before training: 0 hits.

WHY = {
    "4": ("BANKING77", "scored", "train-split duplicate: the benchmark's public train split contains the same sentence as the test item"),
    "5": ("CLINC150+OOS", "scored", "train-split duplicate: the public train split contains the same sentence as the test item"),
    "28": ("WinoGrande", "scored", "train-split duplicate: the public train split contains the same sentence"),
    "26": ("ARC-Easy", "display only", "train-split duplicate (ARC train / MMLU auxiliary_train), the board's known duplicates"),
    "27": ("ARC-Challenge", "display only", "train-split duplicate (ARC train / MMLU auxiliary_train), the board's known duplicates"),
    "24": ("MMLU", "display only", "question also present in MMLU auxiliary_train / other public QA sets"),
    "2": ("ToolRet", "scored", "shared upstream source: ToolRet aggregates user queries of public tool-calling train sets (ToolACE, Glaive) that are in our data"),
    "33": ("SATA-Bench", "scored", "shared upstream source: reading passage from RACE, which is part of MMLU auxiliary_train"),
    "36": ("BRIGHT", "scored", "shared upstream source: a public programming problem statement (LeetCode)"),
    "57": ("MMLU-Pro", "scored", "shared upstream source: MMLU-Pro includes problems from MATH / TheoremQA; the same problem is in our math data"),
    "12": ("ANLI", "scored", "shared upstream text: the premise text also appears in another public dataset in our data"),
    "6": ("RouterBench", "not scored in 0.2.1", "shared upstream source: RouterBench prompts embed GSM8K / MMLU questions that are in our math and QA data"),
}


def bench(i: str) -> str:
    return i.split(":")[1] if i.startswith("candidates-v3:") else i.split(":")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    src = Path(a.src)
    where = collections.defaultdict(set)
    ver = json.loads((src / "results/decision-index/contamination-verified.json").read_text())
    for ids in ver["run_ids"].get(VERIFIED_STAGE1, {}).values():
        for i in ids:
            where[i].add("stage 1 (full-parameter decision training corpus)")
    scans = {}
    for comp, name in COMPONENTS.items():
        d = json.loads((src / f"trainer/data/blocklists/filters/{name}.di-scan.json").read_text())
        scans[comp] = {"strict_hits_in_trained_files": d["strict_total"], "blocklist_sha256": d["blocklist"]["strict_sha256"],
                       "suite_fingerprints_sha256": d["blocklist"]["suite_sha256"],
                       "files": {f: {"lines": v["lines"], "sha256": v["sha256"], "strict": v["strict"]} for f, v in d["files"].items()}}
        for i in d["di_items"]:
            where[i].add(comp)
    items = []
    for i in sorted(where, key=lambda x: (int(bench(x)), x)):
        b = bench(i)
        name, scored, why = WHY[b]
        items.append({"id": i, "benchmark_id": int(b), "benchmark": name, "scored_in_0_2_1": scored, "why": why,
                      "found_in": sorted(where[i])})
    per = collections.Counter((it["benchmark_id"], it["benchmark"], it["scored_in_0_2_1"]) for it in items)
    out = {"schema": "wald/di-trained-on/1",
           "what": "Decision Index 0.2 / 0.2.1 item ids present in the training data of the submitted checkpoint's lineage "
                   "(strict match: the whole state in one training record, plus >= 50 % of option text where options are the "
                   "item's content). Count every one of them as trained on.",
           "total": len(items),
           "per_benchmark": [{"benchmark_id": k[0], "benchmark": k[1], "scored_in_0_2_1": k[2], "items": n} for k, n in sorted(per.items())],
           "scans": scans,
           "stage_4_targeted_lora": "stage-4 data (Home-appliance generator rows; iSarcasmEval / API-Bank / ContractNLI / VAST / "
                                    "NLI4CT / ACOS / RAGTruth train splits; replay) was checked against the full blocklist and the "
                                    "sample rows before training: 0 hits after filtering",
           "items": items}
    (ROOT / "contamination").mkdir(exist_ok=True)
    (ROOT / "contamination/trained-on-di-ids.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"total": len(items), "per_benchmark": out["per_benchmark"]}))


if __name__ == "__main__":
    main()
