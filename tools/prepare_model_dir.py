"""Prepare a weights directory for release: copy the checkpoint's own temperature table without its provenance field,
write serving.json (declared policy and prompt format), and write MANIFEST.json with the sha256 of every file.

    python tools/prepare_model_dir.py WEIGHTS_DIR --temperature path/to/temperature.json --prompt-format plain
"""
import argparse, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("weights"); ap.add_argument("--temperature", required=True)
ap.add_argument("--prompt-format", choices=("plain", "repeat_state_plain"), required=True)
ap.add_argument("--effort", default="medium"); ap.add_argument("--max-model-len", type=int, default=131072)
a = ap.parse_args()
w = Path(a.weights)
t = json.loads(Path(a.temperature).read_text())
t = {k: v for k, v in t.items() if k in ("A", "B", "B256", "B512", "buckets", "single")}
for k, v in t.items():   # keep only the fields the server reads
    if isinstance(v, dict):
        t[k] = {kk: vv for kk, vv in v.items() if kk in ("single", "buckets", "min_rows", "rows", "readout")}
(w / "temperature.json").write_text(json.dumps(t, indent=1) + "\n")
(w / "serving.json").write_text(json.dumps({"effort": a.effort, "prompt_format": a.prompt_format,
                                            "max_model_len": a.max_model_len, "temperature": "temperature.json"}, indent=1) + "\n")
man = {}
for f in sorted(w.rglob("*")):
    if f.is_file() and f.name != "MANIFEST.json" and ".cache" not in f.parts:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        man[str(f.relative_to(w))] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()}
(w / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps({"files": len(man), "serving": json.loads((w / "serving.json").read_text())}))
