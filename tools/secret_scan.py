"""Scan this repository for secrets and private infrastructure before any push. Prints variable NAMES only, never values.

    python tools/secret_scan.py --env /path/to/.env [--extra-pattern REGEX ...]
Exit code 1 on any hit."""
import argparse, os, re, sys

ap = argparse.ArgumentParser(); ap.add_argument("--env", required=True); ap.add_argument("--extra-pattern", action="append", default=[])
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pairs = []
for line in open(a.env):
    m = re.match(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
    if m: pairs.append((m.group(1), m.group(2).strip().strip('"\'')))
files = [os.path.join(d, f) for d, ds, fs in os.walk(root) for f in fs
         if not any(p in d.split(os.sep) for p in (".git", "__pycache__", ".venv", ".pytest_cache"))
         and os.path.abspath(os.path.join(d, f)) != os.path.abspath(__file__)]
pat = re.compile("|".join([r"/Users/", r"/root/", r"/private/tmp", r"autodl", r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9]{20,}",
                           r"ghp_[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}", r"BEGIN [A-Z ]*PRIVATE KEY"] + a.extra_pattern), re.I)
hits = 0
for f in files:
    t = open(f, errors="ignore").read(); rel = os.path.relpath(f, root)
    for name, val in pairs:
        if name in t: print(f"env NAME {name} in {rel}"); hits += 1
        if len(val) >= 8 and val in t: print(f"env VALUE of {name} in {rel}"); hits += 1
    for i, line in enumerate(t.splitlines(), 1):
        for m in pat.finditer(line): print(f"pattern {m.group(0)!r} at {rel}:{i}"); hits += 1
print(f"{len(pairs)} env variables, {len(files)} files, {hits} hits")
sys.exit(1 if hits else 0)
