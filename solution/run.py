"""Regenerate the submission from the provided data bundle, then validate it.

    python solution/run.py                      # writes evaluation/results/hmm_v1.csv
    python solution/run.py --out my_team.csv --no-cache

The parsed tracking is cached in data/cache/ (gitignored, never publish it); --no-cache deletes
it first so the run starts from the raw files.
"""
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.decode import decode, stats, write  # noqa: E402
from solution.features import build  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "hmm_v1.csv"))
p.add_argument("--no-cache", action="store_true")
a = p.parse_args()

if a.no_cache:
    cache = os.path.join(ROOT, "data", "cache")
    if os.path.isdir(cache):
        for f in os.listdir(cache):
            os.remove(os.path.join(cache, f))

E, _ = build()
labels = decode(E)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
write(a.out, labels)
print(stats(labels))
print("sha256", hashlib.sha256(open(a.out, "rb").read()).hexdigest())
subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), a.out], check=True)
