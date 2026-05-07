"""Run the strict-to-lenient experiment for Principle 2.

The judge sees the same trajectory under a strict policy and a lenient
policy that cover identical risk categories but differ in normative
threshold language. A reliable judge should flip ``UNSAFE -> SAFE``
predominantly.

Usage
-----
    python scripts/run_strict_lenient.py --model gpt-4o-mini
    python scripts/run_strict_lenient.py --model gpt-4o-mini --max-items 100

Output: ``results/strict_lenient_<model_tag>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.config import default_judge_model
from pij.experiments import run_strict_lenient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help=f"Default: PIJ_JUDGE_MODEL ({default_judge_model()}).")
    parser.add_argument("--master", default="data/master_table.json")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.master) as f:
        master = json.load(f)
    run_strict_lenient(master, model=args.model,
                       max_items=args.max_items, output_path=args.output)


if __name__ == "__main__":
    main()
