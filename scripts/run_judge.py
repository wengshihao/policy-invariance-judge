"""Run a judge model on every item: 3 jitter reruns + 5 rewrite verdicts.

This implements Principle 1 (rubric-semantics invariance) and is the
backbone of every other experiment.

Usage
-----
    python scripts/run_judge.py --model gpt-4o-mini
    python scripts/run_judge.py --model gpt-4o-mini --max-items 50

Output: ``results/judge_<model_tag>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.config import default_judge_model
from pij.experiments import run_rubric_invariance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help=f"Judge model. Default: PIJ_JUDGE_MODEL "
                             f"({default_judge_model()}).")
    parser.add_argument("--master", default="data/master_table.json")
    parser.add_argument("--rewrites", default="data/rewrites_all.json")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--n-jitter", type=int, default=3,
                        help="Number of identical reruns at T=0.")
    parser.add_argument("--output", default=None,
                        help="Override output path.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoint.")
    args = parser.parse_args()

    with open(args.master) as f:
        master = json.load(f)
    with open(args.rewrites) as f:
        rewrites = json.load(f)

    run_rubric_invariance(
        master, rewrites,
        model=args.model,
        n_jitter_reruns=args.n_jitter,
        max_items=args.max_items,
        output_path=args.output,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
