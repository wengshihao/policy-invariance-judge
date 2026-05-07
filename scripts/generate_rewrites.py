"""Generate certified-equivalent (T1, T2, T4) and near-equivalent
(T3, T5) rewrites of the base safety policy for every item.

Usage
-----
    python scripts/generate_rewrites.py
    python scripts/generate_rewrites.py --max-items 100

Output: ``data/rewrites_all.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.config import default_rewrite_model
from pij.perturbation import generate_rewrites_for_dataset
from pij.policies import BASE_POLICY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="data/master_table.json")
    parser.add_argument("--output", default="data/rewrites_all.json")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--model", default=None,
                        help=f"Generator model. Default: PIJ_REWRITE_MODEL "
                             f"({default_rewrite_model()}).")
    args = parser.parse_args()

    with open(args.master) as f:
        master = json.load(f)
    print(f"Generating rewrites for {len(master) if not args.max_items else args.max_items} items "
          f"with {args.model or default_rewrite_model()}...")
    rewrites = generate_rewrites_for_dataset(
        master, args.output, BASE_POLICY,
        max_items=args.max_items, model=args.model,
    )
    print(f"Wrote {args.output} ({len(rewrites)} rewrite sets)")


if __name__ == "__main__":
    main()
