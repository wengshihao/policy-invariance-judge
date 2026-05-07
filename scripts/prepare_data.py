"""Build the 500-item master table from the public ASSEBench / R-Judge releases.

Usage
-----
    python scripts/prepare_data.py                     # full 500-item pool
    python scripts/prepare_data.py --n-asse 100 --n-rjudge 50  # smaller pool

Both benchmarks are open releases that you must fetch yourself.
The expected layout is::

    data/ASSEBench/ASSEBench/dataset/AgentJudge-strict.json
    data/ASSEBench/ASSEBench/dataset/AgentJudge-loose.json
    data/R-Judge/data/<category>/<file>.json

You can clone them with::

    git clone https://github.com/Eligius-MS/ASSEBench data/ASSEBench
    git clone https://github.com/Lordog/R-Judge        data/R-Judge

The script writes ``data/master_table.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.data_loader import build_master_table, save_master_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-asse", type=int, default=300,
                        help="ASSEBench items (split 50/50 clear/ambiguous).")
    parser.add_argument("--n-rjudge", type=int, default=200,
                        help="R-Judge items (stratified by gold label).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/master_table.json")
    args = parser.parse_args()

    master = build_master_table(args.n_asse, args.n_rjudge, args.seed)
    out = save_master_table(master, args.output)
    print(f"Wrote {out} ({len(master)} items)")


if __name__ == "__main__":
    main()
