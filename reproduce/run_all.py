"""Reproduce the entire paper from scratch.

This script orchestrates the full pipeline assuming you have:

  1. Cloned ASSEBench under ``data/ASSEBench`` and R-Judge under ``data/R-Judge``.
  2. Set ``PIJ_API_KEY`` and ``PIJ_BASE_URL`` in ``.env``.

It will, in order:

  a. build the master table (data/master_table.json)
  b. generate rewrites (data/rewrites_all.json)
  c. for every paper model (override with --models): run the rubric,
     strict-lenient, and irrelevant-context experiments
  d. emit every paper table and figure under results/ and figures/.

API costs scale linearly with --max-items × number of models, so prefer
a small ``--max-items`` for a smoke test.

Usage
-----
    python reproduce/run_all.py                                        # full run
    python reproduce/run_all.py --max-items 25                          # smoke test
    python reproduce/run_all.py --models gpt-4o-mini claude-haiku-4-5   # custom set
    python reproduce/run_all.py --skip-judge                            # only rebuild tables/figures from cached results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.config import DATA_DIR, default_judge_model
from pij.data_loader import build_master_table, save_master_table
from pij.experiments import (
    run_irrelevant_context,
    run_rubric_invariance,
    run_strict_lenient,
)
from pij.perturbation import generate_rewrites_for_dataset
from pij.policies import BASE_POLICY


from reproduce._common import PAPER_MODELS as _PAPER_MODELS

# Default to the four paper models (or whatever PIJ_PAPER_MODELS overrides).
# Pass --models to evaluate a different set; only those whose result files
# exist will appear in the regenerated tables and figures.
PAPER_MODELS_DEFAULT = list(_PAPER_MODELS.values())


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Judge model IDs to evaluate. Defaults to PIJ_JUDGE_MODEL.")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--n-asse", type=int, default=300)
    parser.add_argument("--n-rjudge", type=int, default=200)
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip rebuilding master_table.json.")
    parser.add_argument("--skip-rewrites", action="store_true",
                        help="Skip generating rewrites_all.json.")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip the rubric experiment (Principle 1).")
    parser.add_argument("--skip-strict-lenient", action="store_true")
    parser.add_argument("--skip-irrelevant-context", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    models = args.models or PAPER_MODELS_DEFAULT
    print(f"Using judge models: {models}")

    # 1. Master table.
    master_path = DATA_DIR / "master_table.json"
    if not args.skip_data and not _exists(master_path):
        print("\n[1/6] Building master table...")
        master = build_master_table(args.n_asse, args.n_rjudge)
        save_master_table(master, master_path)
    with open(master_path) as f:
        master = json.load(f)
    print(f"  master table has {len(master)} items")

    # 2. Rewrites.
    rewrites_path = DATA_DIR / "rewrites_all.json"
    if not args.skip_rewrites and not _exists(rewrites_path):
        print("\n[2/6] Generating rewrites (this calls the rewriter API)...")
        generate_rewrites_for_dataset(master, rewrites_path, BASE_POLICY,
                                      max_items=args.max_items)
    with open(rewrites_path) as f:
        rewrites = json.load(f)
    print(f"  rewrites loaded for {len(rewrites)} items")

    # 3. Per-model experiments.
    for model in models:
        print(f"\n[3/6] Running experiments for {model}...")
        if not args.skip_judge:
            run_rubric_invariance(master, rewrites, model=model,
                                  max_items=args.max_items)
        if not args.skip_strict_lenient:
            run_strict_lenient(master, model=model,
                               max_items=args.max_items or 200)
        if not args.skip_irrelevant_context:
            run_irrelevant_context(master, model=model,
                                   max_items=args.max_items or 200)

    # 4. Tables.
    if not args.skip_tables:
        print("\n[4/6] Reproducing tables...")
        for name in ("table1_main_results", "table2_strict_lenient",
                     "table3_judge_cards", "table4_dataset_ablation",
                     "table5_per_transform_ci", "table6_clear_vs_ambiguous",
                     "table7_irrelevant_context", "table8_cross_model"):
            try:
                __import__(f"reproduce.{name}", fromlist=["main"]).main()
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip {name}] {exc}")

    # 5. Figures.
    if not args.skip_figures:
        print("\n[5/6] Rendering figures...")
        for module_name in ("reproduce.figures_main",
                             "reproduce.figures_appendix",
                             "reproduce.figures_extra"):
            try:
                __import__(module_name, fromlist=["main"]).main()
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip {module_name}] {exc}")

    print("\n[6/6] Done. Inspect results/ and figures/ for the rebuilt artifacts.")


if __name__ == "__main__":
    main()
