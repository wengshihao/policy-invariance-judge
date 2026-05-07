"""End-to-end Judge Card walk-through for an arbitrary judge model.

This is the simplest entry point for a downstream user who wants to
audit one of their own judges. The script:

  1. samples a small balanced subset of ASSEBench + R-Judge,
  2. generates rewrites with the configured rewriter model,
  3. runs the rubric, strict-lenient, and irrelevant-context experiments,
  4. emits a Judge Card (PIS, P1, P2, P3) for the model.

For real audits we recommend ``--n-items 200``; the default 30 is a
smoke test.

Usage
-----
    # 30-item smoke test
    python scripts/audit_my_judge.py --model gpt-4o-mini

    # full Judge Card audit
    python scripts/audit_my_judge.py --model gpt-4o-mini --n-items 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.config import DATA_DIR, RESULTS_DIR, default_judge_model
from pij.data_loader import build_master_table, save_master_table
from pij.experiments import (
    run_irrelevant_context,
    run_rubric_invariance,
    run_strict_lenient,
)
from pij.judge_card import (
    build_judge_card,
    render_judge_card_markdown,
    write_judge_card,
)
from pij.perturbation import generate_rewrites_for_dataset
from pij.policies import BASE_POLICY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help=f"Judge model (default: PIJ_JUDGE_MODEL = "
                             f"{default_judge_model()}).")
    parser.add_argument("--n-items", type=int, default=30,
                        help="Items to sample. 30 = smoke test, 200 = real audit.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = args.model or default_judge_model()
    tag = model.replace(".", "_").replace("-", "_")

    # Sample balanced master table.
    master_path = DATA_DIR / f"audit_master_{tag}.json"
    print(f"[1/4] Sampling {args.n_items} items...")
    n_clear = max(args.n_items // 4, 5)
    n_amb = max(args.n_items // 4, 5)
    n_rj = max(args.n_items - n_clear - n_amb, 5)
    master = build_master_table(n_asse=n_clear * 2, n_rjudge=n_rj, seed=args.seed)
    save_master_table(master, master_path)

    # Generate rewrites.
    rewrites_path = DATA_DIR / f"audit_rewrites_{tag}.json"
    print(f"[2/4] Generating rewrites with the rewrite model...")
    rewrites = generate_rewrites_for_dataset(master, rewrites_path, BASE_POLICY)

    # Run experiments.
    print(f"[3/4] Judging with {model}...")
    judge_results = run_rubric_invariance(
        master, rewrites, model=model,
        output_path=RESULTS_DIR / f"audit_judge_{tag}.json", verbose=False,
    )
    sl_rows = run_strict_lenient(
        master, model=model, max_items=args.n_items,
        output_path=RESULTS_DIR / f"audit_strict_lenient_{tag}.json",
        verbose=False,
    )
    run_irrelevant_context(
        master, model=model, max_items=args.n_items,
        output_path=RESULTS_DIR / f"audit_irrelevant_{tag}.json",
        verbose=False,
    )

    # Build the Judge Card.
    print(f"[4/4] Building Judge Card for {model}...")
    card = build_judge_card(model, judge_results, strict_lenient_rows=sl_rows)
    out = RESULTS_DIR / f"audit_card_{tag}.json"
    write_judge_card(card, out)
    print()
    print(render_judge_card_markdown(card))
    print(f"\nJudge Card written to {out}")


if __name__ == "__main__":
    main()
