"""Build the Judge Card from a judge's per-item results.

Usage
-----
    python scripts/make_judge_card.py \
        --model gpt-4o-mini \
        --judge-results results/judge_gpt_4o_mini.json \
        --strict-lenient results/strict_lenient_gpt_4o_mini.json

Outputs both JSON (``results/judge_card_<tag>.json``) and a Markdown
summary written to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.judge_card import build_judge_card, render_judge_card_markdown, write_judge_card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Identifier shown on the Judge Card.")
    parser.add_argument("--judge-results", required=True,
                        help="JSON output of scripts/run_judge.py.")
    parser.add_argument("--strict-lenient", default=None,
                        help="JSON output of scripts/run_strict_lenient.py. "
                             "If omitted, R_dir defaults to 1.")
    parser.add_argument("--u-rate", type=float, default=None,
                        help="Override the U_rate (e.g. for benchmarks "
                             "without ambiguity labels).")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.judge_results) as f:
        judge_results = json.load(f)
    sl_rows = None
    if args.strict_lenient:
        with open(args.strict_lenient) as f:
            sl_rows = json.load(f)

    card = build_judge_card(args.model, judge_results,
                            strict_lenient_rows=sl_rows,
                            U_rate_override=args.u_rate)

    out = args.output or f"results/judge_card_{args.model.replace('.', '_').replace('-', '_')}.json"
    write_judge_card(card, out)
    print(render_judge_card_markdown(card))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
