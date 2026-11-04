"""Reproduce paper Table 10: certified pooled Δ_flip on clear vs ambiguous items.

Usage
-----
    python reproduce/table6_clear_vs_ambiguous.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.analysis import clear_vs_ambiguous
from pij.config import RESULTS_DIR
from reproduce._common import PAPER_MODELS, load_judge_results


def main() -> None:
    rows = []
    for label, model_id in PAPER_MODELS.items():
        try:
            results = load_judge_results(model_id)
        except FileNotFoundError as exc:
            print(f"  [skip] {label}: {exc}")
            continue
        rows.append({"model": label, "data": clear_vs_ambiguous(results)})

    out = RESULTS_DIR / "table6_clear_vs_ambiguous.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}\n")

    print("## Table 10 — Clear vs Ambiguous Pooled Δ_flip\n")
    print("| Model | Clear Δ_flip | n | Ambig. Δ_flip | n | Diff (pp) |")
    print("|-------|--------------|---|---------------|---|-----------|")
    for r in rows:
        d = r["data"]
        clear = d.get("clear", {}).get("certified_pooled", {})
        amb = d.get("ambiguous", {}).get("certified_pooled", {})

        def fmt(x):
            return f"{x['delta_flip'] * 100:+.1f}%" if x else "—"

        diff = ((amb.get("delta_flip", 0) - clear.get("delta_flip", 0)) * 100
                if amb and clear else 0.0)
        print(f"| {r['model']} | {fmt(clear)} | {clear.get('n', 0)} | "
              f"{fmt(amb)} | {amb.get('n', 0)} | {diff:+.1f} |")


if __name__ == "__main__":
    main()
