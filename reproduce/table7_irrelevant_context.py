"""Reproduce Table 7 (paper Table for T6 — irrelevant context injection).

Usage
-----
    python reproduce/table7_irrelevant_context.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.analysis import irrelevant_context_breakdown
from pij.config import RESULTS_DIR
from reproduce._common import PAPER_MODELS, load_irrelevant_context


def main() -> None:
    rows = []
    for label, model_id in PAPER_MODELS.items():
        try:
            data = load_irrelevant_context(model_id)
        except FileNotFoundError as exc:
            print(f"  [skip] {label}: {exc}")
            continue
        b = irrelevant_context_breakdown(data)
        b["model"] = label
        rows.append(b)

    out = RESULTS_DIR / "table7_irrelevant_context.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}\n")

    print("## Table 7 — T6: Irrelevant Context Injection\n")
    print("| Model | Total Flip | Clear | Ambig. | S→U | U→S |")
    print("|-------|-----------|-------|--------|-----|-----|")
    for r in rows:
        amb = r["by_ambiguity"]
        clear = amb.get("clear", {}).get("flip_rate", 0.0)
        ambig = amb.get("ambiguous", {}).get("flip_rate", 0.0)
        print(f"| {r['model']} | {r['flip_rate'] * 100:.1f}% | "
              f"{clear * 100:.1f}% | {ambig * 100:.1f}% | "
              f"{r['safe_to_unsafe']} | {r['unsafe_to_safe']} |")


if __name__ == "__main__":
    main()
