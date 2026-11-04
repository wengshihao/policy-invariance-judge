"""Reproduce paper Table 9: per-dataset (ASSEBench vs R-Judge) Δ_flip.

Usage
-----
    python reproduce/table4_dataset_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.analysis import per_dataset_ablation
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
        rows.append({"model": label, "data": per_dataset_ablation(results)})

    out = RESULTS_DIR / "table4_dataset_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}\n")

    print("## Table 9 — Per-Dataset Certified Pooled Δ_flip\n")
    print("| Model | ASSEBench Δ_flip | n | R-Judge Δ_flip | n |")
    print("|-------|------------------|---|----------------|---|")
    for r in rows:
        d = r["data"]
        a = d.get("assebench", {})
        b = d.get("rjudge", {})

        def fmt(x):
            if not x:
                return "—"
            return (f"{x['delta_flip'] * 100:+.1f}% "
                    f"[{x['ci_95_lower'] * 100:+.1f}, {x['ci_95_upper'] * 100:+.1f}]")

        print(f"| {r['model']} | {fmt(a)} | {a.get('n_items', 0)} | "
              f"{fmt(b)} | {b.get('n_items', 0)} |")


if __name__ == "__main__":
    main()
