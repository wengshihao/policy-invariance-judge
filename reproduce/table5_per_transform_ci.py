"""Reproduce Table 5: per-(model, transform) Δ_flip with bootstrap 95% CIs.

Usage
-----
    python reproduce/table5_per_transform_ci.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.analysis import per_transform_ci
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
        rows.append({"model": label, "stats": per_transform_ci(results)})

    out = RESULTS_DIR / "table5_per_transform_ci.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}\n")

    print("## Table 5 — Per-Transform Δ_flip with 95% CI\n")
    print("| Model | Jitter | T1 | T2 | T3 | T4 | T5 |")
    print("|-------|--------|----|----|----|----|----|")
    for r in rows:
        s = r["stats"]
        cells = []
        for t in ("T1", "T2", "T3", "T4", "T5"):
            v = s["transforms"].get(t, {})
            if not v:
                cells.append("—")
                continue
            cells.append(f"{v['delta_flip'] * 100:+.1f}% "
                         f"[{v['ci_lo'] * 100:+.1f}, {v['ci_hi'] * 100:+.1f}]")
        print(f"| {r['model']} | {s['jitter_rate'] * 100:.1f}% | "
              + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
