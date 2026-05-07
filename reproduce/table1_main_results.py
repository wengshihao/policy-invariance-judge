"""Reproduce Table 1: certified-equivalent Δ_flip by model and transform.

Reads ``results/judge_<tag>.json`` for every paper model and writes
``results/table1_main_results.json`` plus a Markdown table to stdout.

Usage
-----
    python reproduce/table1_main_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.config import RESULTS_DIR
from pij.metrics import cross_model_summary
from reproduce._common import PAPER_MODELS, load_all_judge_results


def _fmt_pct(x: float, signed: bool = True) -> str:
    if signed:
        sign = "+" if x >= 0 else "-"
        return f"{sign}{abs(x) * 100:.1f}%"
    return f"{x * 100:.1f}%"


def main() -> None:
    print("Loading judge results...")
    model_results = load_all_judge_results(PAPER_MODELS)
    summary = cross_model_summary(model_results)

    rows = []
    for label, _ in PAPER_MODELS.items():
        if label not in summary:
            continue
        s = summary[label]
        pt = s["per_transform"]
        # mean per-item jitter
        results = model_results[label]
        jitter_rates = [r.get("baseline_jitter", {}).get("jitter_rate", 0.0)
                        for r in results]
        mean_jitter = sum(jitter_rates) / max(1, len(jitter_rates))
        rows.append({
            "model": label,
            "n_items": s["n_items"],
            "T1": pt.get("T1", 0.0),
            "T2": pt.get("T2", 0.0),
            "T4": pt.get("T4", 0.0),
            "pooled_certified": s["certified_delta_flip"],
            "T3": pt.get("T3", 0.0),
            "T5": pt.get("T5", 0.0),
            "jitter": mean_jitter,
            "ci_95_lower": s["ci_95"][0],
            "ci_95_upper": s["ci_95"][1],
            "significant": s["significant"],
        })

    out_path = RESULTS_DIR / "table1_main_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out_path}")

    # Markdown
    print("\n## Table 1 — Certified-Equivalent Δ_flip\n")
    print("| Model | n | T1 | T2 | T4 | Pooled | T3 | T5 | Jitter |")
    print("|-------|---|----|----|----|--------|----|----|--------|")
    for r in rows:
        sig = "**" if r["significant"] else ""
        print(f"| {r['model']} | {r['n_items']} | {_fmt_pct(r['T1'])} | "
              f"{_fmt_pct(r['T2'])} | {_fmt_pct(r['T4'])} | "
              f"{sig}{_fmt_pct(r['pooled_certified'])}{sig} | "
              f"{_fmt_pct(r['T3'])} | {_fmt_pct(r['T5'])} | "
              f"{_fmt_pct(r['jitter'], signed=False)} |")


if __name__ == "__main__":
    main()
