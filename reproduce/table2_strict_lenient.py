"""Reproduce paper Table 3: strict-to-lenient flip rate and directional ratio.

Usage
-----
    python reproduce/table2_strict_lenient.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.analysis import strict_lenient_breakdown
from pij.config import RESULTS_DIR
from reproduce._common import PAPER_MODELS, load_strict_lenient


def main() -> None:
    rows = []
    for label, model_id in PAPER_MODELS.items():
        try:
            data = load_strict_lenient(model_id)
        except FileNotFoundError as exc:
            print(f"  [skip] {label}: {exc}")
            continue
        b = strict_lenient_breakdown(data)
        rows.append({
            "model": label,
            "flip_rate": b["flip_rate"],
            "unsafe_to_safe": b["unsafe_to_safe"],
            "safe_to_unsafe": b["safe_to_unsafe"],
            "directional_ratio": b["directional_ratio"],
            "n_total": b["n_total"],
            "by_ambiguity": b["by_ambiguity"],
        })

    out = RESULTS_DIR / "table2_strict_lenient.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}\n")

    print("## Table 3 — Strict-to-Lenient Switching\n")
    print("| Model | Flip Rate | Unsafe→Safe | Safe→Unsafe | R_dir |")
    print("|-------|-----------|-------------|-------------|-------|")
    for r in rows:
        print(f"| {r['model']} | {r['flip_rate'] * 100:.1f}% | "
              f"{r['unsafe_to_safe']} | {r['safe_to_unsafe']} | "
              f"{r['directional_ratio']:.3f} |")


if __name__ == "__main__":
    main()
