"""Reproduce Table 8 + Jaccard table: cross-model anchor verdict
agreement and flipped-item Jaccard overlap.

Usage
-----
    python reproduce/table8_cross_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.analysis import pairwise_baseline_agreement, pairwise_flip_jaccard
from pij.config import RESULTS_DIR
from reproduce._common import PAPER_MODELS, load_all_judge_results


def main() -> None:
    model_results = load_all_judge_results(PAPER_MODELS)
    if len(model_results) < 2:
        print("Need at least two judge result files to compute pairwise stats.")
        return
    agreement = pairwise_baseline_agreement(model_results)
    jaccard = pairwise_flip_jaccard(model_results)
    out = RESULTS_DIR / "table8_cross_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"agreement": agreement, "jaccard": jaccard}, indent=2))
    print(f"Wrote {out}\n")

    print("## Pairwise Baseline Agreement\n")
    print("| Pair | n_common | Agreement |")
    print("|------|----------|-----------|")
    for k, v in agreement.items():
        a, b = k.split("__vs__")
        print(f"| {a} vs {b} | {v['n_common']} | {v['agreement_rate']:.3f} |")

    print("\n## Pairwise Flipped-Item Jaccard (certified)\n")
    print("| Pair | Jaccard | |A∩B| | |A∪B| |")
    print("|------|---------|-------|-------|")
    for k, v in jaccard.items():
        a, b = k.split("__vs__")
        print(f"| {a} vs {b} | {v['jaccard']:.3f} | {v['n_intersection']} | "
              f"{v['n_union']} |")


if __name__ == "__main__":
    main()
