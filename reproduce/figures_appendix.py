"""Reproduce all eight appendix figures used in the paper.

The paper carries the following figures in the appendix:

    figures/app_per_transform_ci.pdf   — Fig: per-transform CIs faceted by model
    figures/app_jitter_dist.pdf        — Fig: distribution of per-item jitter
    figures/app_jitter_vs_dflip.pdf    — Fig: jitter vs certified Δ_flip
    figures/app_domain_heatmap.pdf     — Fig: per-domain certified Δ_flip
    figures/app_strict_lenient_dir.pdf — Fig: direction of strict→lenient flips
    figures/app_pairwise_jaccard.pdf   — Fig: Jaccard overlap of flipped items
    figures/app_pis_decomposition.pdf  — Fig: PIS deduction decomposition
    figures/app_pis_weights.pdf        — Fig: rank stability under random weights

Usage
-----
    python reproduce/figures_appendix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.analysis import (
    pairwise_flip_jaccard,
    per_domain_certified_dflip,
    per_transform_ci,
    strict_lenient_breakdown,
)
from pij.config import FIGURES_DIR
from pij.figures import (
    fig_domain_heatmap,
    fig_jitter_distribution,
    fig_jitter_vs_dflip,
    fig_pairwise_jaccard,
    fig_per_transform_ci,
    fig_pis_decomposition,
    fig_pis_weight_stability,
    fig_strict_lenient_direction,
)
from pij.judge_card import build_judge_card
from reproduce._common import (
    PAPER_MODELS,
    load_all_judge_results,
    load_strict_lenient,
)


def main() -> None:
    print("Loading model results...")
    model_results = load_all_judge_results(PAPER_MODELS)
    if not model_results:
        print("No judge results found. Run scripts/run_judge.py first.")
        return

    per_model_pt = {label: per_transform_ci(model_results[label])
                    for label in model_results}

    print("app_per_transform_ci")
    fig_per_transform_ci(per_model_pt, FIGURES_DIR / "app_per_transform_ci.pdf")

    print("app_jitter_dist")
    fig_jitter_distribution(model_results, FIGURES_DIR / "app_jitter_dist.pdf")

    print("app_jitter_vs_dflip")
    fig_jitter_vs_dflip(per_model_pt, FIGURES_DIR / "app_jitter_vs_dflip.pdf")

    print("app_domain_heatmap")
    domain_grid = {label: per_domain_certified_dflip(model_results[label])
                   for label in model_results}
    fig_domain_heatmap(domain_grid, FIGURES_DIR / "app_domain_heatmap.pdf")

    print("app_pairwise_jaccard")
    jacc = pairwise_flip_jaccard(model_results)
    fig_pairwise_jaccard(jacc, list(model_results),
                         FIGURES_DIR / "app_pairwise_jaccard.pdf")

    sl_breakdowns = {}
    cards = {}
    for label, model_id in PAPER_MODELS.items():
        if label not in model_results:
            continue
        try:
            sl_rows = load_strict_lenient(model_id)
        except FileNotFoundError:
            print(f"  [skip strict-lenient] {label}")
            continue
        sl_breakdowns[label] = strict_lenient_breakdown(sl_rows)
        cards[label] = build_judge_card(model_id, model_results[label],
                                        strict_lenient_rows=sl_rows)

    if sl_breakdowns:
        print("app_strict_lenient_dir")
        fig_strict_lenient_direction(sl_breakdowns,
                                     FIGURES_DIR / "app_strict_lenient_dir.pdf")

    if cards:
        print("app_pis_decomposition")
        fig_pis_decomposition(cards, FIGURES_DIR / "app_pis_decomposition.pdf")
        print("app_pis_weights")
        fig_pis_weight_stability(cards, FIGURES_DIR / "app_pis_weights.pdf")

    print(f"\nAll appendix figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
