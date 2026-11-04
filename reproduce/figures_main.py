"""Reproduce the main-text figures used in the paper.

The published paper carries two figures in the main text:

  * ``figures/overview.pdf``        — the three-principle overview, shipped as a
                                      static asset (not regenerated from results).
  * ``figures/fig5_disagreement.pdf`` — disagreement decomposition (paper Fig. 2,
                                      §5 Principle 3).

This script regenerates the second one. ``overview.pdf`` is bundled in the
repo and is not produced from experiment results.

For supplementary plots that did not make the final paper but are useful
for inspection (forest plots, heatmaps, ambiguity moderation, judge-card
panel), see ``reproduce/figures_extra.py``.

Usage
-----
    python reproduce/figures_main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.analysis import disagreement_decomposition
from pij.config import FIGURES_DIR
from pij.figures import fig_disagreement
from reproduce._common import PAPER_MODELS, load_all_judge_results


def main() -> None:
    print("Loading model results...")
    model_results = load_all_judge_results(PAPER_MODELS)
    if not model_results:
        print("No judge results found. Run scripts/run_judge.py first.")
        return

    print("Fig 2 — disagreement decomposition")
    decomp = {label: disagreement_decomposition(model_results[label])
              for label in model_results}
    fig_disagreement(decomp, FIGURES_DIR / "fig5_disagreement.pdf")
    print(f"\nWrote {FIGURES_DIR / 'fig5_disagreement.pdf'}")
    print("Note: figures/overview.pdf is shipped statically, not regenerated.")


if __name__ == "__main__":
    main()
