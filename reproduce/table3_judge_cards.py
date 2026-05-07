"""Reproduce Table 3: Judge Cards (PIS, P1, P2, P3).

Combines per-judge results with strict-lenient logs to instantiate the
Judge Card for every paper model. Writes
``results/judge_cards_final.json``.

Usage
-----
    python reproduce/table3_judge_cards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pij.config import RESULTS_DIR
from pij.judge_card import build_judge_card
from reproduce._common import PAPER_MODELS, load_judge_results, load_strict_lenient


def main() -> None:
    cards = {}
    for label, model_id in PAPER_MODELS.items():
        try:
            judge_results = load_judge_results(model_id)
            sl_rows = load_strict_lenient(model_id)
        except FileNotFoundError as exc:
            print(f"  [skip] {label}: {exc}")
            continue
        card = build_judge_card(model_id, judge_results, strict_lenient_rows=sl_rows)
        cards[label] = card

    out = RESULTS_DIR / "judge_cards_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, indent=2))
    print(f"Wrote {out}\n")

    print("## Table 3 — Judge Cards\n")
    print("| Model | PIS | P1 (cert. Δ_flip) | P2 (R_dir) | P3 (U_rate) | Interpretation |")
    print("|-------|-----|-------------------|------------|-------------|----------------|")
    for label, card in cards.items():
        p = card["principles"]
        pis = card["policy_invariance_score"]
        if pis["bracket_lower"] == pis["bracket_upper"]:
            pis_text = f"{pis['value']:.2f}"
        else:
            pis_text = f"[{pis['bracket_lower']:.2f}, {pis['bracket_upper']:.2f}]"
        d_lo = p["P1_certified_delta_flip"]["lower"]
        d_hi = p["P1_certified_delta_flip"]["upper"]
        d_text = (f"{d_lo * 100:.1f}%" if not p["P1_certified_delta_flip"]["has_parse_failures"]
                  else f"[{d_lo * 100:.1f}%, {d_hi * 100:.1f}%]")
        rdir = p["P2_directional_ratio"]["directional_ratio"]
        urate = p["P3_unreasonable_rate"]["unreasonable_pct"]
        print(f"| {label} | {pis_text} | {d_text} | {rdir:.2f} | "
              f"{urate * 100:.0f}% | {card['interpretation']} |")


if __name__ == "__main__":
    main()
