"""Judge Card: the per-judge reporting protocol from the paper.

A Judge Card summarises one judge model along the three principles:

  P1 — certified Δ_flip                  (rubric-semantics invariance)
  P2 — directional ratio R_dir           (rubric-threshold invariance)
  P3 — unreasonable flip rate U_rate     (ambiguity-aware calibration)

and rolls them into the Policy Invariance Score (PIS).

The :func:`build_judge_card` function is the single entry point downstream
users call after running their own judge through the protocol.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .metrics import (
    PIS_SCALE,
    PIS_WEIGHTS,
    delta_flip,
    directional_ratio,
    policy_invariance_score,
    unreasonable_rate,
)

CERTIFIED = ("T1", "T2", "T4")
PARSE_FAIL_VERDICTS = {"PARSE_ERROR", "API_ERROR"}


def _certified_delta_endpoints(results: Iterable[dict]) -> dict:
    """Return ``Δ_cert`` lower (parseable-only) and upper (parse-failures
    imputed as flips) endpoints, plus parse-failure bookkeeping.

    Matches the convention in Appendix I of the paper.
    """
    arr_lo, arr_up = [], []
    n_jitter_ok = n_parse_fail = n_skipped = n_valid = 0
    for r in results:
        j = r.get("baseline_jitter", {})
        if not j.get("verdicts"):
            continue
        n_jitter_ok += 1
        jr = float(j.get("jitter_rate", 0.0))
        for t in CERTIFIED:
            rew = r.get("rewrite_verdicts", {}).get(t, {})
            verdict = rew.get("verdict")
            if verdict in ("SAFE", "UNSAFE") and t in r.get("flips", {}):
                v = int(r["flips"][t]) - jr
                arr_lo.append(v)
                arr_up.append(v)
                n_valid += 1
            elif verdict == "SKIPPED":
                n_skipped += 1
            elif verdict in PARSE_FAIL_VERDICTS:
                n_parse_fail += 1
                arr_up.append(1.0 - jr)
    return {
        "n_items_jitter_ok": n_jitter_ok,
        "n_valid_cert_pairs": n_valid,
        "n_parse_fail_cert_pairs": n_parse_fail,
        "n_skipped_cert_pairs": n_skipped,
        "has_parse_failures": n_parse_fail > 0,
        "delta_lower": float(sum(arr_lo) / len(arr_lo)) if arr_lo else 0.0,
        "delta_upper": float(sum(arr_up) / len(arr_up)) if arr_up else 0.0,
    }


def build_judge_card(model: str,
                     judge_results: list[dict],
                     strict_lenient_rows: Optional[list[dict]] = None,
                     U_rate_override: Optional[float] = None,
                     weights: tuple[float, float, float] = PIS_WEIGHTS,
                     scale: float = PIS_SCALE) -> dict:
    """Build a Judge Card from a complete set of per-item results.

    Parameters
    ----------
    model : str
        The judge model identifier shown in the card.
    judge_results : list[dict]
        Output of :func:`pij.judge.judge_with_rewrites` for every item.
    strict_lenient_rows : list[dict] | None
        Output of :func:`pij.experiments.run_strict_lenient` for the same
        items. If ``None``, the card sets ``R_dir = 1`` (best case).
    U_rate_override : float | None
        Skip the automatic clear-vs-ambiguous decomposition and use this
        value instead. Useful for benchmarks without ambiguity labels.
    """
    delta = delta_flip(judge_results)
    cert_endpoints = _certified_delta_endpoints(judge_results)

    if strict_lenient_rows:
        p2 = directional_ratio(strict_lenient_rows)
        rdir = p2["directional_ratio"]
    else:
        p2 = {"directional_ratio": 1.0, "n_total": 0, "n_flips": 0,
              "unsafe_to_safe": 0, "safe_to_unsafe": 0, "flip_rate": 0.0,
              "note": "strict_lenient not provided — defaulting R_dir=1"}
        rdir = 1.0

    if U_rate_override is not None:
        u3 = {"unreasonable_pct": float(U_rate_override),
              "note": "U_rate supplied externally"}
        urate = float(U_rate_override)
    else:
        u3 = unreasonable_rate(judge_results)
        urate = u3["unreasonable_pct"]

    pis_lower_d = policy_invariance_score(cert_endpoints["delta_lower"], rdir, urate,
                                          weights=weights, scale=scale)
    if cert_endpoints["has_parse_failures"]:
        pis_upper_d = policy_invariance_score(cert_endpoints["delta_upper"], rdir, urate,
                                              weights=weights, scale=scale)
        # Higher Δ_flip => lower PIS
        pis_block = {
            "value": None,
            "bracket_lower": pis_upper_d["policy_invariance_score"],
            "bracket_upper": pis_lower_d["policy_invariance_score"],
        }
        interp = f"{pis_upper_d['interpretation']} / {pis_lower_d['interpretation']}"
    else:
        pis_block = {
            "value": pis_lower_d["policy_invariance_score"],
            "bracket_lower": pis_lower_d["policy_invariance_score"],
            "bracket_upper": pis_lower_d["policy_invariance_score"],
        }
        interp = pis_lower_d["interpretation"]

    return {
        "model": model,
        "n_items_tested": len(judge_results),
        "policy_invariance_score": pis_block,
        "interpretation": interp,
        "weights": list(weights),
        "scale": scale,
        "principles": {
            "P1_certified_delta_flip": {
                "lower": round(cert_endpoints["delta_lower"], 4),
                "upper": round(cert_endpoints["delta_upper"], 4),
                "n_valid_pairs": cert_endpoints["n_valid_cert_pairs"],
                "n_parse_fail_pairs": cert_endpoints["n_parse_fail_cert_pairs"],
                "n_skipped_pairs": cert_endpoints["n_skipped_cert_pairs"],
                "n_items_jitter_ok": cert_endpoints["n_items_jitter_ok"],
                "has_parse_failures": cert_endpoints["has_parse_failures"],
            },
            "P2_directional_ratio": p2,
            "P3_unreasonable_rate": u3,
        },
        "delta_flip_full": delta,
        "deduction_breakdown": {
            "at_lower_dflip": pis_lower_d["components"],
            "at_upper_dflip": (
                pis_upper_d["components"] if cert_endpoints["has_parse_failures"]
                else pis_lower_d["components"]
            ),
        },
    }


def write_judge_card(card: Mapping, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2))
    return path


def render_judge_card_markdown(card: Mapping) -> str:
    """Pretty Markdown rendering for a single Judge Card."""
    p = card["principles"]
    pis = card["policy_invariance_score"]
    if pis["value"] is not None:
        pis_text = f"**{pis['value']:.2f}**"
    else:
        pis_text = f"**[{pis['bracket_lower']:.2f}, {pis['bracket_upper']:.2f}]**"
    lines = [
        f"### Judge Card: `{card['model']}`",
        "",
        f"- **Policy Invariance Score (PIS)**: {pis_text}  "
        f"({card['interpretation']})",
        f"- **Items tested**: {card['n_items_tested']}",
        "",
        "| Principle | Metric | Value |",
        "| --------- | ------ | ----- |",
        f"| P1 (rubric-semantics) | certified Δ_flip lower | "
        f"{p['P1_certified_delta_flip']['lower']:.4f} |",
        f"| P1 | certified Δ_flip upper | "
        f"{p['P1_certified_delta_flip']['upper']:.4f} |",
        f"| P2 (rubric-threshold) | R_dir | "
        f"{p['P2_directional_ratio']['directional_ratio']:.3f} "
        f"({p['P2_directional_ratio']['unsafe_to_safe']} U→S, "
        f"{p['P2_directional_ratio']['safe_to_unsafe']} S→U) |",
        f"| P3 (calibration) | U_rate | "
        f"{p['P3_unreasonable_rate']['unreasonable_pct']:.3f} |",
        "",
        f"_Weights {tuple(card['weights'])}, scale S = {card['scale']}._",
    ]
    return "\n".join(lines)
