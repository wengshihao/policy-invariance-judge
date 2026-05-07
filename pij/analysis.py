"""Statistical post-processing used to generate every table and figure
in the paper.

This module consumes the JSON outputs of :mod:`pij.experiments` and
returns plain Python dicts. All tables in the paper can be reproduced by
serialising one of these dicts to JSON or feeding it into
:mod:`pij.figures`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping

import numpy as np

from .metrics import (
    ALL_TRANSFORMS,
    CERTIFIED,
    NEAR_EQUIVALENT,
    _bootstrap_clustered_mean_ci,
    _bootstrap_mean_ci,
    delta_flip,
)


# ── Per-transform CIs (Table 5 / Fig. app_per_transform_ci) ────────────────

def per_transform_ci(results: Iterable[dict]) -> dict:
    """For each transform compute mean Δ_flip and a bootstrap 95% CI."""
    rows = list(results)
    jit_rates = [r["baseline_jitter"]["jitter_rate"] for r in rows
                 if r.get("baseline_jitter", {}).get("verdicts")]
    by_t: dict[str, list[float]] = {}
    for r in rows:
        j = r.get("baseline_jitter", {})
        if not j.get("verdicts"):
            continue
        jr = float(j.get("jitter_rate", 0.0))
        for t in ALL_TRANSFORMS:
            if t in r.get("flips", {}):
                by_t.setdefault(t, []).append(int(r["flips"][t]) - jr)
    out = {
        "n_total": len(rows),
        "jitter_rate": float(np.mean(jit_rates)) if jit_rates else 0.0,
        "transforms": {},
    }
    for t, arr in by_t.items():
        mean, lo, hi = _bootstrap_mean_ci(arr)
        out["transforms"][t] = {
            "n": len(arr),
            "delta_flip": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "certified": t in CERTIFIED,
            "significant": bool(lo > 0),
        }
    return out


# ── Clear vs ambiguous (Table 6) ───────────────────────────────────────────

def clear_vs_ambiguous(results: Iterable[dict]) -> dict:
    """Per-(ambiguity, transform) Δ_flip with item-clustered bootstrap CIs."""
    rows = list(results)
    by_amb_t: dict[str, dict[str, list[tuple[float, str]]]] = {
        a: defaultdict(list) for a in ("clear", "ambiguous", "unknown")
    }
    for r in rows:
        j = r.get("baseline_jitter", {})
        if not j.get("verdicts"):
            continue
        jr = float(j.get("jitter_rate", 0.0))
        amb = r.get("metadata", {}).get("ambiguity", "unknown")
        item_id = r.get("item_id")
        for t in ALL_TRANSFORMS:
            if t in r.get("flips", {}):
                by_amb_t[amb][t].append((int(r["flips"][t]) - jr, item_id))
    out: dict[str, dict] = {}
    for amb, table in by_amb_t.items():
        out[amb] = {}
        for t, pairs in table.items():
            if not pairs:
                continue
            vals, ids = zip(*pairs)
            mean, lo, hi = _bootstrap_clustered_mean_ci(list(vals), list(ids))
            out[amb][t] = {
                "n": len(vals),
                "delta_flip": round(mean, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            }
        cert_pairs = [p for t in CERTIFIED for p in table.get(t, [])]
        if cert_pairs:
            vals, ids = zip(*cert_pairs)
            mean, lo, hi = _bootstrap_clustered_mean_ci(list(vals), list(ids))
            out[amb]["certified_pooled"] = {
                "n": len(vals),
                "n_unique_items": len(set(ids)),
                "delta_flip": round(mean, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            }
    return out


# ── Per-dataset ablation (Table 4) ─────────────────────────────────────────

def per_dataset_ablation(results: Iterable[dict]) -> dict:
    rows = list(results)
    out = {}
    for ds in ("assebench", "rjudge"):
        sub = [r for r in rows if r.get("metadata", {}).get("dataset") == ds]
        if not sub:
            continue
        d = delta_flip(sub)
        pooled = d["by_transform"].get("certified_pooled", {})
        out[ds] = {
            "n_items": len(sub),
            "delta_flip": pooled.get("delta_flip", 0.0),
            "ci_95_lower": pooled.get("ci_95_lower", 0.0),
            "ci_95_upper": pooled.get("ci_95_upper", 0.0),
            "significant": pooled.get("significant", False),
        }
    return out


# ── Domain heatmap (Fig. app_domain_heatmap) ───────────────────────────────

def per_domain_certified_dflip(results: Iterable[dict],
                               min_items: int = 15) -> dict:
    rows = list(results)
    domains = Counter(r.get("metadata", {}).get("policy_domain", "")
                      for r in rows
                      if r.get("metadata", {}).get("policy_domain"))
    keep = [d for d, c in domains.most_common() if c >= min_items]
    out = {}
    for d in keep:
        arr = []
        for r in rows:
            if r.get("metadata", {}).get("policy_domain") != d:
                continue
            j = r.get("baseline_jitter", {})
            if not j.get("verdicts"):
                continue
            jr = float(j.get("jitter_rate", 0.0))
            for t in CERTIFIED:
                if t in r.get("flips", {}):
                    arr.append(int(r["flips"][t]) - jr)
        if arr:
            out[d] = {"n": len(arr), "delta_flip": round(float(np.mean(arr)), 4)}
    return out


# ── Cross-model pairwise (Table 8 + Jaccard fig) ───────────────────────────

def pairwise_baseline_agreement(model_results: Mapping[str, list[dict]]) -> dict:
    anchors: dict[str, dict[str, str]] = {}
    for m, results in model_results.items():
        d = {}
        for r in results:
            j = r.get("baseline_jitter", {})
            if not j or not j.get("anchor_verdict"):
                continue
            d[r["item_id"]] = j["anchor_verdict"]
        anchors[m] = d
    models = list(model_results)
    out = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            common = set(anchors[a]) & set(anchors[b])
            if not common:
                continue
            agree = sum(1 for k in common if anchors[a][k] == anchors[b][k])
            out[f"{a}__vs__{b}"] = {
                "n_common": len(common),
                "agreement_rate": round(agree / len(common), 4),
            }
    return out


def pairwise_flip_jaccard(model_results: Mapping[str, list[dict]]) -> dict:
    flip_sets: dict[str, set[str]] = {}
    for m, results in model_results.items():
        s = set()
        for r in results:
            for t in CERTIFIED:
                if r.get("flips", {}).get(t):
                    s.add(r["item_id"])
        flip_sets[m] = s
    models = list(model_results)
    out = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            sa, sb = flip_sets[a], flip_sets[b]
            if not sa and not sb:
                jacc = 1.0
            else:
                jacc = len(sa & sb) / max(1, len(sa | sb))
            out[f"{a}__vs__{b}"] = {
                "jaccard": round(jacc, 4),
                "n_a": len(sa),
                "n_b": len(sb),
                "n_intersection": len(sa & sb),
                "n_union": len(sa | sb),
            }
    return out


# ── Strict-to-lenient breakdown (Table 2 + ambiguity slice) ────────────────

def strict_lenient_breakdown(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    flips = [r for r in rows if r["flip"]]
    u2s = sum(1 for r in flips if r["direction"] == "strict_only_unsafe")
    s2u = sum(1 for r in flips if r["direction"] == "lenient_only_unsafe")
    by_amb = defaultdict(lambda: [0, 0])
    for r in rows:
        a = r["metadata"].get("ambiguity", "unknown")
        by_amb[a][0] += int(r["flip"])
        by_amb[a][1] += 1
    return {
        "n_total": len(rows),
        "n_flips": len(flips),
        "flip_rate": round(len(flips) / max(1, len(rows)), 4),
        "unsafe_to_safe": u2s,
        "safe_to_unsafe": s2u,
        "directional_ratio": round(u2s / max(1, u2s + s2u) if (u2s + s2u) > 0 else 1.0, 4),
        "by_ambiguity": {
            a: {"flip_rate": round(v[0] / max(1, v[1]), 4), "n": v[1]}
            for a, v in by_amb.items()
        },
    }


# ── T6 (irrelevant context) directional breakdown ─────────────────────────

def irrelevant_context_breakdown(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    flips = [r for r in rows if r["flip"]]
    s2u = sum(1 for r in flips
              if r.get("original_verdict") == "SAFE"
              and r.get("irrelevant_ctx_verdict") == "UNSAFE")
    u2s = sum(1 for r in flips
              if r.get("original_verdict") == "UNSAFE"
              and r.get("irrelevant_ctx_verdict") == "SAFE")
    by_amb = defaultdict(lambda: [0, 0])
    for r in rows:
        a = r["metadata"].get("ambiguity", "unknown")
        by_amb[a][0] += int(r["flip"])
        by_amb[a][1] += 1
    return {
        "n_total": len(rows),
        "flip_rate": round(len(flips) / max(1, len(rows)), 4),
        "safe_to_unsafe": s2u,
        "unsafe_to_safe": u2s,
        "by_ambiguity": {
            a: {"flip_rate": round(v[0] / max(1, v[1]), 4), "n": v[1]}
            for a, v in by_amb.items()
        },
    }


# ── Disagreement decomposition (Fig. fig5_disagreement) ────────────────────

def disagreement_decomposition(results: Iterable[dict]) -> dict:
    rows = list(results)
    explainable = unreasonable = borderline = total = 0
    for r in rows:
        amb = r.get("metadata", {}).get("ambiguity", "unknown")
        for t, flipped in r.get("flips", {}).items():
            if not flipped:
                continue
            total += 1
            cert = t in CERTIFIED
            if not cert:
                explainable += 1
            elif amb == "ambiguous":
                explainable += 1
            elif amb == "clear":
                unreasonable += 1
            else:
                borderline += 1
    return {
        "total_flips": total,
        "explainable": explainable,
        "borderline": borderline,
        "unreasonable": unreasonable,
        "explainable_pct": round(explainable / max(total, 1), 4),
        "unreasonable_pct": round(unreasonable / max(total, 1), 4),
    }
