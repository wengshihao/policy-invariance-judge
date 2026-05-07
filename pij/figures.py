"""Figure factory for the paper.

Each ``fig_*`` function takes the raw analysis dicts and writes one PDF
under ``figures/``. Keeping them small and parameterised means
``reproduce/`` scripts only need to load JSON results and call one
function per figure.

Figures actually used in the paper are listed in
``reproduce/figures_main.py`` (one main-text figure besides the overview)
and ``reproduce/figures_appendix.py`` (eight appendix figures).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .config import FIGURES_DIR
from .metrics import ALL_TRANSFORMS, CERTIFIED


PLOT_PALETTE = {
    "GPT-mini":      "#10a37f",
    "Claude-Haiku":  "#cc785c",
    "DeepSeek":      "#3a6ad8",
    "Gemini-Flash":  "#a259ff",
    "default":       "#444444",
}


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ── Forest plot of certified Δ_flip (Fig. 1) ───────────────────────────────

def fig_cross_model_forest(per_model_delta: Mapping[str, dict],
                           out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(per_model_delta) * 4)))
    y = 0
    yticks, ylabels = [], []
    for model, d in per_model_delta.items():
        color = PLOT_PALETTE.get(model, PLOT_PALETTE["default"])
        for t in CERTIFIED:
            dt = d["by_transform"].get(t)
            if not dt:
                continue
            ax.errorbar(dt["delta_flip"], y,
                        xerr=[[dt["delta_flip"] - dt["ci_95_lower"]],
                              [dt["ci_95_upper"] - dt["delta_flip"]]],
                        fmt="o", color=color, markersize=5, capsize=3, linewidth=1.4, alpha=0.7)
            ylabels.append(f"{model}  {t}")
            yticks.append(y)
            y += 1
        pooled = d["by_transform"].get("certified_pooled")
        if pooled:
            ax.errorbar(pooled["delta_flip"], y,
                        xerr=[[pooled["delta_flip"] - pooled["ci_95_lower"]],
                              [pooled["ci_95_upper"] - pooled["delta_flip"]]],
                        fmt="D", color=color, markersize=8, capsize=4, linewidth=2)
            ylabels.append(f"{model}  pooled")
            yticks.append(y)
            y += 1.5
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0.05, color="red", linestyle=":", alpha=0.4, label="5% threshold")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta_{\mathrm{flip}}$ (rewrite flip rate − baseline jitter)")
    ax.set_title("Certified-equivalent rewrites: Δ_flip across judge models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Heatmap of model × transform Δ_flip (Fig. 2) ───────────────────────────

def fig_transform_heatmap(per_model_delta: Mapping[str, dict],
                          out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    transforms = list(ALL_TRANSFORMS)
    matrix = []
    labels = []
    for model, d in per_model_delta.items():
        labels.append(model)
        matrix.append([d["by_transform"].get(t, {}).get("delta_flip", 0.0)
                       for t in transforms])
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7, 0.65 * len(labels) + 1.6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=-0.05, vmax=0.25)
    ax.set_xticks(range(len(transforms)))
    ax.set_xticklabels(transforms)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, _ in enumerate(labels):
        for j, _ in enumerate(transforms):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.1 else "black")
    ax.axvline(2.5, color="black", linewidth=2)
    ax.set_title(r"$\Delta_{\mathrm{flip}}$ by model × transform")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=r"$\Delta_{\mathrm{flip}}$")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Ambiguity moderation (Fig. 4) ──────────────────────────────────────────

def fig_ambiguity_moderation(per_model_delta: Mapping[str, dict],
                             out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    labels = list(per_model_delta)
    clear_d, amb_d = [], []
    for m in labels:
        am = per_model_delta[m]["ambiguity_moderation"]
        clear_d.append(am.get("clear", {}).get("delta_flip", 0.0))
        amb_d.append(am.get("ambiguous", {}).get("delta_flip", 0.0))
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, clear_d, w, label="Clear", color="#4CAF50", alpha=0.85)
    ax.bar(x + w / 2, amb_d, w, label="Ambiguous", color="#F44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"Certified $\Delta_{\mathrm{flip}}$")
    ax.set_title("Ambiguity moderation of policy invariance (Principle 3)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Disagreement decomposition (Fig. 5) ────────────────────────────────────

def fig_disagreement(per_model_decomp: Mapping[str, dict],
                     out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    labels = list(per_model_decomp)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(labels) + 1.5))
    y = np.arange(len(labels))[::-1]
    color_exp, color_bdr, color_unr = "#4C8B6F", "#E0B341", "#C0504D"
    for yi, m in zip(y, labels):
        d = per_model_decomp[m]
        total = max(1, d["total_flips"])
        e, b, u = (100 * d["explainable"] / total,
                   100 * d["borderline"] / total,
                   100 * d["unreasonable"] / total)
        ax.barh(yi, e, color=color_exp, edgecolor="white")
        ax.barh(yi, b, left=e, color=color_bdr, edgecolor="white")
        ax.barh(yi, u, left=e + b, color=color_unr, edgecolor="white")
        ax.text(101, yi, f"n = {d['total_flips']}", va="center", fontsize=8.5, color="#444")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Share of disagreements (%)")
    ax.set_title("Disagreement decomposition per model")
    handles = [Patch(facecolor=color_exp, label="Explainable"),
               Patch(facecolor=color_bdr, label="Borderline"),
               Patch(facecolor=color_unr, label="Unreasonable")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Judge Card visual (Fig. 6) ─────────────────────────────────────────────

def fig_judge_cards(cards: Mapping[str, dict],
                    out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    labels = list(cards)
    pis_lo = [cards[m]["policy_invariance_score"]["bracket_lower"] for m in labels]
    pis_hi = [cards[m]["policy_invariance_score"]["bracket_upper"] for m in labels]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(labels) + 1.6))
    y = np.arange(len(labels))
    centers = [(a + b) / 2 for a, b in zip(pis_lo, pis_hi)]
    widths = [(b - a) / 2 for a, b in zip(pis_lo, pis_hi)]
    ax.errorbar(centers, y, xerr=widths, fmt="o",
                color="#1f9a3b", markersize=8, capsize=4, linewidth=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Policy Invariance Score")
    for thresh, _ in [(0.7, "ROBUST"), (0.5, "MODERATE"), (0.3, "FRAGILE")]:
        ax.axvline(thresh, color="gray", linestyle=":", alpha=0.4)
    for yi, lo, hi in zip(y, pis_lo, pis_hi):
        if lo == hi:
            ax.text(hi + 0.02, yi, f"{hi:.2f}", va="center", fontsize=9)
        else:
            ax.text(hi + 0.02, yi, f"[{lo:.2f}, {hi:.2f}]", va="center", fontsize=9)
    ax.set_title("Judge Card: Policy Invariance Score")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Per-transform CIs faceted by model (App. fig) ─────────────────────────

def fig_per_transform_ci(per_model_pt: Mapping[str, dict],
                         out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    n = len(per_model_pt)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n + 1, 3.2), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (model, d) in zip(axes, per_model_pt.items()):
        xs = np.arange(len(ALL_TRANSFORMS))
        means, los, his, cert_mask = [], [], [], []
        for t in ALL_TRANSFORMS:
            v = d["transforms"].get(t, {})
            means.append(v.get("delta_flip", 0.0))
            los.append(v.get("ci_lo", 0.0))
            his.append(v.get("ci_hi", 0.0))
            cert_mask.append(t in CERTIFIED)
        color = PLOT_PALETTE.get(model, PLOT_PALETTE["default"])
        bar_colors = [color if c else "#bcbcbc" for c in cert_mask]
        err_lo = [m - lo for m, lo in zip(means, los)]
        err_hi = [hi - m for m, hi in zip(means, his)]
        ax.bar(xs, means, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.errorbar(xs, means, yerr=[err_lo, err_hi], fmt="none", ecolor="black",
                    capsize=2.5, lw=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(0.05, color="red", lw=0.5, ls=":", alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(ALL_TRANSFORMS)
        ax.set_title(model, fontsize=9.5)
        ax.set_ylim(-0.05, 0.3)
    axes[0].set_ylabel(r"$\Delta_{\mathrm{flip}}$")
    handles = [Patch(color="#666", label="Certified-equiv. (T1, T2, T4)"),
               Patch(color="#bcbcbc", label="Near-equiv. (T3, T5)")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Strict-to-lenient direction (App. fig) ────────────────────────────────

def fig_strict_lenient_direction(sl_breakdowns: Mapping[str, dict],
                                 out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    labels = list(sl_breakdowns)
    us = [sl_breakdowns[m]["unsafe_to_safe"] for m in labels]
    su = [sl_breakdowns[m]["safe_to_unsafe"] for m in labels]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = np.arange(len(labels))
    ax.bar(xs, us, color="#3a7d9c", edgecolor="black", lw=0.4, label=r"Unsafe → Safe")
    ax.bar(xs, [-v for v in su], color="#c45a4f", edgecolor="black", lw=0.4,
           label=r"Safe → Unsafe")
    for i, (u, s) in enumerate(zip(us, su)):
        ax.text(i, u + 1, f"{u}", ha="center", fontsize=9)
        ax.text(i, -s - 2, f"{s}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Verdict change count")
    ax.set_title("Direction of strict-to-lenient flips")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Jitter vs Δ_cert scatter ───────────────────────────────────────────────

def fig_jitter_vs_dflip(per_model_pt: Mapping[str, dict],
                        out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for model, d in per_model_pt.items():
        jr = d.get("jitter_rate", 0.0)
        cert_arr = [d["transforms"].get(t, {}).get("delta_flip")
                    for t in CERTIFIED if t in d["transforms"]]
        cert_arr = [v for v in cert_arr if v is not None]
        if not cert_arr:
            continue
        df = float(np.mean(cert_arr))
        color = PLOT_PALETTE.get(model, PLOT_PALETTE["default"])
        ax.scatter(jr, df, s=110, color=color, edgecolor="black", lw=0.5)
        ax.annotate(model, (jr, df), xytext=(7, 3), textcoords="offset points",
                    fontsize=9)
    ax.axhline(0.05, color="red", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel(r"Baseline jitter $P_{\mathrm{jitter}}$")
    ax.set_ylabel(r"Certified pooled $\Delta_{\mathrm{flip}}$")
    ax.set_title("Jitter does not explain rewrite sensitivity")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Pairwise Jaccard heatmap ───────────────────────────────────────────────

def fig_pairwise_jaccard(jaccard_table: Mapping[str, dict],
                         model_labels: Sequence[str],
                         out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    n = len(model_labels)
    matrix = np.eye(n)
    for key, val in jaccard_table.items():
        a, b = key.split("__vs__")
        i, j = model_labels.index(a), model_labels.index(b)
        matrix[i, j] = matrix[j, i] = val["jaccard"]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(model_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    ax.set_title("Pairwise overlap of flipped items (cert.)")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Per-item jitter distribution (App. fig) ───────────────────────────────

def fig_jitter_distribution(model_results,
                            out_path: str | Path) -> Path:
    """Histogram of per-item jitter rates across models."""
    out = _ensure_dir(Path(out_path))
    n = len(model_results)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n + 1, 2.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (label, results) in zip(axes, model_results.items()):
        rates = [r.get("baseline_jitter", {}).get("jitter_rate", 0.0)
                 for r in results
                 if r.get("baseline_jitter", {}).get("verdicts")]
        ax.hist(rates, bins=[0, 0.001, 0.34, 0.67, 1.001],
                color=PLOT_PALETTE.get(label, PLOT_PALETTE["default"]),
                edgecolor="black", lw=0.5, rwidth=0.95, alpha=0.85)
        ax.set_title(label, fontsize=9.5)
        ax.set_xticks([0, 0.33, 0.67, 1.0])
        ax.set_xticklabels(["0", "1/3", "2/3", "1"])
        ax.set_xlabel("Per-item jitter rate")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Item count")
    fig.suptitle(r"Distribution of baseline jitter ($T = 0$, 3 reruns)",
                 fontsize=10.5, y=1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Per-domain heatmap (App. fig) ─────────────────────────────────────────

def fig_domain_heatmap(per_model_domain: Mapping[str, Mapping[str, dict]],
                       out_path: str | Path,
                       min_items: int = 15) -> Path:
    """Per-domain certified-pooled Δ_flip per model.

    ``per_model_domain[label][domain] = {"n": ..., "delta_flip": ...}``.
    Cells with ``n < min_items`` are blanked.
    """
    out = _ensure_dir(Path(out_path))
    domains = sorted({d for v in per_model_domain.values() for d in v})
    labels = list(per_model_domain)
    if not domains or not labels:
        return out
    matrix = np.full((len(labels), len(domains)), np.nan, dtype=float)
    for i, m in enumerate(labels):
        for j, d in enumerate(domains):
            v = per_model_domain[m].get(d)
            if v and v.get("n", 0) >= min_items:
                matrix[i, j] = v["delta_flip"]
    fig, ax = plt.subplots(figsize=(max(7.5, 0.85 * len(domains) + 1.5),
                                    0.7 * len(labels) + 1.6))
    im = ax.imshow(matrix, cmap="OrRd", vmin=0, vmax=0.30, aspect="auto")
    short = {d: (d if len(d) <= 22 else d[:20] + "…") for d in domains}
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([short[d] for d in domains], rotation=30, ha="right",
                       fontsize=8.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(domains)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center",
                    color="white" if v > 0.18 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02,
                      label=r"Certified $\Delta_{\mathrm{flip}}$")
    cb.set_ticks([0, 0.1, 0.2, 0.3])
    cb.set_ticklabels(["0%", "10%", "20%", "30%"])
    ax.set_title(r"Per-domain certified $\Delta_{\mathrm{flip}}$ "
                 "pooled over T1, T2, T4")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Rank stability under random Dirichlet weights (App. fig) ──────────────

def fig_pis_weight_stability(cards: Mapping[str, dict],
                             out_path: str | Path,
                             n_draws: int = 2000,
                             seed: int = 7) -> Path:
    """Rank-frequency of each model under random Dirichlet PIS weights."""
    out = _ensure_dir(Path(out_path))
    labels = list(cards)
    p1, p2, p3 = [], [], []
    for label in labels:
        principles = cards[label]["principles"]
        d = principles["P1_certified_delta_flip"]["lower"]
        rdir = principles["P2_directional_ratio"]["directional_ratio"]
        urate = principles["P3_unreasonable_rate"]["unreasonable_pct"]
        p1.append(max(0.0, d))
        p2.append(1.0 - rdir)
        p3.append(urate)
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    rng = np.random.default_rng(seed)
    K = len(labels)
    rank_counts = np.zeros((K, K))
    for _ in range(n_draws):
        w = rng.dirichlet([1.0, 1.0, 1.0])
        deduction = w[0] * p1 + w[1] * p2 + w[2] * p3
        # smaller deduction = better PIS = lower rank index
        order = np.argsort(deduction)
        for rank_idx, model_idx in enumerate(order):
            rank_counts[model_idx, rank_idx] += 1
    rate = rank_counts / n_draws
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * K + 2), 3.0))
    bottom = np.zeros(K)
    colors = ["#1f9a3b", "#7ad17a", "#f4b54a", "#c45a4f"]
    rank_labels = ["Best (1)", "2", "3", f"Worst ({K})"][:K]
    for r in range(K):
        ax.bar(range(K), rate[:, r], bottom=bottom,
               color=colors[r % len(colors)], edgecolor="black", lw=0.4,
               label=rank_labels[r] if r < len(rank_labels) else f"Rank {r + 1}")
        bottom += rate[:, r]
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Probability of rank")
    ax.set_title("Rank stability under random Dirichlet weights")
    ax.legend(frameon=False, ncol=K, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ── PIS deduction decomposition ────────────────────────────────────────────

def fig_pis_decomposition(cards: Mapping[str, dict],
                          out_path: str | Path) -> Path:
    out = _ensure_dir(Path(out_path))
    labels, p1, p2, p3 = [], [], [], []
    for m, c in cards.items():
        principles = c["principles"]
        d_lo = principles["P1_certified_delta_flip"]["lower"]
        d_hi = principles["P1_certified_delta_flip"]["upper"]
        rdir = principles["P2_directional_ratio"]["directional_ratio"]
        urate = principles["P3_unreasonable_rate"]["unreasonable_pct"]
        if principles["P1_certified_delta_flip"]["has_parse_failures"]:
            for tag, dval in (("lower", d_lo), ("upper", d_hi)):
                labels.append(f"{m} ({tag})")
                p1.append(0.4 * max(0, dval))
                p2.append(0.3 * (1 - rdir))
                p3.append(0.3 * urate)
        else:
            labels.append(m)
            p1.append(0.4 * max(0, d_lo))
            p2.append(0.3 * (1 - rdir))
            p3.append(0.3 * urate)
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(labels) + 2), 3.5))
    ax.bar(xs, p1, color="#3a7d9c", label=r"$0.4\cdot \Delta_{\mathrm{flip}}^{\mathrm{cert}}$",
           edgecolor="black", lw=0.4)
    ax.bar(xs, p2, bottom=np.array(p1), color="#e1a04c",
           label=r"$0.3\cdot(1-R_{\mathrm{dir}})$", edgecolor="black", lw=0.4)
    ax.bar(xs, p3, bottom=np.array(p1) + np.array(p2), color="#c45a4f",
           label=r"$0.3\cdot U_{\mathrm{rate}}$", edgecolor="black", lw=0.4)
    totals = np.array(p1) + np.array(p2) + np.array(p3)
    for i, t in enumerate(totals):
        ax.text(i, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("PIS deduction")
    ax.set_title("PIS deduction decomposed by principle (lower is better)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
