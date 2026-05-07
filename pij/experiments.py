"""High-level experiments used in the paper.

Each function writes a single JSON result file and returns the in-memory
records. Together they cover all three principles.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

from .config import RESULTS_DIR, default_judge_model
from .judge import judge_single, judge_with_rewrites
from .policies import (
    BASE_POLICY,
    IRRELEVANT_CONTEXT_POLICY,
    LENIENT_POLICY,
    STRICT_POLICY,
)


def _model_tag(model: str) -> str:
    return model.replace(".", "_").replace("-", "_")


# ── Principle 1: rubric-semantics invariance under T1..T5 ──────────────────

def run_rubric_invariance(master_table: list[dict],
                          rewrites: list[dict],
                          model: Optional[str] = None,
                          n_jitter_reruns: int = 3,
                          max_items: Optional[int] = None,
                          output_path: str | Path | None = None,
                          resume: bool = True,
                          verbose: bool = True) -> list[dict]:
    """Run jitter + 5 transforms per item; write per-item results to JSON."""
    model = model or default_judge_model()
    output_path = Path(output_path) if output_path else (
        RESULTS_DIR / f"judge_{_model_tag(model)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rewrites_by_id = {r["item_id"]: r for r in rewrites}
    items = [it for it in master_table if it["item_id"] in rewrites_by_id]
    if max_items:
        items = items[:max_items]

    existing: list[dict] = []
    seen: set[str] = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        seen = {r["item_id"] for r in existing}
        if verbose:
            print(f"  resume: {len(existing)} cached")

    todo = [it for it in items if it["item_id"] not in seen]
    results = list(existing)
    for i, item in enumerate(todo):
        if verbose:
            print(f"  [{i + 1}/{len(todo)}] {item['item_id']} ({item.get('ambiguity', '?')})")
        try:
            r = judge_with_rewrites(item, rewrites_by_id[item["item_id"]],
                                    model=model, n_jitter_reruns=n_jitter_reruns)
            r["metadata"] = {
                "dataset": item.get("dataset"),
                "ambiguity": item.get("ambiguity"),
                "gold_label_strict": item.get("gold_label_strict"),
                "gold_label_loose": item.get("gold_label_loose"),
                "policy_domain": item.get("policy_domain"),
            }
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"    ERROR: {exc}")
            continue
        if (i + 1) % 25 == 0:
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if verbose:
        print(f"  wrote {output_path} ({len(results)} items)")
    return results


# ── Principle 2: strict-to-lenient ─────────────────────────────────────────

def run_strict_lenient(master_table: list[dict],
                       model: Optional[str] = None,
                       max_items: int = 200,
                       strict_policy: str = STRICT_POLICY,
                       lenient_policy: str = LENIENT_POLICY,
                       output_path: str | Path | None = None,
                       verbose: bool = True) -> list[dict]:
    """Compare strict vs lenient policy verdicts on a subset of items."""
    model = model or default_judge_model()
    output_path = Path(output_path) if output_path else (
        RESULTS_DIR / f"strict_lenient_{_model_tag(model)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    items = master_table[:max_items]
    for i, item in enumerate(items):
        if verbose and (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(items)}]")
        case = item["case_text"]
        rs = judge_single(strict_policy, case, model=model)
        rl = judge_single(lenient_policy, case, model=model)
        flip = (rs.get("verdict") != rl.get("verdict")
                and rs.get("parse_success") and rl.get("parse_success"))
        sv = rs.get("verdict", "ERROR")
        lv = rl.get("verdict", "ERROR")
        if sv == "UNSAFE" and lv == "SAFE":
            direction = "strict_only_unsafe"
        elif sv == "SAFE" and lv == "UNSAFE":
            direction = "lenient_only_unsafe"
        else:
            direction = "both_same"
        rows.append({
            "item_id": item["item_id"],
            "model": model,
            "strict_verdict": sv,
            "lenient_verdict": lv,
            "flip": flip,
            "direction": direction,
            "metadata": {
                "dataset": item.get("dataset"),
                "ambiguity": item.get("ambiguity"),
                "gold_label_strict": item.get("gold_label_strict"),
                "gold_label_loose": item.get("gold_label_loose"),
            },
        })
        time.sleep(0.1)

    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    if verbose:
        print(f"  wrote {output_path}")
    return rows


# ── Supplementary T6: irrelevant context injection ─────────────────────────

def run_irrelevant_context(master_table: list[dict],
                           model: Optional[str] = None,
                           max_items: int = 200,
                           base_policy: str = BASE_POLICY,
                           noisy_policy: str = IRRELEVANT_CONTEXT_POLICY,
                           output_path: str | Path | None = None,
                           verbose: bool = True) -> list[dict]:
    """Compare verdicts under the base policy vs. an irrelevant-context variant."""
    model = model or default_judge_model()
    output_path = Path(output_path) if output_path else (
        RESULTS_DIR / f"irrelevant_context_{_model_tag(model)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    items = master_table[:max_items]
    for i, item in enumerate(items):
        if verbose and (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(items)}]")
        case = item["case_text"]
        ro = judge_single(base_policy, case, model=model)
        ri = judge_single(noisy_policy, case, model=model)
        flip = (ro.get("verdict") != ri.get("verdict")
                and ro.get("parse_success") and ri.get("parse_success"))
        rows.append({
            "item_id": item["item_id"],
            "model": model,
            "original_verdict": ro.get("verdict", "ERROR"),
            "irrelevant_ctx_verdict": ri.get("verdict", "ERROR"),
            "flip": flip,
            "metadata": {
                "dataset": item.get("dataset"),
                "ambiguity": item.get("ambiguity"),
            },
        })
        time.sleep(0.1)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    if verbose:
        print(f"  wrote {output_path}")
    return rows
