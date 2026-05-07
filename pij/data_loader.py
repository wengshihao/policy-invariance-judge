"""Build the unified master table from ASSEBench and R-Judge.

Schema (one row per trajectory):

    item_id              str   "asse_<id>" or "rjudge_<category>_<id>"
    dataset              str   "assebench" | "rjudge"
    case_text            str   formatted ReAct-style transcript
    gold_label_strict    int   0 = SAFE, 1 = UNSAFE (strict labelling)
    gold_label_loose     int   0 = SAFE, 1 = UNSAFE (lenient labelling)
    ambiguity            str   "clear" | "ambiguous" | "unknown"
    policy_domain        str   risk type / scenario
    harm_category        str
    text_length          int   character count

Both benchmarks are open releases; this module assumes you have unpacked
their official data archives under ``data/ASSEBench/`` and ``data/R-Judge/``.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .config import DATA_DIR


def _format_trajectory(contents: list) -> str:
    lines = []
    for conversation in contents:
        for msg in conversation:
            role = msg.get("role", "unknown").capitalize()
            if role == "Agent":
                thought = msg.get("thought", "")
                action = msg.get("action", "")
                if thought:
                    lines.append(f"Agent [thought]: {thought}")
                if action:
                    lines.append(f"Agent [action]: {action}")
            else:
                content = msg.get("content", "")
                if content:
                    lines.append(f"{role}: {content}")
    return "\n".join(lines)


def load_assebench(root: Path | None = None) -> list[dict]:
    root = Path(root) if root else DATA_DIR / "ASSEBench" / "ASSEBench" / "dataset"
    with open(root / "AgentJudge-strict.json") as f:
        strict = json.load(f)
    with open(root / "AgentJudge-loose.json") as f:
        loose = json.load(f)
    loose_by_id = {item["id"]: item for item in loose}
    out = []
    for entry in strict:
        eid = entry["id"]
        loose_entry = loose_by_id.get(eid, entry)
        case_text = _format_trajectory(entry.get("contents", []))
        is_amb = entry.get("ambiguous", 0) == 1
        out.append({
            "item_id": f"asse_{eid}",
            "dataset": "assebench",
            "case_text": case_text,
            "gold_label_strict": entry.get("label", -1),
            "gold_label_loose": loose_entry.get("label", -1),
            "ambiguity": "ambiguous" if is_amb else "clear",
            "policy_domain": entry.get("risk_type", "unknown"),
            "harm_category": entry.get("application_scenario", "unknown"),
            "risk_description": entry.get("risk_description", ""),
            "failure_mode": entry.get("failure_mode", ""),
            "text_length": len(case_text),
        })
    return out


def load_rjudge(root: Path | None = None) -> list[dict]:
    root = Path(root) if root else DATA_DIR / "R-Judge" / "data"
    out = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for json_file in sorted(category_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            for entry in data:
                eid = entry.get("id", 0)
                case_text = _format_trajectory(entry.get("contents", []))
                label = entry.get("label", -1)
                out.append({
                    "item_id": f"rjudge_{category_dir.name}_{eid}",
                    "dataset": "rjudge",
                    "case_text": case_text,
                    "gold_label_strict": label,
                    "gold_label_loose": label,
                    "ambiguity": "unknown",
                    "policy_domain": entry.get("scenario", category_dir.name),
                    "harm_category": category_dir.name,
                    "risk_description": entry.get("risk_description", ""),
                    "failure_mode": "",
                    "text_length": len(case_text),
                })
    return out


def _balanced_sample(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    safe = [x for x in pool if x["gold_label_strict"] == 0]
    unsafe = [x for x in pool if x["gold_label_strict"] == 1]
    rng.shuffle(safe)
    rng.shuffle(unsafe)
    if not safe or not unsafe:
        rng.shuffle(pool)
        return pool[:n]
    n_each = n // 2
    n_safe = min(n_each, len(safe))
    n_unsafe = min(n - n_safe, len(unsafe))
    return safe[:n_safe] + unsafe[:n_unsafe]


def sample_stratified(items: list[dict], n: int, dataset: str,
                      n_clear: Optional[int] = None,
                      n_ambiguous: Optional[int] = None,
                      seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pool = [x for x in items if x["dataset"] == dataset]
    if n_clear is not None and n_ambiguous is not None:
        clear = [x for x in pool if x["ambiguity"] == "clear"]
        amb = [x for x in pool if x["ambiguity"] == "ambiguous"]
        return (_balanced_sample(clear, n_clear, rng)
                + _balanced_sample(amb, min(n_ambiguous, len(amb)), rng))
    rng.shuffle(pool)
    return pool[:n]


def build_master_table(n_asse: int = 300, n_rjudge: int = 200,
                       seed: int = 42) -> list[dict]:
    """Build the canonical 500-item master table used in the paper."""
    asse = load_assebench()
    rjudge = load_rjudge()
    asse_sample = sample_stratified(
        asse, n_asse, "assebench",
        n_clear=n_asse // 2,
        n_ambiguous=n_asse // 2,
        seed=seed,
    )
    rjudge_sample = sample_stratified(rjudge, n_rjudge, "rjudge", seed=seed)
    return asse_sample + rjudge_sample


def save_master_table(items: list[dict], path: str | Path | None = None) -> Path:
    path = Path(path) if path else DATA_DIR / "master_table.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    return path
