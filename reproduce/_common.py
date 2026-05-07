"""Helpers shared across the reproduce scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from pij.config import RESULTS_DIR


# The four anchor judges the paper reports. Override the values via the
# ``PIJ_PAPER_MODELS`` environment variable when running with different
# providers (e.g., open-source models you serve yourself). Format:
# ``Label1=model1,Label2=model2``. Order matters for table layout.
import os as _os


def _parse_paper_models() -> dict[str, str]:
    raw = _os.environ.get("PIJ_PAPER_MODELS", "").strip()
    if not raw:
        return {
            "GPT-mini":      "gpt-5.4-mini",
            "Claude-Haiku":  "claude-haiku-4-5-20251001",
            "DeepSeek":      "DeepSeek-V3.2",
            "Gemini-Flash":  "gemini-3-flash-preview",
        }
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        label, model_id = pair.split("=", 1)
        out[label.strip()] = model_id.strip()
    return out


PAPER_MODELS = _parse_paper_models()


def _tag(model: str) -> str:
    return model.replace(".", "_").replace("-", "_")


def load_judge_results(model_id: str,
                       results_dir: Path | str = RESULTS_DIR) -> list[dict]:
    """Load ``results/judge_<tag>.json`` for the given model."""
    path = Path(results_dir) / f"judge_{_tag(model_id)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/run_judge.py --model {model_id} first."
        )
    with open(path) as f:
        return json.load(f)


def load_all_judge_results(models: Mapping[str, str] = PAPER_MODELS,
                           results_dir: Path | str = RESULTS_DIR,
                           require_all: bool = False) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for label, model_id in models.items():
        try:
            out[label] = load_judge_results(model_id, results_dir)
        except FileNotFoundError as exc:
            if require_all:
                raise
            print(f"  [skip] {label}: {exc}")
    return out


def load_strict_lenient(model_id: str,
                        results_dir: Path | str = RESULTS_DIR) -> list[dict]:
    path = Path(results_dir) / f"strict_lenient_{_tag(model_id)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/run_strict_lenient.py --model {model_id} first."
        )
    with open(path) as f:
        return json.load(f)


def load_irrelevant_context(model_id: str,
                            results_dir: Path | str = RESULTS_DIR) -> list[dict]:
    path = Path(results_dir) / f"irrelevant_context_{_tag(model_id)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/run_irrelevant_context.py --model {model_id} first."
        )
    with open(path) as f:
        return json.load(f)
