"""Centralized configuration loader.

All credentials and provider URLs are read from environment variables (or
a ``.env`` file in the project root). No keys are baked into source.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def api_key() -> str:
    key = os.environ.get("PIJ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PIJ_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export PIJ_API_KEY in the shell."
        )
    return key


def base_url() -> str:
    return os.environ.get("PIJ_BASE_URL", "https://api.openai.com/v1").strip()


def default_judge_model() -> str:
    return os.environ.get("PIJ_JUDGE_MODEL", "gpt-4o-mini").strip()


def default_rewrite_model() -> str:
    return os.environ.get("PIJ_REWRITE_MODEL", default_judge_model()).strip()


def get_client(api_key_override: Optional[str] = None,
               base_url_override: Optional[str] = None):
    """Return an OpenAI-compatible client. Imported lazily to keep the
    package importable without the ``openai`` dependency for users who
    only need the metrics."""
    from openai import OpenAI
    return OpenAI(
        api_key=api_key_override or api_key(),
        base_url=base_url_override or base_url(),
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"
