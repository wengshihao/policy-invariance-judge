"""policy-invariance-judge: Stress-test LLM safety judges for policy invariance.

The package operationalizes the three principles defined in the paper
"Are LLM Safety Judges Policy-Invariant? A Three-Principle Stress-Test":

    1. Rubric-semantics invariance  (certified-equivalent rewrites)
    2. Rubric-threshold invariance  (strict-to-lenient switching)
    3. Ambiguity-aware calibration  (clear vs. ambiguous decomposition)

The two artifacts the paper releases live in this package:

    * ``pij.metrics.policy_invariance_score`` — the PIS scalar (Eq. 2 in the paper).
    * ``pij.judge_card.build_judge_card``     — the per-judge reporting card.

See ``README.md`` for usage and ``reproduce/`` for the paper's tables and
figures.
"""

from .metrics import policy_invariance_score, delta_flip
from .judge_card import build_judge_card

__all__ = [
    "policy_invariance_score",
    "delta_flip",
    "build_judge_card",
]
__version__ = "1.0.0"
