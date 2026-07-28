#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = DATA_DIR / "annotations"

RAW_REWRITES_PATH = DATA_DIR / "raw_rewrites_all.json"
FINAL_REWRITES_PATH = DATA_DIR / "rewrites_all.json"

ANNOTATORS = ("A", "B", "C")
CERTIFIED_TRANSFORMS = ("T1", "T2", "T4")
DIMENSIONS = (
    "deontic_force",
    "policy_scope",
    "exception_set",
    "burden_of_proof",
    "default_decision",
    "risk_threshold",
)
RATINGS = {"preserved", "weakened", "broken"}

EXPECTED_ITEMS = 500
EXPECTED_CERTIFIED_PAIRS = EXPECTED_ITEMS * len(CERTIFIED_TRANSFORMS)
EXPECTED_FIRST_PASS_RATE = 0.97


class VerificationError(RuntimeError):
    """Raised when a released artefact fails an integrity check."""


def require(condition: bool, message: str) -> None:
    """Raise a readable verification error when ``condition`` is false."""
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> Any:
    """Load one JSON artefact and report a useful error if it is unavailable."""
    require(path.is_file(), f"Missing required file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"Invalid JSON in {path}: {exc}") from exc


def index_rewrites(path: Path) -> dict[str, dict[str, Any]]:
    """Validate and index one full rewrite-set file by item ID."""
    records = load_json(path)
    require(isinstance(records, list), f"{path} must contain a JSON array")
    require(
        len(records) == EXPECTED_ITEMS,
        f"{path} must contain {EXPECTED_ITEMS} items, found {len(records)}",
    )

    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        require(
            isinstance(record, dict),
            f"{path}: item at index {position} must be an object",
        )
        item_id = record.get("item_id")
        require(
            isinstance(item_id, str) and item_id,
            f"{path}: item at index {position} has no valid item_id",
        )
        require(item_id not in indexed, f"{path}: duplicate item_id {item_id}")
        require(
            isinstance(record.get("original_policy"), str),
            f"{path}: {item_id} has no original_policy text",
        )
        rewrites = record.get("rewrites")
        require(
            isinstance(rewrites, dict),
            f"{path}: {item_id} has no rewrites object",
        )

        for transform in CERTIFIED_TRANSFORMS:
            rewrite = rewrites.get(transform)
            pair_id = f"{item_id}_{transform}"
            require(
                isinstance(rewrite, dict),
                f"{path}: missing certified pair {pair_id}",
            )
            require(
                isinstance(rewrite.get("rewrite"), str)
                and bool(rewrite["rewrite"].strip()),
                f"{path}: {pair_id} has no rewrite text",
            )
            require(
                rewrite.get("item_id") == item_id,
                f"{path}: nested item_id mismatch for {pair_id}",
            )
            require(
                rewrite.get("transform_type") == transform,
                f"{path}: nested transform_type mismatch for {pair_id}",
            )
        indexed[item_id] = record

    return indexed


def annotation_path(annotator: str, round_two: bool = False) -> Path:
    """Resolve one released per-annotator certification file."""
    suffix = "-round2" if round_two else ""
    return (
        ANNOTATIONS_DIR
        / f"certified-equivalence-annotator-{annotator}{suffix}.json"
    )


def dimensionally_preserved(record: dict[str, Any]) -> bool:
    """Derive pair equivalence directly from the six dimensional ratings."""
    dimensions = record["dimensions"]
    return all(dimensions[dimension] == "preserved" for dimension in DIMENSIONS)


def load_annotations(
    annotator: str,
    *,
    round_two: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load and validate every record in one annotator file."""
    path = annotation_path(annotator, round_two)
    records = load_json(path)
    require(isinstance(records, dict), f"{path} must contain a JSON object")

    for pair_id, record in records.items():
        require(
            isinstance(record, dict),
            f"{path}: annotation {pair_id} must be an object",
        )
        item_id = record.get("item_id")
        transform = record.get("transform_type")
        require(
            pair_id == f"{item_id}_{transform}",
            f"{path}: pair ID does not match record metadata: {pair_id}",
        )
        require(
            transform in CERTIFIED_TRANSFORMS,
            f"{path}: unexpected transform in certified records: {pair_id}",
        )
        require(
            record.get("certified_expected") is True,
            f"{path}: certified_expected is not true for {pair_id}",
        )

        dimensions = record.get("dimensions")
        require(
            isinstance(dimensions, dict),
            f"{path}: missing dimensions for {pair_id}",
        )
        require(
            set(dimensions) == set(DIMENSIONS),
            f"{path}: incorrect dimension keys for {pair_id}",
        )
        for dimension in DIMENSIONS:
            require(
                dimensions[dimension] in RATINGS,
                f"{path}: invalid {dimension} rating for {pair_id}: "
                f"{dimensions[dimension]!r}",
            )

        derived_overall = dimensionally_preserved(record)
        require(
            isinstance(record.get("overall_equivalent"), bool),
            f"{path}: overall_equivalent must be boolean for {pair_id}",
        )
        require(
            record["overall_equivalent"] == derived_overall,
            f"{path}: overall_equivalent contradicts the six dimensions "
            f"for {pair_id}",
        )
        require(
            isinstance(record.get("notes"), str),
            f"{path}: notes must be a string for {pair_id}",
        )

    return records


def expected_pair_ids(
    rewrites: dict[str, dict[str, Any]],
) -> set[str]:
    """Return every item/transform pair that requires certification."""
    return {
        f"{item_id}_{transform}"
        for item_id in rewrites
        for transform in CERTIFIED_TRANSFORMS
    }


def unanimously_preserved(
    pair_id: str,
    annotations: dict[str, dict[str, dict[str, Any]]],
) -> bool:
    """Apply the paper's strict three-annotator, six-dimension rule."""
    return all(
        dimensionally_preserved(annotations[annotator][pair_id])
        for annotator in ANNOTATORS
    )


def changed_certified_pairs(
    raw: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
) -> set[str]:
    """Identify exactly which certified rewrite texts changed after round one."""
    changed: set[str] = set()
    for item_id in raw:
        for transform in CERTIFIED_TRANSFORMS:
            raw_text = raw[item_id]["rewrites"][transform]["rewrite"]
            final_text = final[item_id]["rewrites"][transform]["rewrite"]
            if raw_text != final_text:
                changed.add(f"{item_id}_{transform}")
    return changed


def verify_unchanged_structure(
    raw: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
) -> None:
    """Ensure raw-to-final differences are limited to replacement rewrites."""
    require(
        set(raw) == set(final),
        "raw_rewrites_all.json and rewrites_all.json have different item IDs",
    )

    for item_id in raw:
        raw_item = raw[item_id]
        final_item = final[item_id]
        require(
            raw_item["original_policy"] == final_item["original_policy"],
            f"Original policy changed between raw and final for {item_id}",
        )
        require(
            set(raw_item["rewrites"]) == set(final_item["rewrites"]),
            f"Transform set changed between raw and final for {item_id}",
        )

        for transform in raw_item["rewrites"]:
            raw_rewrite = raw_item["rewrites"][transform]
            final_rewrite = final_item["rewrites"][transform]
            pair_id = f"{item_id}_{transform}"

            raw_metadata = {
                key: value
                for key, value in raw_rewrite.items()
                if key not in {"rewrite", "validation"}
            }
            final_metadata = {
                key: value
                for key, value in final_rewrite.items()
                if key not in {"rewrite", "validation"}
            }
            require(
                raw_metadata == final_metadata,
                f"Non-text rewrite metadata changed for {pair_id}",
            )

            if transform not in CERTIFIED_TRANSFORMS:
                require(
                    raw_rewrite == final_rewrite,
                    f"Non-certified pair unexpectedly changed: {pair_id}",
                )


def verify() -> dict[str, Any]:
    """Run the complete certified-equivalence verification."""
    raw = index_rewrites(RAW_REWRITES_PATH)
    final = index_rewrites(FINAL_REWRITES_PATH)
    verify_unchanged_structure(raw, final)

    required_pairs = expected_pair_ids(raw)
    require(
        len(required_pairs) == EXPECTED_CERTIFIED_PAIRS,
        f"Expected {EXPECTED_CERTIFIED_PAIRS} certified pairs, "
        f"found {len(required_pairs)}",
    )

    round_one = {
        annotator: load_annotations(annotator)
        for annotator in ANNOTATORS
    }
    for annotator, records in round_one.items():
        require(
            set(records) == required_pairs,
            f"Annotator {annotator} first-pass records do not exactly cover "
            f"all {EXPECTED_CERTIFIED_PAIRS} certified pairs",
        )

    first_passed = {
        pair_id
        for pair_id in required_pairs
        if unanimously_preserved(pair_id, round_one)
    }
    first_failed = required_pairs - first_passed
    first_pass_rate = len(first_passed) / len(required_pairs)
    require(
        abs(first_pass_rate - EXPECTED_FIRST_PASS_RATE) < 1e-12,
        "First-pass unanimous acceptance rate does not reproduce the paper: "
        f"expected {EXPECTED_FIRST_PASS_RATE:.1%}, found "
        f"{first_pass_rate:.4%}",
    )

    changed_pairs = changed_certified_pairs(raw, final)
    require(
        changed_pairs == first_failed,
        "The raw-to-final replacements do not exactly match the failed "
        "first-pass pairs",
    )

    round_two = {
        annotator: load_annotations(annotator, round_two=True)
        for annotator in ANNOTATORS
    }
    for annotator, records in round_two.items():
        require(
            set(records) == first_failed,
            f"Annotator {annotator} round-two records do not exactly cover "
            "the failed first-pass pairs",
        )

    round_two_passed = {
        pair_id
        for pair_id in first_failed
        if unanimously_preserved(pair_id, round_two)
    }
    require(
        round_two_passed == first_failed,
        "At least one replacement rewrite failed unanimous round-two "
        "certification",
    )

    final_certified: set[str] = set()
    for pair_id in required_pairs:
        if pair_id in changed_pairs:
            if unanimously_preserved(pair_id, round_two):
                final_certified.add(pair_id)
        elif unanimously_preserved(pair_id, round_one):
            final_certified.add(pair_id)

    require(
        final_certified == required_pairs,
        "Not every final T1/T2/T4 pair has unanimous human certification",
    )

    replacements_by_transform = Counter(
        pair_id.rsplit("_", 1)[1] for pair_id in changed_pairs
    )
    return {
        "items": len(final),
        "certified_pairs": len(required_pairs),
        "first_passed": len(first_passed),
        "first_failed": len(first_failed),
        "first_pass_rate": first_pass_rate,
        "replacements_by_transform": {
            transform: replacements_by_transform[transform]
            for transform in CERTIFIED_TRANSFORMS
        },
        "round_two_passed": len(round_two_passed),
        "final_certified": len(final_certified),
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a concise, reviewer-readable verification report."""
    print("Certified-equivalence verification")
    print("==================================")
    print(f"Final rewrite items:             {report['items']}")
    print(
        "Certified transforms:            "
        + ", ".join(CERTIFIED_TRANSFORMS)
    )
    print(f"Certified item/transform pairs:  {report['certified_pairs']}")
    print(
        "First-pass unanimous acceptance: "
        f"{report['first_passed']}/{report['certified_pairs']} "
        f"({report['first_pass_rate']:.1%})"
    )
    print(f"First-pass failed pairs:         {report['first_failed']}")
    print(
        "Raw-to-final replacements:       "
        f"{sum(report['replacements_by_transform'].values())} "
        f"({', '.join(f'{key}={value}' for key, value in report['replacements_by_transform'].items())})"
    )
    print(
        "Round-two unanimous acceptance:  "
        f"{report['round_two_passed']}/{report['first_failed']}"
    )
    print(
        "Final unanimously certified:     "
        f"{report['final_certified']}/{report['certified_pairs']}"
    )
    print()
    print(
        "PASS: every final T1/T2/T4 pair is backed by unanimous "
        "six-dimensional human certification."
    )


def main() -> int:
    """CLI entry point."""
    try:
        report = verify()
    except (OSError, VerificationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
