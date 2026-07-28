#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent
DATA_DIR = Path(
    os.environ.get("ANNOTATION_DATA_DIR", DEFAULT_DATA_DIR)
).expanduser()
ANNOTATIONS_DIR = DATA_DIR / "annotations"
ANNOTATOR_IDS = ("A", "B", "C")
DIMENSION_KEYS = (
    "deontic_force",
    "policy_scope",
    "exception_set",
    "burden_of_proof",
    "default_decision",
    "risk_threshold",
)
RATINGS = ("preserved", "weakened", "broken")
OVERALL_EQUIVALENT = "Equivalent — all six dimensions are preserved"
OVERALL_NOT_EQUIVALENT = (
    "Not equivalent — at least one dimension is not preserved"
)

TASK_CONFIGS = {
    "certification-round1": {
        "annotation_name": "certified-equivalence-annotator-{annotator}.json",
        "rewrites_name": "raw_rewrites_all.json",
    },
    "certification-round2": {
        "annotation_name": (
            "certified-equivalence-annotator-{annotator}-round2.json"
        ),
        "rewrites_name": "rewrites_all.json",
    },
    "near-equivalence": {
        "annotation_name": "near-equivalence-sample-annotator-{annotator}.json",
        "rewrites_name": "raw_rewrites_all.json",
    },
}

DIMENSION_LABELS = {
    "deontic_force": (
        "Deontic force — Does the rewrite preserve the strength of the "
        "obligation, prohibition, or permission (for example, must, should, "
        "or may)?"
    ),
    "policy_scope": (
        "Policy scope — Does the rewrite cover the same set of behaviors, "
        "actors, objects, and circumstances?"
    ),
    "exception_set": (
        "Exception set — Are the same exceptions applicable, with the same "
        "triggers and limits?"
    ),
    "burden_of_proof": (
        "Burden of proof — Is the burden of justification or verification "
        "on the same party (agent, user, or third party)?"
    ),
    "default_decision": (
        "Default decision rule — Is the default SAFE/UNSAFE decision the "
        "same when a trigger or exception is uncertain?"
    ),
    "risk_threshold": (
        "Implied risk threshold — Does the rewrite preserve the implied "
        "severity or evidence required for an action to be unsafe?"
    ),
}

CODEBOOK_MARKDOWN = """
### Annotation codebook

Judge whether the rewritten policy preserves the normative meaning of the
original policy. Evaluate each dimension independently.

**Rating scale**

- **preserved** — No material semantic change on this dimension.
- **weakened** — The rewrite is materially less restrictive on this
  dimension, such as narrowing prohibited conduct, expanding an exception,
  shifting verification away from the responsible party, softening the
  unsafe default, or requiring greater risk before conduct is unsafe.
- **broken** — The dimension is materially changed in another way, including
  strengthening, reversing, removing, or adding a rule or condition.

If more than one dimension changes, rate every affected dimension. The scale
has no default choice: inspect both policies before selecting a rating. When
uncertain, choose the best-supported rating and explain the uncertainty in
the optional notes field.

After all six ratings are complete, the interface reveals the overall
question. A pair is overall equivalent only when all six dimensions are
**preserved**. The study-level certification rule is stricter still: all
three independent annotators must rate all six dimensions **preserved**.
"""


def _resolve_configuration() -> tuple[str, str, Path, Path]:
    annotator_id = os.environ.get("ANNOTATOR_ID", "A").upper()
    if annotator_id not in ANNOTATOR_IDS:
        raise ValueError("ANNOTATOR_ID must be A, B, or C")

    task = os.environ.get(
        "ANNOTATION_TASK", "certification-round1"
    ).lower()
    if task not in TASK_CONFIGS:
        choices = ", ".join(TASK_CONFIGS)
        raise ValueError(f"ANNOTATION_TASK must be one of: {choices}")

    config = TASK_CONFIGS[task]
    default_annotation = ANNOTATIONS_DIR / config["annotation_name"].format(
        annotator=annotator_id
    )
    default_rewrites = DATA_DIR / config["rewrites_name"]
    annotation_path = Path(
        os.environ.get("ANNOTATION_FILE", default_annotation)
    ).expanduser()
    rewrites_path = Path(
        os.environ.get("ANNOTATION_REWRITES_FILE", default_rewrites)
    ).expanduser()
    return annotator_id, task, annotation_path, rewrites_path


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_rewrite_index(path: Path) -> dict[str, dict[str, Any]]:
    records = _load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array")

    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Every record in {path} must be an object")
        item_id = record.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Missing item_id in {path}")
        if item_id in indexed:
            raise ValueError(f"Duplicate item_id in {path}: {item_id}")
        indexed[item_id] = record
    return indexed


def _load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    records = _load_json(path)
    if not isinstance(records, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return records


def _build_items(
    annotations: dict[str, dict[str, Any]],
    rewrite_index: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for pair_id, annotation in annotations.items():
        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation {pair_id} must be an object")
        item_id = annotation.get("item_id")
        transform_type = annotation.get("transform_type")
        if not isinstance(item_id, str) or not isinstance(
            transform_type, str
        ):
            raise ValueError(f"Missing item/transform metadata for {pair_id}")
        if pair_id != f"{item_id}_{transform_type}":
            raise ValueError(f"Pair ID does not match its metadata: {pair_id}")
        if item_id not in rewrite_index:
            raise ValueError(f"Missing rewrite item for {pair_id}")

        rewrite_record = rewrite_index[item_id]
        original_policy = rewrite_record.get("original_policy")
        transformed = rewrite_record.get("rewrites", {}).get(transform_type)
        if not isinstance(original_policy, str) or not isinstance(
            transformed, dict
        ):
            raise ValueError(f"Missing policy text for {pair_id}")
        rewritten_policy = transformed.get("rewrite")
        if not isinstance(rewritten_policy, str):
            raise ValueError(f"Missing rewritten policy for {pair_id}")

        items.append(
            {
                "pair_id": pair_id,
                "item_id": item_id,
                "transform_type": transform_type,
                "original_policy": original_policy,
                "rewritten_policy": rewritten_policy,
            }
        )
    if not items:
        raise ValueError("The annotation file contains no pairs")
    return items


ANNOTATOR_ID, ANNOTATION_TASK, ANNOTATIONS_FILE, REWRITES_FILE = (
    _resolve_configuration()
)
ANNOTATIONS = _load_annotations(ANNOTATIONS_FILE)
REWRITE_INDEX = _load_rewrite_index(REWRITES_FILE)
ITEMS = _build_items(ANNOTATIONS, REWRITE_INDEX)


def annotation_is_complete(
    annotation: Optional[dict[str, Any]],
) -> bool:
    if not isinstance(annotation, dict):
        return False
    dimensions = annotation.get("dimensions")
    return (
        isinstance(dimensions, dict)
        and all(dimensions.get(key) in RATINGS for key in DIMENSION_KEYS)
        and isinstance(annotation.get("overall_equivalent"), bool)
    )


def completed_annotation_count() -> int:
    return sum(annotation_is_complete(record) for record in ANNOTATIONS.values())


def _save_annotations() -> None:
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ANNOTATIONS_FILE.with_suffix(
        f"{ANNOTATIONS_FILE.suffix}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(ANNOTATIONS, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, ANNOTATIONS_FILE)
    finally:
        if temporary.exists():
            temporary.unlink()


def _item_at(index: float) -> tuple[int, dict[str, str]]:
    normalized = max(0, min(int(index), len(ITEMS) - 1))
    return normalized, ITEMS[normalized]


def _record_values(index: float) -> tuple[list[Optional[str]], str]:
    _, item = _item_at(index)
    record = ANNOTATIONS.get(item["pair_id"], {})
    dimensions = record.get("dimensions", {})
    ratings = [
        dimensions.get(key) if dimensions.get(key) in RATINGS else None
        for key in DIMENSION_KEYS
    ]
    notes = record.get("notes", "")
    return ratings, notes if isinstance(notes, str) else ""


def _header(index: float) -> str:
    normalized, item = _item_at(index)
    state = (
        "✓ completed"
        if annotation_is_complete(ANNOTATIONS.get(item["pair_id"]))
        else "not completed"
    )
    return (
        f"**Pair {normalized + 1}/{len(ITEMS)} — {state}**\n\n"
        "Only the policy texts are shown. Transformation class, judge output, "
        "flip status, benchmark labels, and other annotators' judgments are "
        "hidden."
    )


def _overall_from_ratings(ratings: list[str]) -> bool:
    return all(rating == "preserved" for rating in ratings)


def _save_record(
    index: float,
    ratings: list[str],
    overall_choice: str,
    notes: str,
) -> int:
    normalized, item = _item_at(index)
    if any(rating not in RATINGS for rating in ratings):
        raise ValueError(
            "Complete all six dimension ratings before submitting")
    if overall_choice not in {
        OVERALL_EQUIVALENT,
        OVERALL_NOT_EQUIVALENT,
    }:
        raise ValueError("Complete the overall-equivalence judgment")

    derived_overall = _overall_from_ratings(ratings)
    selected_overall = overall_choice == OVERALL_EQUIVALENT
    if selected_overall != derived_overall:
        required = (
            "Equivalent"
            if derived_overall
            else "Not equivalent"
        )
        raise ValueError(
            "The overall judgment must follow the six-dimensional rule: "
            f"select {required}."
        )

    previous = ANNOTATIONS.get(item["pair_id"], {})
    ANNOTATIONS[item["pair_id"]] = {
        "item_id": item["item_id"],
        "transform_type": item["transform_type"],
        "certified_expected": previous.get(
            "certified_expected",
            item["transform_type"] in {"T1", "T2", "T4"},
        ),
        "dimensions": dict(zip(DIMENSION_KEYS, ratings)),
        "overall_equivalent": derived_overall,
        "notes": notes.strip(),
    }
    _save_annotations()
    return normalized


def build_app() -> Any:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Gradio is required to launch the annotation interface. "
            "Install it with: pip install 'gradio>=4,<6'"
        ) from exc

    def render(
        index: float,
        message: Optional[str] = None,
    ) -> list[Any]:
        normalized, item = _item_at(index)
        ratings, notes = _record_values(normalized)
        status = message or (
            f"{completed_annotation_count()}/{len(ITEMS)} completed"
        )
        return [
            _header(normalized),
            item["original_policy"],
            item["rewritten_policy"],
            *ratings,
            notes,
            None,
            gr.update(visible=False),
            status,
        ]

    def reveal_overall(
        *ratings: Optional[str],
    ) -> tuple[str, Any, Optional[str]]:
        if any(rating not in RATINGS for rating in ratings):
            return (
                "Complete all six dimension ratings first.",
                gr.update(visible=False),
                None,
            )
        derived = _overall_from_ratings(list(ratings))
        choice = (
            OVERALL_EQUIVALENT
            if derived
            else OVERALL_NOT_EQUIVALENT
        )
        return (
            "Six dimensions complete. Now confirm the overall judgment.",
            gr.update(visible=True),
            choice,
        )

    def submit(
        index: float,
        *values: Optional[str],
    ) -> list[Any]:
        ratings = list(values[: len(DIMENSION_KEYS)])
        overall_choice = values[len(DIMENSION_KEYS)]
        notes = values[len(DIMENSION_KEYS) + 1] or ""
        try:
            saved_index = _save_record(
                index,
                ratings,
                overall_choice,
                notes,
            )
        except ValueError as exc:
            return [int(index), *render(index, f"Cannot save: {exc}")]

        next_index = min(saved_index + 1, len(ITEMS) - 1)
        return [
            next_index,
            *render(
                next_index,
                (
                    f"Saved. {completed_annotation_count()}/{len(ITEMS)} "
                    "completed."
                ),
            ),
        ]

    def move(index: float, offset: int) -> list[Any]:
        new_index = max(0, min(int(index) + offset, len(ITEMS) - 1))
        return [new_index, *render(new_index)]

    with gr.Blocks(
        title="Policy Rewrite Equivalence Annotation",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# Policy Rewrite Equivalence Annotation")
        gr.Markdown(
            "Compare the two policies only. Complete all six semantic "
            "dimensions before making the overall judgment."
        )
        with gr.Accordion("Codebook and instructions", open=True):
            gr.Markdown(CODEBOOK_MARKDOWN)

        with gr.Row():
            index_slider = gr.Slider(
                minimum=0,
                maximum=len(ITEMS) - 1,
                step=1,
                value=0,
                label="Pair index",
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
            )

        item_header = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original policy")
                original_box = gr.Textbox(
                    lines=18,
                    interactive=False,
                    show_label=False,
                )
            with gr.Column():
                gr.Markdown("### Rewritten policy")
                rewrite_box = gr.Textbox(
                    lines=18,
                    interactive=False,
                    show_label=False,
                )

        gr.Markdown("### Six-dimensional assessment")
        dimension_radios = [
            gr.Radio(
                choices=list(RATINGS),
                value=None,
                label=DIMENSION_LABELS[key],
            )
            for key in DIMENSION_KEYS
        ]
        notes_box = gr.Textbox(
            label="Notes (optional)",
            lines=3,
        )
        continue_button = gr.Button(
            "Continue to overall judgment",
            variant="primary",
        )

        with gr.Column(visible=False) as overall_section:
            gr.Markdown("### Overall equivalence judgment")
            overall_radio = gr.Radio(
                choices=[
                    OVERALL_EQUIVALENT,
                    OVERALL_NOT_EQUIVALENT,
                ],
                value=None,
                label=(
                    "Confirm the judgment implied by the six dimensions"
                ),
            )
            submit_button = gr.Button(
                "Submit and continue",
                variant="primary",
            )

        with gr.Row():
            previous_button = gr.Button("← Previous")
            skip_button = gr.Button("Skip →")

        render_outputs = [
            item_header,
            original_box,
            rewrite_box,
            *dimension_radios,
            notes_box,
            overall_radio,
            overall_section,
            status_text,
        ]
        move_outputs = [index_slider, *render_outputs]

        index_slider.change(
            render,
            inputs=[index_slider],
            outputs=render_outputs,
        )
        continue_button.click(
            reveal_overall,
            inputs=dimension_radios,
            outputs=[status_text, overall_section, overall_radio],
        )
        submit_button.click(
            submit,
            inputs=[
                index_slider,
                *dimension_radios,
                overall_radio,
                notes_box,
            ],
            outputs=move_outputs,
        )
        previous_button.click(
            lambda index: move(index, -1),
            inputs=[index_slider],
            outputs=move_outputs,
        )
        skip_button.click(
            lambda index: move(index, 1),
            inputs=[index_slider],
            outputs=move_outputs,
        )
        app.load(
            render,
            inputs=[index_slider],
            outputs=render_outputs,
        )
    return app


def main() -> None:
    print(f"Annotator: {ANNOTATOR_ID}")
    print(f"Task: {ANNOTATION_TASK}")
    print(f"Annotation file: {ANNOTATIONS_FILE}")
    print(f"Rewrite file: {REWRITES_FILE}")
    print(f"Loaded pairs: {len(ITEMS)}")
    print(
        f"Completed annotations: "
        f"{completed_annotation_count()}/{len(ITEMS)}"
    )
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
