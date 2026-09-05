"""Build and audit certified numeric metamorphic views for GSM8K and SVAMP.

This is a CPU-only pre-generation step.  It intentionally does *not* call an
LLM.  Run it before expensive teacher generation to inspect relation coverage
and spot-check transformed questions.

Example:
    python -m relacats_v1.data_creation.build_numeric_metamorphic_candidates \
        --datasets gsm8k svamp \
        --split train \
        --max-questions 1000 \
        --output-root relacats_v1/outputs/numeric_metamorphic_candidates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relacats_v1.common import atomic_write_json, atomic_write_jsonl
from relacats_v1.core.numeric_metamorphic_views import (
    generate_numeric_metamorphic_views,
)
from relacats_v1.data_creation.dataset_adapter import (
    NUMERIC_DATASETS,
    load_dataset_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "relacats_v1/outputs/numeric_metamorphic_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(NUMERIC_DATASETS),
        choices=NUMERIC_DATASETS,
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--preview-per-dataset",
        type=int,
        default=20,
        help="How many question-level transformed examples to copy into preview.jsonl.",
    )
    return parser.parse_args()


def _question_record(example: Any) -> dict[str, Any]:
    views = generate_numeric_metamorphic_views(example.stem)
    return {
        "dataset_name": example.dataset_name,
        "split": example.split,
        "source_index": example.source_index,
        "question_id": example.question_id,
        "original_question": example.stem,
        "gold_original_answer": example.correct_answer,
        "answer_type": "number",
        "relation_mode": "numeric_metamorphic",
        "num_certified_views": len(views),
        "views": [
            {
                **view.to_metadata(),
                "original_question": view.original_question,
                "transformed_question": view.transformed_question,
            }
            for view in views
        ],
    }


def main() -> None:
    args = parse_args()
    if args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    if args.preview_per_dataset < 0:
        raise ValueError("--preview-per-dataset must be non-negative")

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    overall = {
        "schema_version": "relacats-v1.numeric-metamorphic-audit.1",
        "datasets": {},
    }

    for dataset in args.datasets:
        examples = load_dataset_examples(dataset, args.split, args.max_questions)
        records = [_question_record(example) for example in examples]

        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(dataset_dir / "candidates.jsonl", records)
        atomic_write_jsonl(
            dataset_dir / "preview.jsonl",
            records[: args.preview_per_dataset],
        )

        subtype_counts: dict[str, int] = {}
        view_count_histogram: dict[str, int] = {}
        verified_failures = 0
        for record in records:
            view_count = str(record["num_certified_views"])
            view_count_histogram[view_count] = view_count_histogram.get(view_count, 0) + 1
            for view in record["views"]:
                subtype = view["relation_subtype"]
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
                if not view.get("relation_verified", False):
                    verified_failures += 1

        n = len(records)
        four_view = sum(record["num_certified_views"] == 4 for record in records)
        summary = {
            "dataset": dataset,
            "questions": n,
            "view_count_histogram": view_count_histogram,
            "relation_subtype_counts": subtype_counts,
            "four_view_questions": four_view,
            "four_view_coverage": (four_view / n) if n else 0.0,
            "certification_failures": verified_failures,
            "policy": (
                "Only deterministic reversible transforms are emitted. "
                "Missing numerical relations are skipped rather than forced."
            ),
        }
        atomic_write_json(dataset_dir / "summary.json", summary)
        overall["datasets"][dataset] = summary

        print("=" * 90)
        print(dataset)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    atomic_write_json(output_root / "summary.json", overall)
    print("=" * 90)
    print(f"Wrote numeric metamorphic audit to: {output_root}")


if __name__ == "__main__":
    main()
