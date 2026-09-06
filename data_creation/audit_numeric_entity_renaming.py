"""CPU-only coverage and safety audit for numeric entity-renaming relations.

This command loads GSM8K/SVAMP examples, constructs the proposed
``entity_rename`` profile, verifies every view, reports coverage, and writes a
small human-inspection sample. It never loads a language model or uses a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from relacats_v1.core.numeric_metamorphic_views import (
    generate_numeric_metamorphic_views,
    recognized_person_names,
)
from relacats_v1.data_creation.dataset_adapter import (
    NUMERIC_DATASETS,
    load_numeric_examples,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "relacats_v1/outputs/entity_rename_coverage"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=NUMERIC_DATASETS, default=list(NUMERIC_DATASETS))
    p.add_argument("--split", default="train")
    p.add_argument("--max-questions", type=int, default=1000)
    p.add_argument("--show-examples", type=int, default=12)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    if args.show_examples < 0:
        raise ValueError("--show-examples must be non-negative")

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    inspection_rows: list[dict[str, Any]] = []

    print("Numeric entity-renaming coverage audit")
    print("======================================")
    print("CPU-only: no model generation is performed.\n")

    for dataset in args.datasets:
        examples = load_numeric_examples(dataset, args.split, args.max_questions)
        total = len(examples)
        eligible = 0
        full_four = 0
        verified = 0
        one_name = 0
        multi_name = 0
        shown = 0

        for example in examples:
            names = recognized_person_names(example.stem)
            if len(names) == 1:
                one_name += 1
            elif len(names) >= 2:
                multi_name += 1

            views = generate_numeric_metamorphic_views(
                example.stem,
                profile="entity_rename",
            )
            if all(view.verify() for view in views):
                verified += 1
            else:
                raise RuntimeError(f"{example.question_id}: uncertified entity-renaming view")

            rename_views = [
                view for view in views
                if view.relation_subtype.startswith("entity_renaming_")
            ]
            if rename_views:
                eligible += 1
            if len(views) == 4 and len(rename_views) == 3:
                full_four += 1

            if shown < args.show_examples and rename_views:
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "question_id": example.question_id,
                    "recognized_names": json.dumps(names, ensure_ascii=False),
                    "original_question": example.stem,
                }
                for index, view in enumerate(rename_views, start=1):
                    row[f"rename_{index}_mapping"] = json.dumps(
                        dict(view.entity_mapping), ensure_ascii=False
                    )
                    row[f"rename_{index}_question"] = view.transformed_question
                inspection_rows.append(row)
                shown += 1

        summary = {
            "dataset": dataset,
            "questions": total,
            "rename_eligible_questions": eligible,
            "rename_coverage_percent": 100.0 * eligible / total if total else 0.0,
            "full_identity_plus_3_rename_questions": full_four,
            "full_four_view_percent": 100.0 * full_four / total if total else 0.0,
            "identity_only_fallback_questions": total - eligible,
            "exactly_one_recognized_name": one_name,
            "two_or_more_recognized_names": multi_name,
            "all_views_verified_questions": verified,
        }
        summaries.append(summary)

        print(
            f"[{dataset}] questions={total}  rename-eligible={eligible} "
            f"({summary['rename_coverage_percent']:.2f}%)  "
            f"full I+3R={full_four} ({summary['full_four_view_percent']:.2f}%)  "
            f"fallback-I-only={total - eligible}  verified={verified}/{total}"
        )
        print(
            f"  recognized names: exactly-one={one_name}, two-or-more={multi_name}"
        )

    _write_csv(output_dir / "entity_rename_coverage.csv", summaries)
    _write_csv(output_dir / "entity_rename_inspection.csv", inspection_rows)
    (output_dir / "entity_rename_coverage.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nOutputs:")
    print(f"  {output_dir / 'entity_rename_coverage.csv'}")
    print(f"  {output_dir / 'entity_rename_inspection.csv'}")
    print(f"  {output_dir / 'entity_rename_coverage.json'}")


if __name__ == "__main__":
    main()
