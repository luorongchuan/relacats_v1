#!/usr/bin/env python3
"""Verify the seven MCQ identity-only CaTS baseline pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = (
    "qwen2_5_7b_instruct",
    "llama3_1_8b_instruct",
    "deepseek_r1_distill_qwen_1_5b",
)
DATASETS = (
    "arc_easy",
    "commonsense_qa",
    "logiqa",
    "openbookqa",
    "reclor",
    "sciq",
    "winogrande",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2] / "relacats_v1/outputs/generated_data_identity_only"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--expected-questions", type=int, default=1000)
    return parser.parse_args()


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return [f"invalid JSON: {exc}"]

    samples = payload.get("samples", [])
    if payload.get("relation_mode") != "identity_only":
        errors.append(f"relation_mode={payload.get('relation_mode')!r}")
    if payload.get("answer_type") != "option letter":
        errors.append(f"answer_type={payload.get('answer_type')!r}")
    if payload.get("num_views") != 1:
        errors.append(f"num_views={payload.get('num_views')!r}")
    if payload.get("samples_per_view") != 32:
        errors.append(f"samples_per_view={payload.get('samples_per_view')!r}")
    if payload.get("attempted_budget") != 32:
        errors.append(f"attempted_budget={payload.get('attempted_budget')!r}")
    if not isinstance(samples, list) or len(samples) != 32:
        errors.append(f"sample_count={len(samples) if isinstance(samples, list) else 'not-list'}")
        return errors

    if any(sample.get("relation_id") != "g0" for sample in samples):
        errors.append("non-g0 relation_id found")
    if any(sample.get("relation_type") != "identity" for sample in samples):
        errors.append("non-identity relation_type found")
    if any(sample.get("transformed_question") != sample.get("original_question") for sample in samples):
        errors.append("transformed question differs from original")
    if any(sample.get("permutation") != sample.get("inverse_permutation") for sample in samples):
        errors.append("identity permutation/inverse mismatch")
    return errors


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    failures = 0

    for model in MODELS:
        print("\n" + "=" * 100)
        print(model)
        for dataset in DATASETS:
            qdir = root / model / dataset / "questions"
            files = sorted(qdir.glob("*.json")) if qdir.is_dir() else []
            bad: list[tuple[str, list[str]]] = []
            for path in files:
                errors = validate_file(path)
                if errors:
                    bad.append((path.name, errors))
            count_ok = len(files) == args.expected_questions
            status = "OK" if count_ok and not bad else "FAIL"
            print(
                f"{dataset:16s} {status:4s} "
                f"questions={len(files)}/{args.expected_questions} bad_files={len(bad)}"
            )
            if not count_ok or bad:
                failures += 1
                for name, errors in bad[:5]:
                    print(f"  {name}: {'; '.join(errors)}")

    if failures:
        raise SystemExit(f"Verification failed for {failures} model/dataset groups")
    print("\nAll seven MCQ 32I baseline groups passed verification.")


if __name__ == "__main__":
    main()
