#!/usr/bin/env python3
"""Verify completed GSM8K/SVAMP numeric-metamorphic teacher pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path(
    "/home/luorongchuan/workspace_135/RelaCaTS/relacats_v1/outputs/generated_data"
)
DEFAULT_MODELS = (
    "qwen2_5_7b_instruct",
    "llama3_1_8b_instruct",
    "deepseek_r1_distill_qwen_1_5b",
)
EXPECTED = {"gsm8k": 1000, "svamp": 700}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    failures: list[str] = []

    for model in args.models:
        print("\n" + "#" * 100)
        print("MODEL:", model)
        for dataset, expected_count in EXPECTED.items():
            qdir = root / model / dataset / "questions"
            files = sorted(qdir.glob("*.json")) if qdir.exists() else []
            if not args.allow_partial and len(files) != expected_count:
                failures.append(
                    f"{model}/{dataset}: expected {expected_count} question files, got {len(files)}"
                )

            view_hist = Counter()
            subtype_counts = Counter()
            invalid_responses = 0
            structural_bad = 0

            for path in files:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    failures.append(f"{path}: cannot read JSON: {exc}")
                    structural_bad += 1
                    continue

                samples = payload.get("samples")
                ok = (
                    payload.get("relation_mode") == "numeric_metamorphic"
                    and payload.get("answer_mapping") == "identity"
                    and int(payload.get("attempted_budget", -1)) == 32
                    and isinstance(samples, list)
                    and len(samples) == 32
                    and payload.get("all_relations_verified") is True
                )
                if not ok:
                    failures.append(f"{path}: invalid question-level metamorphic structure")
                    structural_bad += 1
                    continue

                if any(sample.get("relation_verified") is not True for sample in samples):
                    failures.append(f"{path}: contains an uncertified sample relation")
                    structural_bad += 1

                counts = payload.get("view_sample_counts", {})
                if not isinstance(counts, dict) or sum(int(v) for v in counts.values()) != 32:
                    failures.append(f"{path}: view_sample_counts does not sum to 32")
                    structural_bad += 1

                num_views = int(payload.get("num_views", 0))
                view_hist[num_views] += 1
                subtype_counts.update(payload.get("relation_subtypes", []))
                invalid_responses += int(payload.get("invalid_response_count", 0))

            print("\n", dataset)
            print("question files:", len(files))
            print("view-count histogram:", dict(sorted(view_hist.items())))
            print("relation subtype counts:", dict(subtype_counts))
            print("invalid model responses (not a structural failure):", invalid_responses)
            print("structural failures:", structural_bad)

    if failures:
        print("\n" + "!" * 100)
        print("VERIFICATION FAILED")
        for message in failures[:50]:
            print("-", message)
        if len(failures) > 50:
            print(f"... and {len(failures) - 50} more failures")
        raise SystemExit(1)

    print("\nAll numeric-metamorphic generation structure checks passed.")


if __name__ == "__main__":
    main()
