"""Generate 32-sample identity-only CaTS baselines for MCQ datasets.

This module complements ``generate_relational_data.py``.  The relational
pipeline intentionally uses option permutations for MCQ datasets, whereas the
CaTS baseline needed for a matched comparison is one *unchanged* MCQ prompt
sampled 32 times (32I).

The implementation reuses the existing generation/confidence-scoring helper so
prompt formatting, answer extraction, P(True) scoring, and JSON payloads stay
identical to the RelaCaTS generator.  Only the view constructor is replaced by
a single identity option view.

It is designed to coexist with the already generated GSM8K/SVAMP identity-only
folders.  Its run metadata is stored in a separate file
``mcq_identity_generation_metadata.json`` at each model root.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from relacats_v1.common import (
    atomic_write_json,
    batched,
    read_json,
    validate_or_write_metadata,
)
from relacats_v1.core.relational_views import (
    OptionPermutation,
    RelationalView,
    render_multiple_choice_question,
)
from relacats_v1.data_creation.dataset_adapter import (
    MCQ_DATASETS,
    MCQExample,
    load_mcq_examples,
)
from relacats_v1.data_creation import generate_relational_data as base


LOGGER = logging.getLogger("relacats.mcq_identity_generate")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "relacats_v1/outputs/generated_data_identity_only"
FORMAL_DATASETS = (
    "arc_easy",
    "commonsense_qa",
    "logiqa",
    "openbookqa",
    "reclor",
    "sciq",
    "winogrande",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(FORMAL_DATASETS),
        choices=MCQ_DATASETS,
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--confidence-temperature",
        type=float,
        default=0.0,
        help="Temperature for the one-token Yes/No confidence query.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--question-batch-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.datasets:
        raise ValueError("--datasets must not be empty")
    unexpected = [name for name in args.datasets if name not in FORMAL_DATASETS]
    if unexpected:
        raise ValueError(
            "This formal 32I launcher is restricted to the seven CaTS MCQ "
            f"datasets; got {unexpected}."
        )
    if args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    if args.question_batch_size <= 0:
        raise ValueError("--question-batch-size must be positive")
    if args.max_new_tokens <= 0 or args.max_model_len <= 0:
        raise ValueError("token limits must be positive")
    if args.max_new_tokens >= args.max_model_len:
        raise ValueError("--max-new-tokens must be smaller than --max-model-len")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    if args.temperature < 0 or args.confidence_temperature < 0:
        raise ValueError("temperatures must be non-negative")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")


def _identity_view(example: MCQExample, args: argparse.Namespace) -> RelationalView:
    """Construct exactly one unchanged option-MCQ view with 32 responses."""

    labels = example.labels
    permutation = OptionPermutation.identity(len(labels))
    rendered = render_multiple_choice_question(example.stem, example.options, labels)
    return RelationalView(
        relation_id="g0",
        relation_type="identity",
        original_question=rendered,
        transformed_question=rendered,
        original_options=example.options,
        transformed_options=example.options,
        option_permutation=permutation,
        samples_per_view=32,
        answer_type="option letter",
        relation_mode="identity_only",
        is_duplicate_view=False,
    )


def _identity_views_for_example(
    example: MCQExample, args: argparse.Namespace
) -> tuple[RelationalView, ...]:
    return (_identity_view(example, args),)


def _question_path(output_root: Path, example: MCQExample) -> Path:
    return base._question_path(output_root, example)


def _validate_existing_question(path: Path) -> bool:
    """Return True for a complete 32I checkpoint, fail loudly for collisions."""

    if not path.exists():
        return False
    payload = read_json(path)
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    ok = (
        isinstance(payload, dict)
        and payload.get("relation_mode") == "identity_only"
        and payload.get("answer_type") == "option letter"
        and int(payload.get("num_views", -1)) == 1
        and int(payload.get("samples_per_view", -1)) == 32
        and int(payload.get("attempted_budget", -1)) == 32
        and isinstance(samples, list)
        and len(samples) == 32
        and all(sample.get("relation_id") == "g0" for sample in samples)
        and all(sample.get("relation_type") == "identity" for sample in samples)
    )
    if not ok:
        raise RuntimeError(
            f"Existing checkpoint is not a complete MCQ 32I baseline: {path}. "
            "Move/remove the conflicting file instead of silently overwriting it."
        )
    return True


def _load_examples(args: argparse.Namespace) -> list[MCQExample]:
    examples: list[MCQExample] = []
    for dataset in args.datasets:
        examples.extend(load_mcq_examples(dataset, args.split, args.max_questions))
    return examples


def _metadata(args: argparse.Namespace, examples: Sequence[MCQExample]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.dataset_name] = counts.get(example.dataset_name, 0) + 1
    return {
        "schema_version": "relacats-v1.mcq-identity-generation-config.1",
        "model_name": str(Path(args.model_name).expanduser()),
        "datasets": list(args.datasets),
        "dataset_question_counts": counts,
        "split": args.split,
        "max_questions_per_dataset": args.max_questions,
        "relation_mode": "identity_only",
        "answer_type": "option letter",
        "num_views": 1,
        "samples_per_view": 32,
        "total_budget": 32,
        "temperature": args.temperature,
        "confidence_temperature": args.confidence_temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_root = base.resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(args)
    validate_or_write_metadata(
        output_root / "mcq_identity_generation_metadata.json",
        _metadata(args, examples),
    )

    selected = [
        example
        for global_index, example in enumerate(examples)
        if global_index % args.num_shards == args.shard_index
    ]
    pending: list[MCQExample] = []
    resumed = 0
    for example in selected:
        path = _question_path(output_root, example)
        if _validate_existing_question(path):
            resumed += 1
        else:
            pending.append(example)

    LOGGER.info(
        "worker %d/%d owns %d MCQ questions (%d pending, %d resumed); profile=1I x 32",
        args.shard_index,
        args.num_shards,
        len(selected),
        len(pending),
        resumed,
    )
    if not pending:
        return

    # The shared generation helper calls its module-level view constructor.
    # Override that one seam inside this dedicated process.  No other process
    # is affected, and all generation/confidence/canonicalization logic remains
    # the same as the ordinary RelaCaTS generator.
    base._views_for_example = _identity_views_for_example

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
    )

    completed = 0
    for question_batch in batched(pending, args.question_batch_size):
        payloads = base._generate_question_batch_uniform(
            examples=question_batch,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=SamplingParams,
            model_name=args.model_name,
            args=args,
        )
        for example, payload in zip(question_batch, payloads):
            # The shared helper derives these fields from our identity view;
            # assert them before writing so a future refactor cannot silently
            # turn a baseline run back into relational sampling.
            if payload.get("relation_mode") != "identity_only":
                raise RuntimeError("MCQ identity generator produced non-identity payload")
            if payload.get("attempted_budget") != 32 or len(payload.get("samples", [])) != 32:
                raise RuntimeError("MCQ identity generator violated the 32-response budget")
            atomic_write_json(_question_path(output_root, example), payload)
            completed += 1
        LOGGER.info(
            "worker %d/%d checkpoint %d/%d pending MCQ questions",
            args.shard_index,
            args.num_shards,
            completed,
            len(pending),
        )


if __name__ == "__main__":
    main()
