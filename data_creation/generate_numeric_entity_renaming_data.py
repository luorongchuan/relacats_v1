"""Generate GSM8K/SVAMP data with identity + certified entity renaming.

Preferred per-question profile:

    identity x8
    entity_renaming_1 x8
    entity_renaming_2 x8
    entity_renaming_3 x8

When no conservatively recognised person name is available, the generator
falls back to identity x32. It never substitutes number-representation or
equivalent-quantity relations merely to fill the budget.

Use a NEW output root; do not overwrite the legacy numeric-metamorphic pool.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from relacats_v1.common import atomic_write_json, batched, read_json, validate_or_write_metadata
from relacats_v1.core.numeric_metamorphic_views import generate_numeric_metamorphic_views
from relacats_v1.data_creation.dataset_adapter import NUMERIC_DATASETS, load_numeric_examples
from relacats_v1.data_creation import generate_numeric_metamorphic_data as legacy_gen

LOGGER = logging.getLogger("relacats.numeric_entity_rename_generate")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "relacats_v1/outputs/generated_data_entity_rename/qwen2_5_7b_instruct"
TOTAL_BUDGET = legacy_gen.TOTAL_BUDGET
RELATION_PROFILE = "entity_rename"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--datasets", nargs="+", choices=NUMERIC_DATASETS, default=list(NUMERIC_DATASETS))
    p.add_argument("--split", default="train")
    p.add_argument("--max-questions", type=int, default=1000)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--confidence-temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--question-batch-size", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _question_path(output_root: Path, example: Any) -> Path:
    return legacy_gen._question_path(output_root, example)


def _validate_existing_question(path: Path) -> bool:
    if not path.exists():
        return False
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Existing checkpoint is not a JSON object: {path}")
    if payload.get("relation_profile") != RELATION_PROFILE:
        raise RuntimeError(
            f"Existing checkpoint at {path} is not relation_profile={RELATION_PROFILE}. "
            "Use a fresh output root for the entity-renaming experiment."
        )
    if payload.get("relation_mode") != "numeric_metamorphic":
        raise RuntimeError(f"Unexpected relation_mode in {path}")
    if int(payload.get("attempted_budget", -1)) != TOTAL_BUDGET:
        raise RuntimeError(f"Existing checkpoint at {path} does not have N={TOTAL_BUDGET}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != TOTAL_BUDGET:
        raise RuntimeError(f"Existing checkpoint has invalid sample count: {path}")
    if any(sample.get("relation_verified") is not True for sample in samples):
        raise RuntimeError(f"Existing checkpoint contains uncertified relations: {path}")
    return True


def _entity_views(question: str):
    return generate_numeric_metamorphic_views(question, profile=RELATION_PROFILE)


def load_examples(args: argparse.Namespace) -> list[Any]:
    result: list[Any] = []
    for dataset_name in args.datasets:
        result.extend(load_numeric_examples(dataset_name, args.split, args.max_questions))
    return result


def main() -> None:
    args = parse_args()
    legacy_gen.validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_root = resolve_path(args.output_root)
    examples = load_examples(args)
    selected = [
        example
        for global_index, example in enumerate(examples)
        if global_index % args.num_shards == args.shard_index
    ]

    metadata = {
        "schema_version": "relacats-v1.numeric-entity-rename-generation-config.1",
        "model_name": str(Path(args.model_name).expanduser()),
        "datasets": list(args.datasets),
        "split": args.split,
        "max_questions_per_dataset": args.max_questions,
        "total_budget": TOTAL_BUDGET,
        "relation_profile": RELATION_PROFILE,
        "preferred_profile": "identity + 3 certified entity renamings, 8 responses/view",
        "fallback_policy": "identity x32 when no conservative person-name rename is available",
        "answer_mapping": "identity",
        "relation_mode": "numeric_metamorphic",
        "relation_subtypes": [
            "identity",
            "entity_renaming_1",
            "entity_renaming_2",
            "entity_renaming_3",
        ],
        "temperature": args.temperature,
        "confidence_temperature": args.confidence_temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
    }
    validate_or_write_metadata(
        output_root / "numeric_entity_rename_generation_metadata.json",
        metadata,
    )

    pending: list[Any] = []
    resumed = 0
    for example in selected:
        path = _question_path(output_root, example)
        if _validate_existing_question(path):
            resumed += 1
        else:
            pending.append(example)

    LOGGER.info(
        "worker %d/%d owns %d questions (%d pending, %d resumed)",
        args.shard_index,
        args.num_shards,
        len(selected),
        len(pending),
        resumed,
    )
    if not pending:
        return

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

    # Reuse the battle-tested generation/scoring path while substituting only
    # the pure relation-view constructor in its module namespace.
    original_view_builder = legacy_gen.generate_numeric_metamorphic_views
    legacy_gen.generate_numeric_metamorphic_views = _entity_views
    try:
        completed = 0
        for question_batch in batched(pending, args.question_batch_size):
            payloads = legacy_gen.generate_question_batch(
                examples=question_batch,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params_cls=SamplingParams,
                model_name=args.model_name,
                args=args,
            )
            by_question = {payload["question_id"]: payload for payload in payloads}
            for example in question_batch:
                payload = by_question[example.question_id]
                payload["relation_profile"] = RELATION_PROFILE
                atomic_write_json(_question_path(output_root, example), payload)
                completed += 1
            LOGGER.info(
                "worker %d/%d checkpoint %d/%d pending questions",
                args.shard_index,
                args.num_shards,
                completed,
                len(pending),
            )
    finally:
        legacy_gen.generate_numeric_metamorphic_views = original_view_builder


if __name__ == "__main__":
    main()
