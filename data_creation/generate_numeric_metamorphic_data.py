"""Generate certified numeric-metamorphic teacher data for GSM8K and SVAMP.

This generator is intentionally separate from the historical mixed-task
``generate_relational_data.py``.  The old model roots already contain a
``generation_metadata.json`` describing the previous identity-only numeric
profile.  This module writes a separate
``numeric_metamorphic_generation_metadata.json`` while storing question JSONs
in the same conventional locations::

    <output-root>/gsm8k/questions/*.json
    <output-root>/svamp/questions/*.json

Thus the seven existing MCQ datasets are untouched, and the old GSM8K/SVAMP
identity-only pools can remain archived under a different root.

Each original question receives a total generation budget of 32 responses.
Four certified views use 8 responses each.  If a conservative transform is not
available, the same total budget is distributed as evenly as possible across
the certified views; no unsafe view is invented merely to fill a quota.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from relacats_v1.common import (
    atomic_write_json,
    batched,
    build_reasoning_prompt,
    confidence_from_logprobs,
    confidence_suffix,
    read_json,
    stable_id,
    validate_or_write_metadata,
)
from relacats_v1.core import canonicalize_answer
from relacats_v1.core.numeric_metamorphic_views import (
    NumericMetamorphicView,
    generate_numeric_metamorphic_views,
)
from relacats_v1.data_creation.dataset_adapter import (
    NUMERIC_DATASETS,
    load_numeric_examples,
)
from relacats_v1.data_creation.generate_relational_data import extract_numeric_answer


LOGGER = logging.getLogger("relacats.numeric_metamorphic_generate")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "relacats_v1/outputs/generated_data/qwen2_5_7b_instruct"
TOTAL_BUDGET = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=NUMERIC_DATASETS,
        default=list(NUMERIC_DATASETS),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--confidence-temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if not args.datasets:
        raise ValueError("at least one numeric dataset is required")
    if args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    if args.question_batch_size <= 0:
        raise ValueError("--question-batch-size must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    if args.temperature < 0 or args.confidence_temperature < 0:
        raise ValueError("temperatures must be non-negative")
    if args.max_new_tokens >= args.max_model_len:
        raise ValueError("--max-new-tokens must be smaller than --max-model-len")


def allocate_view_samples(number_of_views: int, total_budget: int = TOTAL_BUDGET) -> tuple[int, ...]:
    """Deterministically distribute a fixed response budget over certified views."""

    if number_of_views <= 0:
        raise ValueError("number_of_views must be positive")
    if total_budget < number_of_views:
        raise ValueError("total_budget must be at least the number of views")
    base, remainder = divmod(total_budget, number_of_views)
    counts = tuple(base + (1 if index < remainder else 0) for index in range(number_of_views))
    if sum(counts) != total_budget or min(counts) <= 0:
        raise AssertionError("invalid view-budget allocation")
    return counts


def _question_path(output_root: Path, example: Any) -> Path:
    filename = f"{example.source_index:06d}_{stable_id(example.question_id)}.json"
    return output_root / example.dataset_name / "questions" / filename


def _validate_existing_question(path: Path) -> bool:
    """Return True for a compatible completed checkpoint, otherwise fail loudly."""

    if not path.exists():
        return False
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Existing checkpoint is not a JSON object: {path}")
    if payload.get("relation_mode") != "numeric_metamorphic":
        raise RuntimeError(
            f"Existing numeric checkpoint at {path} is not numeric_metamorphic. "
            "Move/delete that dataset directory or use a different output root."
        )
    if int(payload.get("attempted_budget", -1)) != TOTAL_BUDGET:
        raise RuntimeError(
            f"Existing checkpoint at {path} does not have attempted_budget={TOTAL_BUDGET}"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != TOTAL_BUDGET:
        raise RuntimeError(f"Existing checkpoint has an invalid sample list: {path}")
    if any(sample.get("relation_verified") is not True for sample in samples):
        raise RuntimeError(f"Existing checkpoint contains an uncertified relation: {path}")
    return True


def _sampling_params(cls: Any, **kwargs: Any) -> Any:
    return cls(**kwargs)


def _build_prompt_items(
    examples: Sequence[Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, tuple[NumericMetamorphicView, ...]]]:
    items: list[dict[str, Any]] = []
    views_by_question: dict[str, tuple[NumericMetamorphicView, ...]] = {}
    for example in examples:
        views = generate_numeric_metamorphic_views(example.stem)
        if not views or any(not view.verify() for view in views):
            raise RuntimeError(f"{example.question_id}: uncertified metamorphic view")
        counts = allocate_view_samples(len(views))
        views_by_question[example.question_id] = views
        for view, sample_count in zip(views, counts):
            items.append(
                {
                    "example": example,
                    "view": view,
                    "sample_count": sample_count,
                    "prompt": build_reasoning_prompt(
                        tokenizer, view.transformed_question, "number"
                    ),
                }
            )
    return items, views_by_question


def generate_question_batch(
    *,
    examples: Sequence[Any],
    llm: Any,
    tokenizer: Any,
    sampling_params_cls: Any,
    model_name: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Generate and confidence-score a batch while preserving total N=32/question."""

    if not examples:
        return []
    prompt_items, views_by_question = _build_prompt_items(examples, tokenizer)

    # vLLM's SamplingParams.n is shared by a call.  Group view prompts by the
    # deterministic number of responses allocated to that view.  Four-view
    # questions normally form one n=8 group; conservative fallback profiles
    # are supported without changing the total question budget.
    outputs_by_item: dict[int, Any] = {}
    by_count: dict[int, list[int]] = {}
    for index, item in enumerate(prompt_items):
        by_count.setdefault(int(item["sample_count"]), []).append(index)

    for sample_count, item_indices in sorted(by_count.items()):
        prompts = [prompt_items[index]["prompt"] for index in item_indices]
        generation = _sampling_params(
            sampling_params_cls,
            n=sample_count,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        generated = llm.generate(prompts, generation, use_tqdm=False)
        if len(generated) != len(item_indices):
            raise RuntimeError("vLLM returned a different number of prompt outputs")
        for item_index, request_output in zip(item_indices, generated):
            outputs_by_item[item_index] = request_output

    provisional: list[dict[str, Any]] = []
    confidence_prompts: list[str] = []
    suffix = confidence_suffix(model_name)

    for item_index, item in enumerate(prompt_items):
        example = item["example"]
        view: NumericMetamorphicView = item["view"]
        prompt = str(item["prompt"])
        sample_count = int(item["sample_count"])
        request_output = outputs_by_item[item_index]
        if len(request_output.outputs) != sample_count:
            raise RuntimeError(
                f"{example.question_id}/{view.relation_id}: expected {sample_count} "
                f"responses, got {len(request_output.outputs)}"
            )

        relation_metadata = view.to_metadata()
        for sample_index, candidate in enumerate(request_output.outputs):
            response = str(candidate.text)
            extracted = extract_numeric_answer(response)
            canonical = canonicalize_answer(
                extracted,
                relation_metadata,
                answer_type="number",
            )
            record: dict[str, Any] = {
                "sample_id": stable_id(
                    example.question_id, view.relation_id, sample_index, length=24
                ),
                "question_id": example.question_id,
                "source_index": example.source_index,
                "dataset_name": example.dataset_name,
                "split": example.split,
                "relation_type": view.relation_type,
                "relation_subtype": view.relation_subtype,
                "relation_mode": view.relation_mode,
                "answer_mapping": view.answer_mapping,
                "answer_type": "number",
                "relation_id": view.relation_id,
                "view_index": int(view.relation_id.removeprefix("g")),
                "sample_index_in_view": sample_index,
                "view_sample_budget": sample_count,
                "original_question": view.original_question,
                "transformed_question": view.transformed_question,
                "original_prompt": build_reasoning_prompt(
                    tokenizer, view.original_question, "number"
                ),
                "transformed_prompt": prompt,
                "option_labels": [],
                "original_options": [],
                "transformed_options": [],
                "permutation": None,
                "inverse_permutation": None,
                "response": response,
                "gold_original_answer": example.correct_answer,
                "relation_weight": 1.0,
                "dependency_weight": 1.0,
                "finish_reason": getattr(candidate, "finish_reason", None),
                "generated_token_count": len(getattr(candidate, "token_ids", []) or []),
            }
            record.update(relation_metadata)
            record.update(canonical.to_record_fields())
            provisional.append(record)
            confidence_prompts.append(f"{prompt}{response} {suffix}")

    confidence_sampling = _sampling_params(
        sampling_params_cls,
        max_tokens=1,
        temperature=float(args.confidence_temperature),
        logprobs=20,
        seed=args.seed,
    )
    confidence_outputs = llm.generate(
        confidence_prompts, confidence_sampling, use_tqdm=False
    )
    if len(confidence_outputs) != len(provisional):
        raise RuntimeError("vLLM returned a different number of confidence outputs")

    for record, output in zip(provisional, confidence_outputs):
        candidate = output.outputs[0]
        if not candidate.logprobs or not candidate.logprobs[0]:
            raise RuntimeError(f"No confidence logprobs for {record['sample_id']}")
        yes, no, yes_found, no_found = confidence_from_logprobs(candidate.logprobs[0])
        record.update(
            {
                "confidence": yes,
                "true_prob": yes,
                "false_prob": no,
                "yes_token_found": yes_found,
                "no_token_found": no_found,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {
        example.question_id: [] for example in examples
    }
    for record in provisional:
        grouped[record["question_id"]].append(record)

    payloads: list[dict[str, Any]] = []
    for example in examples:
        samples = grouped[example.question_id]
        if len(samples) != TOTAL_BUDGET:
            raise RuntimeError(
                f"{example.question_id}: generated {len(samples)} responses, expected {TOTAL_BUDGET}"
            )
        views = views_by_question[example.question_id]
        counts = allocate_view_samples(len(views))
        view_sample_counts = {
            view.relation_id: count for view, count in zip(views, counts)
        }
        valid_count = sum(bool(sample["is_valid_answer"]) for sample in samples)
        payloads.append(
            {
                "schema_version": "relacats-v1.raw-question.1",
                "question_id": example.question_id,
                "source_index": example.source_index,
                "dataset_name": example.dataset_name,
                "split": example.split,
                "original_question": example.stem,
                "original_options": [],
                "gold_original_answer": example.correct_answer,
                "answer_type": "number",
                "relation_mode": "numeric_metamorphic",
                "answer_mapping": "identity",
                "num_views": len(views),
                "relation_subtypes": [view.relation_subtype for view in views],
                "view_sample_counts": view_sample_counts,
                "samples_per_view": counts[0] if len(set(counts)) == 1 else None,
                "attempted_budget": len(samples),
                "valid_response_count": valid_count,
                "invalid_response_count": len(samples) - valid_count,
                "all_relations_verified": all(view.verify() for view in views),
                "samples": samples,
            }
        )
    return payloads


def load_examples(args: argparse.Namespace) -> list[Any]:
    result: list[Any] = []
    for dataset_name in args.datasets:
        result.extend(
            load_numeric_examples(dataset_name, args.split, args.max_questions)
        )
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
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
        "schema_version": "relacats-v1.numeric-metamorphic-generation-config.1",
        "model_name": str(Path(args.model_name).expanduser()),
        "datasets": list(args.datasets),
        "split": args.split,
        "max_questions_per_dataset": args.max_questions,
        "total_budget": TOTAL_BUDGET,
        "preferred_profile": "4 certified views x 8 responses",
        "fallback_policy": "evenly distribute 32 over all certified views",
        "answer_mapping": "identity",
        "relation_mode": "numeric_metamorphic",
        "relation_subtypes": [
            "identity",
            "layout_wrapper",
            "number_representation",
            "equivalent_quantity",
        ],
        "temperature": args.temperature,
        "confidence_temperature": args.confidence_temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
    }
    validate_or_write_metadata(
        output_root / "numeric_metamorphic_generation_metadata.json", metadata
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

    completed = 0
    for question_batch in batched(pending, args.question_batch_size):
        payloads = generate_question_batch(
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
            atomic_write_json(_question_path(output_root, example), payload)
            completed += 1
        LOGGER.info(
            "worker %d/%d checkpoint %d/%d pending questions",
            args.shard_index,
            args.num_shards,
            completed,
            len(pending),
        )


if __name__ == "__main__":
    main()
