"""Strict-denominator CPU aggregation for RelaCaTS and CaTS baselines.

The response pool is never filtered at question level.  Malformed answers,
missing confidence scores, and incomplete budgets are explicitly counted and
remain in the accuracy denominator instead of silently making a dataset look
better.  Test-time relational transformations are intentionally absent in
RelaCaTS-v1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from relacats_v1.common import atomic_write_json, read_json, read_jsonl, stable_id
from relacats_v1.evaluation.method_names import (
    TABLE2_METHOD_ORDER,
    canonical_method_name,
    canonicalize_report_methods,
)


# Keep the method sets explicit.  The first seven names are the original
# Table-2 baselines; the three ``RelaCaTS-*`` names are the calibrated
# methods trained in this repository.  Older confidence artifacts may still
# contain ``CaTS-*`` labels, which are accepted by ``canonical_method_name``
# in the predictor dispatch below.
FIXED_METHODS = (
    "SC",
    "CISC",
    "Self-Certainty",
    "Best-of-N",
    "RelaCaTS-SC",
)
DYNAMIC_METHODS = (
    "RelaCaTS-ES",
    "ASC",
    "RelaCaTS-ASC",
    "ESC",
    "RASC",
)


# A confidence record produced by the current v1 pipeline only contains the
# calibrated P(Yes) score.  CISC, Self-Certainty, and RASC normally require
# additional *untrained* or token/reasoning-level scores.  We therefore expose
# the source policy in every report instead of silently presenting the
# calibrated score as an exact reproduction of those baselines.  If a caller
# supplies one of the optional fields listed here, the corresponding method is
# evaluated from that field and its status becomes ``exact``.
METHOD_SCORE_FIELDS: dict[str, tuple[str, ...]] = {
    "CISC": (
        "cisc_confidence",
        "untrained_confidence",
        "response_probability",
        "base_confidence",
        "model_confidence",
    ),
    "Self-Certainty": (
        "self_certainty",
        "self_certainty_score",
        "self_certainty_confidence",
    ),
    "RASC": (
        "rasc_score",
        "rasc_confidence",
        "sufficiency_score",
        "reasoning_quality_score",
    ),
}

# Fields that are optional in legacy confidence JSONL but required by one of
# the newly exposed baselines when available.  ``_question_groups`` keeps
# these fields instead of reducing every record to calibrated P(Yes), allowing
# exact CISC/Self-Certainty/RASC artifacts to pass through unchanged.
OPTIONAL_METHOD_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        field
        for fields in METHOD_SCORE_FIELDS.values()
        for field in fields
    )
)
OPTIONAL_METHOD_FIELDS += (
    "self_certainty_token_probabilities",
    "self_certainty_token_logprobs",
    "token_probabilities",
    "token_logprobs",
    "self_certainty_vocab_size",
    "vocab_size",
)

METHOD_METADATA: dict[str, dict[str, Any]] = {
    "SC": {
        "family": "baseline",
        "implementation_status": "exact",
        "score_source": "answer majority",
    },
    "CISC": {
        "family": "baseline",
        "implementation_status": "proxy_if_missing_field",
        "score_source": "untrained confidence field; calibrated confidence fallback",
    },
    "Self-Certainty": {
        "family": "baseline",
        "implementation_status": "exact_if_token_scores_else_proxy",
        "score_source": "self-certainty field; calibrated confidence fallback",
    },
    "Best-of-N": {
        "family": "baseline",
        "implementation_status": "exact",
        "score_source": "record confidence",
    },
    "RelaCaTS-SC": {
        "family": "RelaCaTS",
        "implementation_status": "exact",
        "score_source": "calibrated confidence",
    },
    "RelaCaTS-ES": {
        "family": "RelaCaTS",
        "implementation_status": "exact",
        "score_source": "calibrated confidence",
    },
    "ASC": {
        "family": "baseline",
        "implementation_status": "exact",
        "score_source": "answer frequency",
    },
    "RelaCaTS-ASC": {
        "family": "RelaCaTS",
        "implementation_status": "exact",
        "score_source": "calibrated confidence",
    },
    "ESC": {
        "family": "baseline",
        "implementation_status": "exact",
        "score_source": "unanimous non-overlapping windows",
    },
    "RASC": {
        "family": "baseline",
        "implementation_status": "exact_if_reasoning_score_else_proxy",
        "score_source": "reasoning/sufficiency score; calibrated confidence fallback",
    },
}


@dataclass(frozen=True)
class AggregateConfig:
    budgets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    # Match the original CaTS analysis grid (0.00, 0.01, ..., 1.00).
    thresholds: tuple[float, ...] = tuple(index / 100 for index in range(101))
    curve_max_budget: int = 32
    budget_targets: tuple[int, ...] = (16,)
    dynamic_min_valid: int = 2
    # RASC's original implementation fills a small high-quality buffer before
    # stopping.  Keep the capacity configurable while retaining a conservative
    # default for the Table-2 budget protocol.
    rasc_buffer_size: int = 5
    # ESC is parameterized by a non-overlapping window size rather than a
    # confidence threshold.  An empty tuple means “use every size from 2
    # through curve_max_budget”, which keeps the CLI backwards compatible.
    esc_window_sizes: tuple[int, ...] = ()
    # CISC normalizes confidence within each question using a temperature-
    # scaled softmax.  The CISC paper tunes this on held-out questions; expose
    # the value so a reproduction can provide the published/tuned setting.
    cisc_temperature: float = 1.0
    cisc_normalization: str = "softmax"

    def __post_init__(self) -> None:
        if not self.budgets or any(value <= 0 for value in self.budgets):
            raise ValueError("budgets must contain positive integers")
        if 16 not in self.budgets:
            raise ValueError("The fixed budgets must include 16")
        if self.curve_max_budget <= 0:
            raise ValueError("curve_max_budget must be positive")
        if not self.thresholds or any(not 0 <= value <= 1 for value in self.thresholds):
            raise ValueError("thresholds must lie in [0, 1]")
        if any(value <= 0 for value in self.budget_targets):
            raise ValueError("budget_targets must be positive")
        if self.dynamic_min_valid <= 0:
            raise ValueError("dynamic_min_valid must be positive")
        if self.rasc_buffer_size <= 0:
            raise ValueError("rasc_buffer_size must be positive")
        if any(value <= 0 for value in self.esc_window_sizes):
            raise ValueError("esc_window_sizes must contain positive integers")
        if any(value > self.curve_max_budget for value in self.esc_window_sizes):
            raise ValueError("esc_window_sizes cannot exceed curve_max_budget")
        if not math.isfinite(self.cisc_temperature) or self.cisc_temperature <= 0:
            raise ValueError("cisc_temperature must be finite and positive")
        if self.cisc_normalization not in {"softmax", "linear", "none"}:
            raise ValueError(
                "cisc_normalization must be one of: softmax, linear, none"
            )


def _normalise_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    return text.upper() if text else None


def _confidence(record: Mapping[str, Any]) -> float | None:
    value = record.get("confidence")
    return _finite_float(value)


def _finite_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for missing/malformed values."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_score(
    record: Mapping[str, Any], fields: Sequence[str]
) -> tuple[float | None, str | None]:
    """Find the first finite optional method score and its field name."""

    for field in fields:
        value = _finite_float(record.get(field))
        if value is not None:
            return value, field
    return None, None


def _self_certainty_from_tokens(record: Mapping[str, Any]) -> float | None:
    """Compute the published Self-Certainty score when token distributions exist.

    The Self-Certainty paper defines ``-(1/nV) * sum log(V p_j)`` over all
    generated positions and vocabulary entries.  Confidence artifacts made by
    early v1 runs do not retain these distributions; this helper intentionally
    returns ``None`` in that case rather than fabricating a token score.
    """

    # A producer may save either probabilities or log-probabilities.  Accept a
    # few unambiguous spellings to make the evaluator useful with exported
    # artifacts from different inference backends.
    values = record.get("self_certainty_token_probabilities")
    log_values = record.get("self_certainty_token_logprobs")
    if values is None:
        values = record.get("token_probabilities")
    if log_values is None:
        log_values = record.get("token_logprobs")

    vocab_raw = record.get("self_certainty_vocab_size", record.get("vocab_size"))
    vocab_size = None
    if vocab_raw is not None:
        try:
            candidate_vocab = int(vocab_raw)
        except (TypeError, ValueError):
            candidate_vocab = 0
        if candidate_vocab > 0:
            vocab_size = candidate_vocab

    rows: Any = values if values is not None else log_values
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    use_logs = values is None
    terms: list[float] = []
    for row in rows:
        # Each token position should contain a full-vocabulary vector.  A
        # scalar is accepted as a one-element vocabulary, which is useful for
        # compact test fixtures but still follows the same equation.
        if isinstance(row, (int, float)):
            vector = [row]
        elif isinstance(row, (list, tuple)):
            vector = list(row)
        elif isinstance(row, Mapping):
            vector = list(row.values())
        else:
            return None
        if not vector:
            return None
        vocab = vocab_size or len(vector)
        # A saved vocabulary size is a contract that each position contains
        # one probability for every vocabulary item.  A shorter vector is
        # normally a top-k/sparse export and cannot be used in the published
        # full-vocabulary equation, so fall back to the explicitly labelled
        # calibrated-confidence proxy instead of fabricating a score.
        if vocab_size is not None and len(vector) != vocab_size:
            return None
        for raw_value in vector:
            value = _finite_float(raw_value)
            if value is None:
                return None
            if use_logs:
                log_probability = value
            else:
                if value <= 0.0:
                    return None
                log_probability = math.log(value)
            # p=0 contributes -infinity and is not a useful finite score for
            # ranking; reject such a sparse/truncated export rather than
            # pretending it is a full-vocabulary Self-Certainty score.
            if not math.isfinite(log_probability):
                return None
            terms.append(-(math.log(vocab) + log_probability))
    return sum(terms) / len(terms) if terms else None


def _method_score(
    record: Mapping[str, Any], method: str
) -> tuple[float | None, str]:
    """Return a method-specific score and whether it is exact or a proxy.

    CISC uses an untrained confidence/probability, Self-Certainty uses its
    token-level certainty score, and RASC uses a reasoning/sufficiency score.
    If those fields are absent (the normal v1 confidence artifact), the only
    available score is calibrated P(Yes); using it is explicitly labelled
    ``proxy`` in the returned source string and report metadata.
    """

    canonical = canonical_method_name(method)
    if canonical == "RelaCaTS-SC" or canonical == "RelaCaTS-ES" or canonical == "RelaCaTS-ASC":
        value = _confidence(record)
        return value, "calibrated_confidence"
    if canonical == "CISC":
        value, field = _optional_score(record, METHOD_SCORE_FIELDS["CISC"])
        if value is not None:
            return value, f"exact:{field}"
        return _confidence(record), "proxy:calibrated_confidence"
    if canonical == "Self-Certainty":
        value, field = _optional_score(record, METHOD_SCORE_FIELDS["Self-Certainty"])
        if value is not None:
            return value, f"exact:{field}"
        value = _self_certainty_from_tokens(record)
        if value is not None:
            return value, "exact:token_distributions"
        return _confidence(record), "proxy:calibrated_confidence"
    if canonical == "RASC":
        value, field = _optional_score(record, METHOD_SCORE_FIELDS["RASC"])
        if value is not None:
            return value, f"exact:{field}"
        return _confidence(record), "proxy:calibrated_confidence"
    return _confidence(record), "confidence"


def _answer(record: Mapping[str, Any]) -> str | None:
    return _normalise_answer(record.get("extracted_answer", record.get("answer")))


def _winner(scores: Mapping[str, float | int]) -> str | None:
    """Argmax with CaTS-compatible first-observed tie breaking."""

    if not scores:
        return None
    return max(scores, key=scores.__getitem__)


def _majority(records: Sequence[Mapping[str, Any]]) -> str | None:
    counts: dict[str, int] = {}
    for record in records:
        answer = _answer(record)
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1
    return _winner(counts)


def _weighted_vote(records: Sequence[Mapping[str, Any]]) -> str | None:
    return _weighted_vote_with(records, lambda record: _confidence(record))


def _weighted_vote_with(
    records: Sequence[Mapping[str, Any]],
    score_getter: Callable[[Mapping[str, Any]], float | None],
) -> str | None:
    scores: dict[str, float] = {}
    for record in records:
        answer = _answer(record)
        score = score_getter(record)
        if answer is None or score is None:
            continue
        scores[answer] = scores.get(answer, 0.0) + score
    return _winner(scores)


def _cisc_vote(
    records: Sequence[Mapping[str, Any]],
    temperature: float = 1.0,
    normalization: str = "softmax",
) -> str | None:
    """Confidence-informed self-consistency (CISC).

    The released CISC implementation applies a temperature-scaled softmax to
    per-response confidence and sums utilities by answer.  Multiplying all
    utilities by a common normalization is unnecessary for the argmax, so we
    use a numerically stable shifted exponential here.  ``linear`` and
    ``none`` are accepted for compatibility with the released CISC utility;
    the paper's recommended/default protocol is ``softmax``.  Missing scores
    are skipped; if every optional score is absent ``_method_score`` provides
    the explicitly documented calibrated-confidence proxy.
    """

    candidates: list[tuple[str, float]] = []
    for record in records:
        answer = _answer(record)
        score, _ = _method_score(record, "CISC")
        if answer is not None and score is not None:
            candidates.append((answer, score))
    if not candidates:
        return None
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("CISC temperature must be finite and positive")
    if normalization == "softmax":
        # CISC scores are probabilities in the original protocol.  For
        # exported logits/certainty values, softmax remains well-defined and
        # preserves the intended confidence ordering.
        maximum = max(score for _, score in candidates)
        weights = [
            (answer, math.exp((score - maximum) / temperature))
            for answer, score in candidates
        ]
    elif normalization == "linear":
        minimum = min(score for _, score in candidates)
        weights = [
            (answer, temperature * (score - minimum) + 1.0)
            for answer, score in candidates
        ]
    elif normalization == "none":
        weights = list(candidates)
    else:
        raise ValueError(
            "CISC normalization must be one of: softmax, linear, none"
        )
    totals: dict[str, float] = {}
    for answer, weight in weights:
        totals[answer] = totals.get(answer, 0.0) + weight
    return _winner(totals)


def _self_certainty_vote(records: Sequence[Mapping[str, Any]]) -> str | None:
    candidates = [
        record
        for record in records
        if _answer(record) is not None and _method_score(record, "Self-Certainty")[0] is not None
    ]
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda record: _method_score(record, "Self-Certainty")[0],
    )
    return _answer(selected)


def _fixed_predict(
    method: str,
    records: Sequence[Mapping[str, Any]],
    budget: int,
    cisc_temperature: float = 1.0,
    cisc_normalization: str = "softmax",
) -> tuple[str | None, int, str]:
    if len(records) < budget:
        return None, len(records), "insufficient_samples"
    prefix = records[:budget]
    method = canonical_method_name(method)
    if method == "SC":
        prediction = _majority(prefix)
    elif method == "CISC":
        prediction = _cisc_vote(prefix, cisc_temperature, cisc_normalization)
    elif method == "Self-Certainty":
        prediction = _self_certainty_vote(prefix)
    elif method == "Best-of-N":
        candidates = [
            record
            for record in prefix
            if _answer(record) is not None and _confidence(record) is not None
        ]
        prediction = (
            _answer(max(candidates, key=lambda item: _confidence(item)))
            if candidates
            else None
        )
    elif method == "RelaCaTS-SC":
        prediction = _weighted_vote_with(
            prefix, lambda record: _method_score(record, "RelaCaTS-SC")[0]
        )
    else:
        raise ValueError(f"Unknown fixed method: {method}")
    return prediction, budget, "ok" if prediction is not None else "no_valid_answer"


def _dynamic_predict(
    method: str,
    records: Sequence[Mapping[str, Any]],
    threshold: float,
    max_budget: int,
    min_valid: int,
    rasc_buffer_size: int = 5,
) -> tuple[str | None, int, str]:
    method = canonical_method_name(method)
    prefix = records[:max_budget]
    if method == "ASC":
        scores: dict[str, int] = {}
        valid = 0
        for index, record in enumerate(prefix):
            answer = _answer(record)
            if answer is None:
                continue
            scores[answer] = scores.get(answer, 0) + 1
            valid += 1
            if valid >= min_valid:
                leader = _winner(scores)
                ratio = scores[leader] / valid if leader is not None else 0.0
                if ratio >= threshold:
                    return leader, index + 1, "early_stop"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _winner(scores)
    elif method == "RelaCaTS-ES":
        for index, record in enumerate(prefix):
            answer = _answer(record)
            confidence = _method_score(record, "RelaCaTS-ES")[0]
            if answer is not None and confidence is not None and confidence >= threshold:
                return answer, index + 1, "early_stop"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _weighted_vote_with(
            prefix, lambda record: _method_score(record, "RelaCaTS-ES")[0]
        )
    elif method == "RelaCaTS-ASC":
        scores_float: dict[str, float] = {}
        valid = 0
        for index, record in enumerate(prefix):
            answer = _answer(record)
            confidence = _method_score(record, "RelaCaTS-ASC")[0]
            if answer is None or confidence is None:
                continue
            scores_float[answer] = scores_float.get(answer, 0.0) + confidence
            valid += 1
            if valid >= min_valid:
                total = sum(scores_float.values())
                leader = _winner(scores_float)
                ratio = (
                    scores_float[leader] / total
                    if leader is not None and total > 0
                    else 0.0
                )
                if ratio >= threshold:
                    return leader, index + 1, "early_stop"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _winner(scores_float)
    elif method == "ESC":
        # ESC (Early-Stopping Self-Consistency) samples complete, sequential,
        # non-overlapping windows.  A window can trigger stopping only when
        # every response in it has the same valid answer.  If no window is
        # unanimous, vote over the complete windows and never charge a
        # trailing partial window.
        window_size = int(round(threshold))
        if window_size < 2:
            # Backwards-compatible callers may still pass a normalized value
            # in [0, 1].  Map it to a useful window while making the emitted
            # curves use integer sizes (see ``evaluate_records``).
            window_size = max(2, int(round(float(threshold) * max_budget)))
        window_size = min(window_size, max_budget)
        usable = (len(prefix) // window_size) * window_size
        sampled = prefix[:usable]
        for start in range(0, usable, window_size):
            window = sampled[start : start + window_size]
            answers = [_answer(record) for record in window]
            if answers and all(answer is not None for answer in answers) and len(set(answers)) == 1:
                return answers[0], start + window_size, "early_stop"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _majority(sampled)
        used = usable
        return prediction, used, "ok" if prediction is not None else "no_valid_answer"
    elif method == "RASC":
        # RASC uses a high-quality buffer: retain responses whose reasoning /
        # sufficiency score clears the threshold, stop when the buffer reaches
        # capacity, then vote among buffered answers.  The current v1 records
        # do not include RASC's learned feature score, so ``_method_score``
        # falls back to calibrated confidence and the report marks this as a
        # proxy.  Artifacts carrying ``rasc_score`` (or an accepted alias) use
        # the exact same control flow with that score.
        capacity = max(1, int(rasc_buffer_size))
        buffer: list[Mapping[str, Any]] = []
        for index, record in enumerate(prefix):
            answer = _answer(record)
            score = _method_score(record, "RASC")[0]
            if answer is None or score is None or score < threshold:
                continue
            buffer.append(record)
            if len(buffer) >= capacity:
                prediction = _weighted_vote_with(
                    buffer, lambda item: _method_score(item, "RASC")[0]
                )
                return prediction, index + 1, "early_stop" if prediction is not None else "no_valid_answer"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _weighted_vote_with(
            prefix, lambda item: _method_score(item, "RASC")[0]
        )
    else:
        raise ValueError(f"Unknown dynamic method: {method}")
    return prediction, max_budget, "ok" if prediction is not None else "no_valid_answer"


def _question_groups(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str | None], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold_by_question: dict[str, str | None] = {}
    seen_samples: dict[str, tuple[Any, ...]] = {}
    invalid_answers = 0
    missing_confidence = 0
    yes_missing = 0
    total_input = 0

    for row_index, raw in enumerate(records):
        total_input += 1
        question_id = str(
            raw.get("question_id")
            or f"legacy:{stable_id(raw.get('prompt', ''), length=20)}"
        )
        generation_index = int(raw.get("generation_index", len(grouped[question_id])))
        sample_id = str(
            raw.get("sample_id")
            or stable_id(question_id, generation_index, row_index, length=24)
        )
        slim = {
            "sample_id": sample_id,
            "question_id": question_id,
            "generation_index": generation_index,
            "dataset_name": raw.get("dataset_name"),
            "correct_answer": raw.get("correct_answer"),
            "extracted_answer": raw.get("extracted_answer", raw.get("answer")),
            "confidence": raw.get("confidence"),
            "confidence_valid": raw.get("confidence_valid"),
            "yes_token_found_top20": raw.get("yes_token_found_top20"),
        }
        for field in OPTIONAL_METHOD_FIELDS:
            if field in raw:
                slim[field] = raw[field]
        signature = (
            question_id,
            generation_index,
            _normalise_answer(slim["correct_answer"]),
            _normalise_answer(slim["extracted_answer"]),
            _confidence(slim),
            tuple(
                (field, repr(slim.get(field)))
                for field in OPTIONAL_METHOD_FIELDS
                if field in slim
            ),
        )
        if sample_id in seen_samples:
            if seen_samples[sample_id] != signature:
                raise ValueError(f"Conflicting duplicate sample_id: {sample_id}")
            continue
        seen_samples[sample_id] = signature
        gold = _normalise_answer(slim["correct_answer"])
        if question_id in gold_by_question and gold_by_question[question_id] != gold:
            raise ValueError(f"Conflicting gold answers for question {question_id}")
        gold_by_question[question_id] = gold
        grouped[question_id].append(slim)
        if _answer(slim) is None:
            invalid_answers += 1
        if _confidence(slim) is None:
            missing_confidence += 1
        if slim.get("yes_token_found_top20") is False:
            yes_missing += 1

    duplicate_generation_indices = 0
    for question_id, question_records in grouped.items():
        question_records.sort(key=lambda item: (item["generation_index"], item["sample_id"]))
        indices = [record["generation_index"] for record in question_records]
        duplicate_generation_indices += len(indices) - len(set(indices))
    diagnostics = {
        "input_records": total_input,
        "unique_samples": len(seen_samples),
        "duplicate_samples_ignored": total_input - len(seen_samples),
        "duplicate_generation_indices": duplicate_generation_indices,
        "invalid_extracted_answers": invalid_answers,
        "missing_or_nonfinite_confidence": missing_confidence,
        "yes_token_missing_from_top20": yes_missing,
    }
    return dict(grouped), gold_by_question, diagnostics


Predictor = Callable[[Sequence[Mapping[str, Any]]], tuple[str | None, int, str]]


def _score_method(
    question_ids: Sequence[str],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_by_question: Mapping[str, str | None],
    predictor: Predictor,
) -> dict[str, Any]:
    correct = 0
    invalid = 0
    insufficient = 0
    invalid_gold = 0
    early_stops = 0
    used_total = 0
    used_observed_total = 0
    observed_questions = 0
    for question_id in question_ids:
        question_records = grouped.get(question_id, ())
        gold = gold_by_question.get(question_id)
        prediction, used, status = predictor(question_records)
        used_total += used
        if question_id in grouped:
            observed_questions += 1
            used_observed_total += used
        if gold is None:
            invalid_gold += 1
        if prediction is None:
            invalid += 1
        if status == "insufficient_samples":
            insufficient += 1
        if status == "early_stop":
            early_stops += 1
        if prediction is not None and gold is not None and prediction == gold:
            correct += 1
    total = len(question_ids)
    return {
        "questions_total": total,
        "questions_observed": observed_questions,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100.0 * correct / total if total else 0.0,
        "avg_samples_used": used_total / total if total else 0.0,
        "avg_samples_used_observed": (
            used_observed_total / observed_questions if observed_questions else 0.0
        ),
        "invalid_predictions": invalid,
        "insufficient_sample_questions": insufficient,
        "invalid_gold_questions": invalid_gold,
        "early_stop_questions": early_stops,
    }


def _method_runtime_metadata(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """Annotate whether optional baseline scores were present in the input.

    This is deliberately data-driven: a report made from the ordinary v1
    confidence JSONL will say ``proxy`` for CISC/Self-Certainty/RASC, while a
    richer artifact carrying native fields is marked ``exact``.  Consumers can
    therefore distinguish a genuine baseline reproduction from a fair,
    explicitly labelled fallback run.
    """

    rows = [record for records in grouped.values() for record in records]
    metadata: dict[str, dict[str, Any]] = {
        method: dict(values) for method, values in METHOD_METADATA.items()
    }
    for method in ("CISC", "Self-Certainty", "RASC"):
        native_fields: set[str] = set()
        native_count = 0
        for record in rows:
            if method == "Self-Certainty":
                scalar_fields = METHOD_SCORE_FIELDS[method]
                scalar_value, scalar_field = _optional_score(record, scalar_fields)
                token_value = _self_certainty_from_tokens(record)
                if scalar_value is not None and scalar_field is not None:
                    native_fields.add(scalar_field)
                    native_count += 1
                elif token_value is not None:
                    native_fields.add("token_distributions")
                    native_count += 1
            else:
                value, field = _optional_score(record, METHOD_SCORE_FIELDS[method])
                if value is not None and field is not None:
                    native_fields.add(field)
                    native_count += 1
        metadata[method]["native_score_records"] = native_count
        metadata[method]["native_score_fields"] = sorted(native_fields)
        metadata[method]["implementation_status"] = (
            "exact" if native_count else "proxy"
        )
    return metadata


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    config: AggregateConfig | None = None,
    expected_question_ids: Sequence[str] | None = None,
    expected_question_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate confidence records entirely on CPU.

    ``expected_question_ids`` is the strongest strict-denominator contract.  If
    only a count is known, missing questions are represented as explicit empty
    questions and therefore score as incorrect rather than disappearing.
    """

    config = config or AggregateConfig()
    grouped, gold_by_question, diagnostics = _question_groups(records)
    observed_ids = sorted(grouped)
    if expected_question_ids is None:
        question_ids = list(observed_ids)
    else:
        question_ids = list(dict.fromkeys(str(item) for item in expected_question_ids))
        unexpected = sorted(set(observed_ids) - set(question_ids))
        if unexpected:
            raise ValueError(
                f"Found {len(unexpected)} question IDs outside expected_question_ids"
            )
    if expected_question_count is not None:
        if expected_question_count < len(question_ids):
            raise ValueError(
                f"expected_question_count={expected_question_count} is smaller than "
                f"the {len(question_ids)} known questions"
            )
        missing_count = expected_question_count - len(question_ids)
        question_ids.extend(
            f"__missing_question_{index:08d}" for index in range(missing_count)
        )

    sample_counts = [len(grouped[question_id]) for question_id in observed_ids]
    diagnostics.update(
        {
            "questions_total_denominator": len(question_ids),
            "questions_observed": len(observed_ids),
            "questions_missing_entirely": len(question_ids) - len(observed_ids),
            "questions_without_valid_answer": sum(
                1
                for question_id in observed_ids
                if not any(_answer(record) is not None for record in grouped[question_id])
            ),
            "min_samples_per_observed_question": min(sample_counts) if sample_counts else 0,
            "max_samples_per_observed_question": max(sample_counts) if sample_counts else 0,
        }
    )

    fixed_rows: list[dict[str, Any]] = []
    for budget in config.budgets:
        for method in FIXED_METHODS:
            metrics = _score_method(
                question_ids,
                grouped,
                gold_by_question,
                lambda rows, method=method, budget=budget: _fixed_predict(
                    method,
                    rows,
                    budget,
                    config.cisc_temperature,
                    config.cisc_normalization,
                ),
            )
            fixed_rows.append(
                {
                    "row_type": "fixed_budget",
                    "method": method,
                    "budget": budget,
                    "budget_cap": budget,
                    "threshold": None,
                    **metrics,
                }
            )

    curves: dict[str, list[dict[str, Any]]] = {method: [] for method in DYNAMIC_METHODS}
    for method in DYNAMIC_METHODS:
        if method == "ESC":
            esc_parameters = config.esc_window_sizes or tuple(
                range(2, config.curve_max_budget + 1)
            )
            parameters: Sequence[float | int] = esc_parameters
        else:
            parameters = config.thresholds
        for threshold in parameters:
            metrics = _score_method(
                question_ids,
                grouped,
                gold_by_question,
                lambda rows, method=method, threshold=threshold: _dynamic_predict(
                    method,
                    rows,
                    threshold,
                    config.curve_max_budget,
                    config.dynamic_min_valid,
                    config.rasc_buffer_size,
                ),
            )
            curves[method].append(
                {
                    "row_type": "threshold_curve",
                    "method": method,
                    "budget": None,
                    "budget_cap": config.curve_max_budget,
                    "threshold": threshold,
                    # Keep ``threshold`` populated for backwards-compatible
                    # consumers, but expose ESC's actual integer control
                    # parameter explicitly instead of making readers infer it
                    # from a float column.
                    "parameter_type": (
                        "window_size" if method == "ESC" else "confidence_threshold"
                    ),
                    "window_size": (
                        int(threshold) if method == "ESC" else None
                    ),
                    **metrics,
                }
            )

    # Map a dynamic curve to a requested *average* budget without choosing the
    # threshold by test accuracy.  This avoids hidden oracle tuning.
    budget_matches: list[dict[str, Any]] = []
    for target in config.budget_targets:
        for method in DYNAMIC_METHODS:
            selected = min(
                curves[method],
                key=lambda row: (
                    abs(row["avg_samples_used"] - target),
                    float(
                        row.get("window_size")
                        if row.get("parameter_type") == "window_size"
                        else row.get("threshold", 0.0)
                    ),
                ),
            )
            budget_matches.append(
                {
                    **selected,
                    "row_type": "dynamic_budget_match",
                    "budget": target,
                    "budget_target": target,
                    "budget_gap": selected["avg_samples_used"] - target,
                    "selection_rule": "closest average samples; accuracy not used",
                }
            )

    return {
        "schema_version": 1,
        # Keep the report namespace and display order explicit.  This avoids
        # downstream table builders having to infer whether a ``CaTS-*``
        # spelling refers to an original baseline or to one of the trained
        # RelaCaTS methods.
        "evaluation_namespace": "RelaCaTS",
        "method_order": list(TABLE2_METHOD_ORDER),
        "protocol": "RelaCaTS-v1 uses original CaTS evaluation without relational views",
        "config": asdict(config),
        "diagnostics": diagnostics,
        "method_metadata": _method_runtime_metadata(grouped),
        "fixed_budget_results": fixed_rows,
        "threshold_curves": curves,
        "dynamic_budget_matches": budget_matches,
    }


def _discover_confidence_files(inputs: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_file():
            files.append(path.resolve())
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        direct = path / "confidence.jsonl"
        if direct.is_file():
            files.append(direct.resolve())
            continue
        merged = sorted(path.rglob("confidence.jsonl"))
        if merged:
            files.extend(item.resolve() for item in merged)
            continue
        chunks = sorted(path.rglob("chunks/chunk-*.jsonl"))
        if chunks:
            raise ValueError(
                f"Confidence chunks exist below {path}, but no merged confidence.jsonl; "
                "the confidence stage did not finish, so refusing a partial report"
            )
        raise FileNotFoundError(f"No confidence JSONL artifacts below {path}")
    unique = list(dict.fromkeys(files))
    if not unique:
        raise ValueError("No confidence inputs provided")
    _validate_confidence_manifests(unique)
    return unique


def _validate_confidence_manifests(files: Sequence[Path]) -> None:
    """Reject incomplete or partially supplied confidence shard sets.

    A report made from one of two GPU shards would otherwise have a deceptively
    high accuracy because absent questions disappear from the denominator.  The
    confidence stage writes a manifest for every shard, so validate it before
    any CPU aggregation.  Legacy single-shard files without metadata remain
    supported.
    """

    shard_groups: dict[Path, tuple[int, set[int]]] = {}
    for path in files:
        artifact_dir = path.parent
        manifest_path = artifact_dir / "confidence_manifest.json"
        metadata_path = artifact_dir / "confidence_metadata.json"
        if not manifest_path.exists() and not metadata_path.exists():
            continue
        if not manifest_path.exists() or not metadata_path.exists():
            raise ValueError(
                f"Incomplete confidence artifact metadata beside {path}; "
                "rerun confidence calculation or provide a new output directory"
            )
        try:
            manifest = read_json(manifest_path)
            metadata = read_json(metadata_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid confidence metadata beside {path}") from exc
        if not isinstance(manifest, dict) or not isinstance(metadata, dict):
            raise ValueError(f"Confidence metadata must be JSON objects: {path}")
        if manifest.get("complete") is not True:
            raise ValueError(f"Confidence manifest is incomplete: {manifest_path}")
        expected = manifest.get("expected_samples")
        actual = manifest.get("samples")
        if expected is not None and actual != expected:
            raise ValueError(
                f"Confidence sample count mismatch in {manifest_path}: "
                f"{actual} != {expected}"
            )
        # Do not trust a stale manifest alone: a truncated JSONL with an old
        # manifest would otherwise make missing questions disappear from the
        # denominator.  Recompute the count, question set, and producer digest.
        observed_samples = 0
        observed_questions: set[str] = set()
        observed_digest = hashlib.sha256()
        observed_sample_ids: set[str] = set()
        try:
            for record in read_jsonl(path):
                sample_id = str(record.get("sample_id", ""))
                if not sample_id:
                    raise ValueError(f"record without sample_id in {path}")
                if sample_id in observed_sample_ids:
                    raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
                observed_sample_ids.add(sample_id)
                question_id = str(record.get("question_id", ""))
                if not question_id:
                    raise ValueError(f"record without question_id in {path}")
                observed_questions.add(question_id)
                observed_digest.update(sample_id.encode("utf-8"))
                observed_digest.update(b"\0")
                observed_samples += 1
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid confidence JSONL: {path}") from exc
        if actual is not None and observed_samples != actual:
            raise ValueError(
                f"Confidence file/manifest count mismatch in {path}: "
                f"{observed_samples} != {actual}"
            )
        if manifest.get("questions") is not None and len(observed_questions) != manifest["questions"]:
            raise ValueError(
                f"Confidence question count mismatch in {path}: "
                f"{len(observed_questions)} != {manifest['questions']}"
            )
        expected_questions = manifest.get("expected_questions")
        if (
            expected_questions is not None
            and manifest.get("questions") is not None
            and int(expected_questions) != int(manifest["questions"])
        ):
            raise ValueError(
                f"Confidence expected question count mismatch in {manifest_path}: "
                f"{manifest['questions']} != {expected_questions}"
            )
        digest = manifest.get("sample_id_sha256")
        if digest is not None and observed_digest.hexdigest() != digest:
            raise ValueError(f"Confidence sample digest mismatch in {path}")
        try:
            num_shards = int(metadata.get("num_shards", 1))
            shard_index = int(metadata.get("shard_index", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid shard metadata in {metadata_path}") from exc
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(f"Invalid shard identity in {metadata_path}")
        # The parent of shard-xxxxx-of-yyyyy is the natural group.  For a
        # single artifact generated with num_shards=1, grouping by itself
        # avoids imposing a sibling directory convention on legacy outputs.
        group = artifact_dir.parent.resolve() if num_shards > 1 else artifact_dir.resolve()
        if group in shard_groups:
            previous_n, indices = shard_groups[group]
            if previous_n != num_shards:
                raise ValueError(f"Conflicting num_shards under {group}")
            indices.add(shard_index)
        else:
            shard_groups[group] = (num_shards, {shard_index})

    for group, (num_shards, indices) in shard_groups.items():
        if num_shards > 1 and indices != set(range(num_shards)):
            missing = sorted(set(range(num_shards)) - indices)
            raise ValueError(
                f"Missing confidence shards under {group}: {missing}; "
                "do not report a partial denominator"
            )


def _manifest_expected_questions(files: Sequence[Path]) -> int | None:
    """Infer a strict question denominator from complete confidence shards."""

    groups: dict[Path, dict[str, Any]] = {}
    for path in files:
        artifact_dir = path.parent
        manifest_path = artifact_dir / "confidence_manifest.json"
        metadata_path = artifact_dir / "confidence_metadata.json"
        if not manifest_path.is_file() or not metadata_path.is_file():
            return None
        manifest = read_json(manifest_path)
        metadata = read_json(metadata_path)
        if not isinstance(manifest, dict) or not isinstance(metadata, dict):
            return None
        expected = manifest.get("expected_questions")
        if expected is None:
            return None
        num_shards = int(metadata.get("num_shards", 1))
        shard_index = int(metadata.get("shard_index", 0))
        group = artifact_dir.parent.resolve() if num_shards > 1 else artifact_dir.resolve()
        state = groups.setdefault(
            group,
            {
                "num_shards": num_shards,
                "indices": set(),
                "expected": {},
                "disjoint": True,
            },
        )
        state["indices"].add(shard_index)
        state["expected"][shard_index] = int(expected)
        state["disjoint"] = state["disjoint"] and bool(
            metadata.get("responses_already_sharded", False)
        )

    if not groups:
        return None
    total = 0
    for state in groups.values():
        expected_values = list(state["expected"].values())
        if state["num_shards"] > 1 and state["disjoint"]:
            total += sum(expected_values)
        else:
            # In the legacy modulo-by-sample protocol every confidence shard
            # contains every question, so the maximum is the correct count.
            total += max(expected_values)
    return total


def _iter_files(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        yield from read_jsonl(path)


def _flat_rows(report: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    yield from report["fixed_budget_results"]
    for rows in report["threshold_curves"].values():
        yield from rows
    yield from report["dynamic_budget_matches"]


CSV_COLUMNS = (
    "row_type",
    "method",
    "budget",
    "budget_target",
    "budget_cap",
    "threshold",
    "parameter_type",
    "window_size",
    "questions_total",
    "questions_observed",
    "correct",
    "accuracy",
    "accuracy_percent",
    "avg_samples_used",
    "avg_samples_used_observed",
    "invalid_predictions",
    "insufficient_sample_questions",
    "invalid_gold_questions",
    "early_stop_questions",
    "budget_gap",
    "selection_rule",
)


def _markdown(report: Mapping[str, Any]) -> str:
    diagnostics = report["diagnostics"]
    lines = [
        "# RelaCaTS-v1 evaluation",
        "",
        "Test-time relational transformations: **disabled** (original CaTS protocol).",
        "Method labels: original baselines (`SC`, `CISC`, `Self-Certainty`, `Best-of-N`, `ASC`, `ESC`, `RASC`) plus `RelaCaTS-SC`, `RelaCaTS-ES`, and `RelaCaTS-ASC`.",
        "",
        "## Coverage and invalid outputs",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    for key in (
        "questions_total_denominator",
        "questions_observed",
        "questions_missing_entirely",
        "unique_samples",
        "invalid_extracted_answers",
        "missing_or_nonfinite_confidence",
        "yes_token_missing_from_top20",
        "questions_without_valid_answer",
        "duplicate_samples_ignored",
        "duplicate_generation_indices",
    ):
        lines.append(f"| {key} | {diagnostics.get(key, 0)} |")

    lines.extend(
        [
            "",
            "Accuracy always uses `questions_total_denominator`; invalid or missing "
            "questions are not dropped.",
            "",
        "## Fixed-budget results",
            "",
            "| Method | Budget | Accuracy | Avg used | Invalid | Insufficient |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["fixed_budget_results"]:
        lines.append(
            f"| {row['method']} | {row['budget']} | "
            f"{row['accuracy_percent']:.2f}% | {row['avg_samples_used']:.3f} | "
            f"{row['invalid_predictions']} | {row['insufficient_sample_questions']} |"
        )

    lines.extend(
        [
            "",
            "## Method implementation and score sources",
            "",
            "CISC, Self-Certainty, and RASC require optional native confidence, "
            "token-certainty, or reasoning-score fields.  When those fields are "
            "absent, this report uses calibrated P(Yes) only as an explicitly "
            "labelled proxy.",
            "",
            "| Method | Status | Native records | Native fields |",
            "|---|---|---:|---|",
        ]
    )
    for method in (*FIXED_METHODS, *DYNAMIC_METHODS):
        metadata = report.get("method_metadata", {}).get(method, {})
        native_fields = ", ".join(metadata.get("native_score_fields", [])) or "—"
        lines.append(
            f"| {method} | {metadata.get('implementation_status', 'unknown')} | "
            f"{metadata.get('native_score_records', '—')} | {native_fields} |"
        )

    lines.extend(
        [
            "",
            "## Dynamic methods at requested average budgets",
            "",
            "Parameters are selected solely by closeness to the requested average "
            "sample count, never by test accuracy.  ESC's parameter is an integer "
            "non-overlapping window size; other dynamic methods use a confidence "
            "threshold.",
            "",
            "| Method | Target | Threshold/window | Accuracy | Avg used | Gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["dynamic_budget_matches"]:
        parameter = (
            row.get("window_size")
            if row.get("parameter_type") == "window_size"
            else row.get("threshold")
        )
        if parameter is None:
            parameter = "—"
        lines.append(
            f"| {row['method']} | {row['budget_target']} | "
            f"{float(parameter):.4f} | {row['accuracy_percent']:.2f}% | "
            f"{row['avg_samples_used']:.3f} | {row['budget_gap']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Threshold curves",
            "",
            "Default confidence threshold grid: `0.00` through `1.00` in steps of "
            "`0.01`; ESC additionally reports window sizes `2..curve_max_budget`.",
            "",
            "| Method | Threshold/window | Accuracy | Avg used | Early stops | Invalid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in DYNAMIC_METHODS:
        # ``run_aggregation`` canonicalizes legacy ``CaTS-*`` labels at the
        # report boundary, so look up either spelling while older in-memory
        # reports remain readable.
        canonical = canonical_method_name(method)
        rows = report["threshold_curves"].get(method)
        if rows is None:
            rows = report["threshold_curves"].get(canonical, [])
        for row in rows:
            parameter = (
                row.get("window_size")
                if row.get("parameter_type") == "window_size"
                else row.get("threshold")
            )
            if parameter is None:
                parameter = "—"
            lines.append(
                f"| {method} | {float(parameter):.4f} | "
                f"{row['accuracy_percent']:.2f}% | {row['avg_samples_used']:.3f} | "
                f"{row['early_stop_questions']} | {row['invalid_predictions']} |"
            )
    return "\n".join(lines) + "\n"


def write_reports(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    # Keep the writer safe for callers that construct/restore a report
    # directly (rather than going through ``run_aggregation``).  Normalizing
    # twice is harmless and guarantees that every persisted format uses the
    # canonical RelaCaTS labels.
    report = canonicalize_report_methods(report)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "evaluation.json"
    csv_path = output / "evaluation.csv"
    markdown_path = output / "evaluation.md"
    atomic_write_json(json_path, report)

    temporary_csv = output / ".evaluation.csv.tmp"
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in _flat_rows(report):
            writer.writerow(row)
    temporary_csv.replace(csv_path)

    temporary_markdown = output / ".evaluation.md.tmp"
    temporary_markdown.write_text(_markdown(report), encoding="utf-8")
    temporary_markdown.replace(markdown_path)
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def run_aggregation(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    config: AggregateConfig | None = None,
    expected_question_count: int | None = None,
) -> dict[str, Any]:
    files = _discover_confidence_files(inputs)
    if expected_question_count is None:
        expected_question_count = _manifest_expected_questions(files)
    report = evaluate_records(
        _iter_files(files),
        config=config,
        expected_question_count=expected_question_count,
    )
    # Normalize legacy internal labels at the report boundary.  This keeps
    # older confidence artifacts and callers that still pass ``CaTS-*``
    # aliases compatible, while every newly persisted JSON/CSV/Markdown report
    # exposes the unambiguous ``RelaCaTS-*`` names.
    report = canonicalize_report_methods(report)
    report["input_files"] = [str(path) for path in files]
    output_path = Path(output_dir).expanduser().resolve()
    # Add output paths before serialising evaluation.json.  Previously these
    # fields were appended only after write_reports(), so the returned Python
    # object had them while the persisted JSON silently did not.
    report["output_files"] = {
        "json": str(output_path / "evaluation.json"),
        "csv": str(output_path / "evaluation.csv"),
        "markdown": str(output_path / "evaluation.md"),
    }
    write_reports(report, output_path)
    return report


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list")
    return values


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated float list")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only Table-2 evaluation: SC/CISC/Self-Certainty/Best-of-N/"
            "ASC/ESC/RASC baselines plus RelaCaTS-SC/ES/ASC"
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="Confidence artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", type=_parse_ints, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--thresholds",
        type=_parse_floats,
        default=tuple(index / 100 for index in range(101)),
    )
    parser.add_argument("--curve-max-budget", type=int, default=32)
    parser.add_argument("--budget-targets", type=_parse_ints, default=(16,))
    parser.add_argument("--dynamic-min-valid", type=int, default=2)
    parser.add_argument("--rasc-buffer-size", type=int, default=5)
    parser.add_argument(
        "--esc-window-sizes",
        type=_parse_ints,
        default=(),
        help="ESC non-overlapping window sizes (default: every size 2..curve-max-budget)",
    )
    parser.add_argument("--cisc-temperature", type=float, default=1.0)
    parser.add_argument(
        "--cisc-normalization",
        choices=("softmax", "linear", "none"),
        default="softmax",
    )
    parser.add_argument(
        "--expected-questions",
        type=int,
        help="Strict denominator override; entirely missing questions count wrong",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AggregateConfig(
        budgets=tuple(args.budgets),
        thresholds=tuple(args.thresholds),
        curve_max_budget=args.curve_max_budget,
        budget_targets=tuple(args.budget_targets),
        dynamic_min_valid=args.dynamic_min_valid,
        rasc_buffer_size=args.rasc_buffer_size,
        esc_window_sizes=tuple(args.esc_window_sizes),
        cisc_temperature=args.cisc_temperature,
        cisc_normalization=args.cisc_normalization,
    )
    report = run_aggregation(
        args.input,
        args.output_dir,
        config=config,
        expected_question_count=args.expected_questions,
    )
    print(json.dumps(report["output_files"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
