"""CPU contract tests for the complete evaluation method set."""

from __future__ import annotations

from relacats_v1.evaluation.aggregate import (
    AggregateConfig,
    _dynamic_predict,
    _self_certainty_from_tokens,
    evaluate_records,
)
from relacats_v1.evaluation.synthetic_smoke import (
    EXPECTED_DYNAMIC_METHODS,
    EXPECTED_FIXED_METHODS,
    run_smoke,
)


def test_synthetic_aggregate_emits_all_canonical_methods():
    """Run the tiny no-GPU fixture and verify the public report contract."""

    report = run_smoke()
    fixed = {row["method"] for row in report["fixed_budget_results"]}
    dynamic = set(report["threshold_curves"])
    assert fixed == EXPECTED_FIXED_METHODS
    assert dynamic == EXPECTED_DYNAMIC_METHODS
    assert report["evaluation_namespace"] == "RelaCaTS"
    assert set(report["method_order"]) == fixed | dynamic


def test_optional_cisc_score_is_not_reduced_to_calibrated_proxy():
    """Exported CISC scores must survive grouping and affect its vote.

    Real v1 confidence artifacts normally lack this optional field and are
    therefore explicitly evaluated with the documented calibrated-confidence
    proxy.  When an artifact does provide an untrained CISC score, however, the
    evaluator must not discard it while slimming records into question groups.
    """

    records = []
    # Calibrated confidence favors B, while the independent CISC score favors
    # A.  The gold answer is A, making accidental proxy use observable.
    calibrated = (("B", 0.95, 0.01), ("B", 0.90, 0.02), ("A", 0.10, 0.99), ("A", 0.05, 0.98))
    for index, (answer, confidence, cisc_confidence) in enumerate(calibrated):
        records.append(
            {
                "sample_id": f"cisc-q-r{index}",
                "question_id": "cisc-q",
                "generation_index": index,
                "dataset_name": "synthetic",
                "correct_answer": "A",
                "extracted_answer": answer,
                "confidence": confidence,
                "cisc_confidence": cisc_confidence,
            }
        )

    report = evaluate_records(
        records,
        config=AggregateConfig(
            budgets=(4, 16),
            thresholds=(0.5,),
            curve_max_budget=4,
            budget_targets=(4,),
        ),
    )
    cisc = next(
        row
        for row in report["fixed_budget_results"]
        if row["method"] == "CISC" and row["budget"] == 4
    )
    relacats_sc = next(
        row
        for row in report["fixed_budget_results"]
        if row["method"] == "RelaCaTS-SC" and row["budget"] == 4
    )
    assert cisc["correct"] == 1
    assert relacats_sc["correct"] == 0


def test_esc_checks_non_overlapping_windows():
    """ESC must charge complete sequential windows, not a sliding window."""

    records = [
        {"extracted_answer": answer, "confidence": 0.5}
        for answer in ("A", "B", "B", "B")
    ]
    prediction, used, status = _dynamic_predict(
        "ESC", records, threshold=2, max_budget=4, min_valid=2
    )
    # The first (A,B) window is not unanimous; the second (B,B) window is,
    # so a sliding implementation would incorrectly stop after sample 3.
    assert prediction == "B"
    assert used == 4
    assert status == "early_stop"


def test_rasc_optional_reasoning_score_is_used():
    """A native RASC score must override the calibrated-confidence proxy."""

    records = []
    # Calibrated confidence favors B, while the reasoning/sufficiency score
    # favors A.  A capacity of two makes the selected answer observable.
    for index, (answer, confidence, rasc_score) in enumerate(
        (("B", 0.95, 0.90), ("A", 0.10, 0.95), ("A", 0.05, 0.96), ("B", 0.80, 0.10))
    ):
        records.append(
            {
                "sample_id": f"rasc-q-r{index}",
                "question_id": "rasc-q",
                "generation_index": index,
                "correct_answer": "A",
                "extracted_answer": answer,
                "confidence": confidence,
                "rasc_score": rasc_score,
            }
        )
    prediction, used, status = _dynamic_predict(
        "RASC", records, threshold=0.5, max_budget=4, min_valid=2, rasc_buffer_size=2
    )
    assert prediction == "A"
    assert used == 2
    assert status == "early_stop"


def test_self_certainty_rejects_truncated_top_k_vectors():
    """A top-k export must not be mistaken for a full-vocabulary score."""

    assert (
        _self_certainty_from_tokens(
            {
                "self_certainty_vocab_size": 4,
                "self_certainty_token_probabilities": [[0.25, 0.25]],
            }
        )
        is None
    )
