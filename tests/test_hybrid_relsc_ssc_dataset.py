from relacats_v1.data_creation.build_hybrid_relsc_ssc_dataset import (
    hybrid_scores,
    numeric_ssc_scores,
    pure_relsc_scores,
)


def _sample(answer, confidence, valid=True):
    return {
        "canonicalized_answer": answer if valid else None,
        "is_valid_answer": valid,
        "confidence": confidence,
        "relation_weight": 1.0,
        "dependency_weight": 1.0,
    }


def test_mcq_relsc_is_pure_count_not_confidence_weighted():
    samples = [
        _sample("A", 0.05),
        _sample("A", 0.05),
        _sample("B", 0.99),
    ]
    scores = pure_relsc_scores(samples)
    assert scores["A"] == 2 / 3
    assert scores["B"] == 1 / 3

    method, hybrid = hybrid_scores(samples, answer_type="option letter")
    assert method == "relsc"
    assert hybrid == scores


def test_numeric_ssc_uses_confidence_weighting():
    samples = [
        _sample("10", 0.05),
        _sample("10", 0.05),
        _sample("20", 0.90),
    ]
    scores = numeric_ssc_scores(samples)
    assert abs(scores["10"] - 0.10) < 1e-12
    assert abs(scores["20"] - 0.90) < 1e-12

    method, hybrid = hybrid_scores(samples, answer_type="number")
    assert method == "ssc"
    assert hybrid == scores


def test_invalid_answers_are_excluded_from_both_denominators():
    samples = [
        _sample("A", 0.2),
        _sample("B", 0.8),
        _sample("X", 1.0, valid=False),
    ]
    relsc = pure_relsc_scores(samples)
    assert relsc == {"A": 0.5, "B": 0.5}

    numeric = [
        _sample("1", 0.2),
        _sample("2", 0.8),
        _sample("3", 1.0, valid=False),
    ]
    ssc = numeric_ssc_scores(numeric)
    assert abs(ssc["1"] - 0.2) < 1e-12
    assert abs(ssc["2"] - 0.8) < 1e-12


def test_empty_valid_support_is_undefined():
    samples = [_sample("A", 0.9, valid=False)]
    assert pure_relsc_scores(samples) == {}
    assert numeric_ssc_scores(samples) == {}
