from argparse import Namespace

from relacats_v1.data_creation.dataset_adapter import MCQExample
from relacats_v1.data_creation.generate_mcq_identity_only_data import _identity_view


def _args():
    return Namespace()


def test_mcq_identity_view_is_exactly_one_32_sample_identity_view():
    example = MCQExample(
        dataset_name="arc_easy",
        split="train",
        source_index=0,
        question_id="arc_easy:train:0:test",
        stem="Question: Which option is correct?",
        options=("alpha", "beta", "gamma", "delta"),
        correct_index=1,
    )
    view = _identity_view(example, _args())

    assert view.relation_id == "g0"
    assert view.relation_type == "identity"
    assert view.relation_mode == "identity_only"
    assert view.answer_type == "option letter"
    assert view.samples_per_view == 32
    assert view.transformed_question == view.original_question
    assert view.transformed_options == example.options
    assert view.permutation == {"A": "A", "B": "B", "C": "C", "D": "D"}
    assert view.inverse_permutation == view.permutation


def test_winogrande_identity_baseline_does_not_force_swap_view():
    example = MCQExample(
        dataset_name="winogrande",
        split="train",
        source_index=0,
        question_id="winogrande:train:0:test",
        stem="Question: Alex thanked Jordan because _ helped.",
        options=("Alex", "Jordan"),
        correct_index=1,
    )
    view = _identity_view(example, _args())

    assert view.samples_per_view == 32
    assert view.transformed_options == ("Alex", "Jordan")
    assert view.permutation == {"A": "A", "B": "B"}
