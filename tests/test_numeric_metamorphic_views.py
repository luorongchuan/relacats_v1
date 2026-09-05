from relacats_v1.core.numeric_metamorphic_views import (
    generate_numeric_metamorphic_views,
    integer_to_english,
)


def test_integer_to_english_examples():
    assert integer_to_english(0) == "zero"
    assert integer_to_english(8) == "eight"
    assert integer_to_english(19) == "nineteen"
    assert integer_to_english(48) == "forty-eight"
    assert integer_to_english(100) == "one hundred"
    assert integer_to_english(248) == "two hundred forty-eight"


def test_four_views_on_typical_gsm8k_question():
    question = (
        "Natalia sold 48 clips in April and half as many in May. "
        "How many clips did Natalia sell altogether?"
    )
    views = generate_numeric_metamorphic_views(question)
    assert len(views) == 4
    assert [v.relation_subtype for v in views] == [
        "identity",
        "layout_wrapper",
        "number_representation",
        "equivalent_quantity",
    ]
    assert all(v.verify() for v in views)
    assert all(v.answer_mapping == "identity" for v in views)
    assert "forty-eight" in views[2].transformed_question
    assert "(47 + 1)" in views[3].transformed_question


def test_numeric_edits_are_exactly_reversible():
    question = "A box has 36 red balls and 12 blue balls. How many balls are there?"
    views = generate_numeric_metamorphic_views(question)
    for view in views:
        assert view.verify()
        if view.edit is not None:
            assert view.edit.reverse(view.transformed_question) == question


def test_unsafe_decimal_currency_and_percentage_are_not_modified():
    question = "A price changes from $4.50 to $5.00, a 10% increase. What is the difference?"
    views = generate_numeric_metamorphic_views(question)
    assert [v.relation_subtype for v in views] == ["identity", "layout_wrapper"]
    assert all(v.verify() for v in views)


def test_integer_currency_is_rendered_naturally():
    question = "Edward spent $ 6 on a book. How much did he spend?"
    views = generate_numeric_metamorphic_views(question)
    assert len(views) == 4
    word_view = next(v for v in views if v.relation_subtype == "number_representation")
    expr_view = next(v for v in views if v.relation_subtype == "equivalent_quantity")
    assert "six dollars" in word_view.transformed_question
    assert "$ six" not in word_view.transformed_question
    assert "(5 + 1) dollars" in expr_view.transformed_question
    assert all(v.verify() for v in views)


def test_two_different_number_spans_preferred_when_available():
    question = "There are 48 apples in 6 baskets. How many apples per basket?"
    views = generate_numeric_metamorphic_views(question)
    word_view = next(v for v in views if v.relation_subtype == "number_representation")
    expr_view = next(v for v in views if v.relation_subtype == "equivalent_quantity")
    assert word_view.edit is not None
    assert expr_view.edit is not None
    assert (word_view.edit.start, word_view.edit.end) != (
        expr_view.edit.start,
        expr_view.edit.end,
    )
    assert all(v.verify() for v in views)
