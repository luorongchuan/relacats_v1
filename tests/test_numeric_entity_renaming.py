from relacats_v1.core.numeric_metamorphic_views import (
    generate_numeric_metamorphic_views,
    recognized_person_names,
)


def test_entity_rename_profile_has_identity_plus_three_views():
    question = (
        "Natalia sold 48 clips on Monday. John sold 12 clips. "
        "How many clips did Natalia and John sell altogether?"
    )
    views = generate_numeric_metamorphic_views(question, profile="entity_rename")
    assert [view.relation_subtype for view in views] == [
        "identity",
        "entity_renaming_1",
        "entity_renaming_2",
        "entity_renaming_3",
    ]
    assert all(view.verify() for view in views)
    assert all("48" in view.transformed_question for view in views)
    assert all("12" in view.transformed_question for view in views)


def test_entity_rename_is_exact_and_consistent_for_repeated_name():
    question = "Mary has 7 books. Mary buys 3 more books. How many books does Mary have?"
    views = generate_numeric_metamorphic_views(question, profile="entity_rename")
    assert len(views) == 4
    for view in views[1:]:
        mapping = dict(view.entity_mapping)
        assert "Mary" in mapping
        target = mapping["Mary"]
        assert view.transformed_question.count(target) == 3
        assert "Mary" not in view.transformed_question
        assert view.verify()


def test_entity_rename_falls_back_to_identity_without_safe_name():
    question = "A store has 12 boxes with 5 pencils in each box. How many pencils are there?"
    views = generate_numeric_metamorphic_views(question, profile="entity_rename")
    assert len(views) == 1
    assert views[0].relation_subtype == "identity"
    assert views[0].transformed_question == question
    assert views[0].verify()


def test_month_names_are_not_treated_as_people():
    question = "A shop sold 10 items in April and 12 items in May. How many items total?"
    assert recognized_person_names(question) == ()
    views = generate_numeric_metamorphic_views(question, profile="entity_rename")
    assert len(views) == 1
