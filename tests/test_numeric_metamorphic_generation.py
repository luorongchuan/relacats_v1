from relacats_v1.data_creation.generate_numeric_metamorphic_data import (
    TOTAL_BUDGET,
    allocate_view_samples,
)


def test_standard_four_view_budget():
    assert allocate_view_samples(4) == (8, 8, 8, 8)
    assert sum(allocate_view_samples(4)) == TOTAL_BUDGET


def test_two_view_fallback_budget():
    assert allocate_view_samples(2) == (16, 16)
    assert sum(allocate_view_samples(2)) == TOTAL_BUDGET


def test_three_view_fallback_keeps_total_budget():
    assert allocate_view_samples(3) == (11, 11, 10)
    assert sum(allocate_view_samples(3)) == TOTAL_BUDGET


def test_budget_allocation_is_positive():
    for number_of_views in range(1, 9):
        counts = allocate_view_samples(number_of_views)
        assert sum(counts) == TOTAL_BUDGET
        assert min(counts) > 0
