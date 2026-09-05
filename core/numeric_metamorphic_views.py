"""Certified metamorphic relation views for scalar math word problems.

This module targets GSM8K and SVAMP.  Every emitted transformation is an
*input-side* deterministic relation whose correct scalar answer is invariant:
``phi_g(T) = T``.  We deliberately avoid changing the final answer by an
artificial affine code.

The four preferred views are:

    g0: identity
    g1: layout wrapper
    g2: number representation
        e.g. ``48`` -> ``forty-eight`` and ``$ 6`` -> ``six dollars``
    g3: equivalent quantity expression
        e.g. ``48`` -> ``(47 + 1)`` and ``$100`` -> ``(99 + 1) dollars``

Only locally certifiable edits are emitted.  Decimals, percentages, ratios,
times, dates and other ambiguous numeric forms are left unchanged.  If a
relation cannot be constructed safely it is omitted rather than forced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable


class NumericRelationError(ValueError):
    """Raised when a numeric relation cannot be certified."""


# Standalone non-negative integer.  We additionally reject an integer whose
# immediate left context, after whitespace, is '$'; currency is handled by a
# dedicated parser below so that we never create unnatural strings such as
# ``$ six``.
_INTEGER_RE = re.compile(r"(?<![\w.$%/:-])\d+(?![\w.%/:-])")

# Conservative integer currency form.  Decimal money such as $4.50 is not
# matched.  The whole '$ ... number' span is replaced, which makes the edit
# exactly reversible and yields natural English such as 'six dollars'.
_CURRENCY_RE = re.compile(r"\$\s*(\d{1,4})(?![\d,.%/:-])")

_SMALL = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def integer_to_english(value: int) -> str:
    """Convert an integer in [0, 999] to deterministic English words."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise NumericRelationError(f"value must be an integer; got {value!r}")
    if not 0 <= value <= 999:
        raise NumericRelationError(
            f"number-word transform supports only integers in [0, 999]; got {value}"
        )
    if value < 20:
        return _SMALL[value]
    if value < 100:
        tens = (value // 10) * 10
        ones = value % 10
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_SMALL[ones]}"
    hundreds = value // 100
    rest = value % 100
    prefix = f"{_SMALL[hundreds]} hundred"
    return prefix if rest == 0 else f"{prefix} {integer_to_english(rest)}"


@dataclass(frozen=True)
class CertifiedEdit:
    """One exact reversible substring edit used to certify a relation view."""

    start: int
    end: int
    original: str
    replacement: str

    def apply(self, text: str) -> str:
        if text[self.start : self.end] != self.original:
            raise NumericRelationError("edit span does not match original text")
        return text[: self.start] + self.replacement + text[self.end :]

    def reverse(self, transformed: str) -> str:
        replacement_end = self.start + len(self.replacement)
        if transformed[self.start : replacement_end] != self.replacement:
            raise NumericRelationError(
                "transformed text does not contain certified replacement"
            )
        return (
            transformed[: self.start]
            + self.original
            + transformed[replacement_end:]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericMetamorphicView:
    """A certified invariant relation view for a scalar math word problem."""

    relation_id: str
    relation_type: str
    relation_subtype: str
    original_question: str
    transformed_question: str
    certification: str
    edit: CertifiedEdit | None = None
    answer_type: str = "number"
    relation_mode: str = "numeric_metamorphic"
    answer_mapping: str = "identity"

    def __post_init__(self) -> None:
        if not self.original_question.strip():
            raise NumericRelationError("original question must not be empty")
        if not self.transformed_question.strip():
            raise NumericRelationError("transformed question must not be empty")
        if self.answer_mapping != "identity":
            raise NumericRelationError(
                "numeric metamorphic views currently require phi_g(T)=T"
            )

    def verify(self) -> bool:
        """Mechanically verify that the stored transform is reversible/auditable."""

        if self.relation_subtype == "identity":
            return self.transformed_question == self.original_question
        if self.relation_subtype == "layout_wrapper":
            marker_start = "[Problem]\n"
            marker_end = "\n[/Problem]\nSolve the mathematical problem above."
            return (
                self.transformed_question.startswith(marker_start)
                and self.transformed_question.endswith(marker_end)
                and self.transformed_question[
                    len(marker_start) : -len(marker_end)
                ]
                == self.original_question
            )
        if self.edit is None:
            return False
        try:
            return self.edit.reverse(self.transformed_question) == self.original_question
        except NumericRelationError:
            return False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "relation_subtype": self.relation_subtype,
            "relation_mode": self.relation_mode,
            "answer_type": self.answer_type,
            "answer_mapping": self.answer_mapping,
            "certification": self.certification,
            "certified_edit": self.edit.to_dict() if self.edit is not None else None,
            "relation_verified": self.verify(),
        }


def _is_currency_left_context(question: str, start: int) -> bool:
    """Whether the number begins immediately after a dollar sign + whitespace."""

    return re.search(r"\$\s*$", question[:start]) is not None


def _candidate_integer_spans(question: str) -> list[tuple[int, int, int, str]]:
    """Return safe non-currency standalone integer spans left-to-right."""

    result: list[tuple[int, int, int, str]] = []
    for match in _INTEGER_RE.finditer(question):
        if _is_currency_left_context(question, match.start()):
            continue
        token = match.group(0)
        try:
            value = int(token)
        except ValueError:
            continue
        result.append((match.start(), match.end(), value, token))
    return result


def _candidate_currency_spans(question: str) -> list[tuple[int, int, int, str]]:
    """Return integer-dollar spans such as '$12' and '$ 6'."""

    result: list[tuple[int, int, int, str]] = []
    for match in _CURRENCY_RE.finditer(question):
        token = match.group(0)
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        result.append((match.start(), match.end(), value, token))
    return result


def _currency_words(value: int) -> str:
    unit = "dollar" if value == 1 else "dollars"
    return f"{integer_to_english(value)} {unit}"


def _choose_number_word_edit(question: str) -> CertifiedEdit | None:
    # Prefer ordinary quantities because this changes only their surface form.
    for start, end, value, token in _candidate_integer_spans(question):
        if 0 <= value <= 999:
            replacement = integer_to_english(value)
            if replacement != token:
                return CertifiedEdit(start, end, token, replacement)

    # Currency-only questions are common in GSM8K.  Replacing the whole money
    # span avoids '$ six' and preserves the monetary unit explicitly.
    for start, end, value, token in _candidate_currency_spans(question):
        if 0 <= value <= 999:
            return CertifiedEdit(start, end, token, _currency_words(value))
    return None


def _choose_equivalent_expression_edit(
    question: str,
    *,
    avoid_span: tuple[int, int] | None = None,
) -> CertifiedEdit | None:
    """Choose a simple arithmetic re-expression while preserving semantics."""

    for start, end, value, token in _candidate_integer_spans(question):
        if avoid_span is not None and (start, end) == avoid_span:
            continue
        if 2 <= value <= 9999:
            left = value - 1
            if left + 1 != value:
                raise AssertionError("equivalent-expression certificate failed")
            return CertifiedEdit(start, end, token, f"({left} + 1)")

    for start, end, value, token in _candidate_currency_spans(question):
        if avoid_span is not None and (start, end) == avoid_span:
            continue
        if 2 <= value <= 9999:
            left = value - 1
            unit = "dollar" if value == 1 else "dollars"
            return CertifiedEdit(start, end, token, f"({left} + 1) {unit}")
    return None


def generate_numeric_metamorphic_views(
    question: str,
    *,
    include_layout: bool = True,
    include_number_words: bool = True,
    include_equivalent_expression: bool = True,
) -> tuple[NumericMetamorphicView, ...]:
    """Generate conservative certified relation views for GSM8K/SVAMP.

    No language model is used to create the transformation itself.  Every
    accepted edit is deterministic and reversible, and the expected scalar
    answer is unchanged.
    """

    original = str(question).strip()
    if not original:
        raise NumericRelationError("question must not be empty")

    views: list[NumericMetamorphicView] = [
        NumericMetamorphicView(
            relation_id="g0",
            relation_type="identity",
            relation_subtype="identity",
            original_question=original,
            transformed_question=original,
            certification="exact_identity",
        )
    ]

    if include_layout:
        transformed = (
            f"[Problem]\n{original}\n[/Problem]\n"
            "Solve the mathematical problem above."
        )
        views.append(
            NumericMetamorphicView(
                relation_id=f"g{len(views)}",
                relation_type="invariant",
                relation_subtype="layout_wrapper",
                original_question=original,
                transformed_question=transformed,
                certification="original_problem_embedded_verbatim",
            )
        )

    word_edit: CertifiedEdit | None = None
    if include_number_words:
        word_edit = _choose_number_word_edit(original)
        if word_edit is not None:
            view = NumericMetamorphicView(
                relation_id=f"g{len(views)}",
                relation_type="invariant",
                relation_subtype="number_representation",
                original_question=original,
                transformed_question=word_edit.apply(original),
                certification="single_reversible_numeric_representation_edit",
                edit=word_edit,
            )
            if not view.verify():
                raise NumericRelationError(
                    "number-representation view failed certification"
                )
            views.append(view)

    if include_equivalent_expression:
        avoid = (word_edit.start, word_edit.end) if word_edit is not None else None
        expression_edit = _choose_equivalent_expression_edit(
            original, avoid_span=avoid
        )
        # Reusing the same source quantity is allowed if it is the only safe
        # quantity; the two views still challenge different representations.
        if expression_edit is None:
            expression_edit = _choose_equivalent_expression_edit(original)
        if expression_edit is not None:
            view = NumericMetamorphicView(
                relation_id=f"g{len(views)}",
                relation_type="invariant",
                relation_subtype="equivalent_quantity",
                original_question=original,
                transformed_question=expression_edit.apply(original),
                certification="single_reversible_integer_identity_expression",
                edit=expression_edit,
            )
            if not view.verify():
                raise NumericRelationError(
                    "equivalent-quantity view failed certification"
                )
            views.append(view)

    if any(not view.verify() for view in views):
        raise NumericRelationError("at least one generated view failed certification")
    return tuple(views)


def relation_coverage(questions: Iterable[str]) -> dict[str, int]:
    """Return a CPU-only coverage audit over a collection of questions."""

    counts = {
        "questions": 0,
        "identity": 0,
        "layout_wrapper": 0,
        "number_representation": 0,
        "equivalent_quantity": 0,
        "four_view_questions": 0,
    }
    for question in questions:
        views = generate_numeric_metamorphic_views(question)
        counts["questions"] += 1
        subtypes = {view.relation_subtype for view in views}
        for key in (
            "identity",
            "layout_wrapper",
            "number_representation",
            "equivalent_quantity",
        ):
            counts[key] += int(key in subtypes)
        counts["four_view_questions"] += int(len(views) == 4)
    return counts
