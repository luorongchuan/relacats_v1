"""Certified metamorphic relation views for scalar math word problems.

This module is intentionally conservative.  It targets the two scalar-answer
training datasets used by RelaCaTS, GSM8K and SVAMP, and constructs *input-side*
relations whose correct numerical answer is invariant.

The goal is not to paraphrase the whole problem with an LLM.  Instead, every
accepted transform is deterministic, reversible, and locally certifiable.
That keeps relation noise small and makes the generated view auditable.

Formal profile (when all views are available):

    g0: identity
    g1: layout wrapper
    g2: number representation (e.g. 48 -> forty-eight)
    g3: equivalent quantity expression (e.g. 48 -> (47 + 1))

All four views use phi_g(T) = T, so numeric canonicalization remains the
ordinary identity canonicalization.  The transformed final answer is *not*
artificially changed.

If a safe numerical literal cannot be found, g2/g3 are omitted rather than
forcing a potentially unsafe transformation.  Downstream code may either use
only certified views or choose a fallback sampling policy.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any, Iterable


class NumericRelationError(ValueError):
    """Raised when a numeric relation cannot be certified."""


# We deliberately transform only simple standalone non-negative integers.
# Exclusions are conservative: decimals, percentages, currency, ratios,
# hyphenated forms, and numbers embedded in identifiers are left untouched.
_INTEGER_RE = re.compile(r"(?<![\w.$%/:-])\d+(?![\w.%/:-])")

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
    if rest == 0:
        return prefix
    return f"{prefix} {integer_to_english(rest)}"


@dataclass(frozen=True)
class CertifiedEdit:
    """One exact, reversible substring edit used to certify a relation view."""

    start: int
    end: int
    original: str
    replacement: str

    def apply(self, text: str) -> str:
        if text[self.start : self.end] != self.original:
            raise NumericRelationError("edit span does not match original text")
        return text[: self.start] + self.replacement + text[self.end :]

    def reverse(self, transformed: str) -> str:
        prefix_len = self.start
        replacement_end = prefix_len + len(self.replacement)
        if transformed[prefix_len:replacement_end] != self.replacement:
            raise NumericRelationError("transformed text does not contain certified replacement")
        return (
            transformed[:prefix_len]
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
            raise NumericRelationError("numeric metamorphic views currently require phi_g(T)=T")

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


def _candidate_integer_spans(question: str) -> list[tuple[int, int, int, str]]:
    """Return safe standalone integer spans in deterministic left-to-right order."""

    result: list[tuple[int, int, int, str]] = []
    for match in _INTEGER_RE.finditer(question):
        token = match.group(0)
        # Avoid very long integers (years/IDs are common nuisance cases); the
        # representation transform itself is limited to <= 999.
        try:
            value = int(token)
        except ValueError:
            continue
        result.append((match.start(), match.end(), value, token))
    return result


def _choose_number_word_edit(question: str) -> CertifiedEdit | None:
    for start, end, value, token in _candidate_integer_spans(question):
        if 0 <= value <= 999:
            replacement = integer_to_english(value)
            if replacement != token:
                return CertifiedEdit(start, end, token, replacement)
    return None


def _choose_equivalent_expression_edit(
    question: str,
    *,
    avoid_span: tuple[int, int] | None = None,
) -> CertifiedEdit | None:
    """Choose a simple arithmetic re-expression n -> (n-1 + 1).

    We use only n >= 2 and n <= 9999.  The expression is trivially certifiable
    by integer arithmetic and intentionally adds only one elementary operation.
    """

    for start, end, value, token in _candidate_integer_spans(question):
        if avoid_span is not None and (start, end) == avoid_span:
            continue
        if 2 <= value <= 9999:
            left = value - 1
            replacement = f"({left} + 1)"
            # Independent arithmetic certificate rather than trusting string construction.
            if left + 1 != value:
                raise AssertionError("equivalent-expression arithmetic certificate failed")
            return CertifiedEdit(start, end, token, replacement)
    return None


def generate_numeric_metamorphic_views(
    question: str,
    *,
    include_layout: bool = True,
    include_number_words: bool = True,
    include_equivalent_expression: bool = True,
) -> tuple[NumericMetamorphicView, ...]:
    """Generate conservative certified relation views for GSM8K/SVAMP.

    The function never calls a language model and never changes the expected
    numerical answer.  Unsafe/unavailable relation types are skipped.
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
        transformed = f"[Problem]\n{original}\n[/Problem]\nSolve the mathematical problem above."
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
            transformed = word_edit.apply(original)
            view = NumericMetamorphicView(
                relation_id=f"g{len(views)}",
                relation_type="invariant",
                relation_subtype="number_representation",
                original_question=original,
                transformed_question=transformed,
                certification="single_reversible_integer_to_english_edit",
                edit=word_edit,
            )
            if not view.verify():
                raise NumericRelationError("number-representation view failed certification")
            views.append(view)

    if include_equivalent_expression:
        avoid = (word_edit.start, word_edit.end) if word_edit is not None else None
        expression_edit = _choose_equivalent_expression_edit(original, avoid_span=avoid)
        # If there is only one safe number, using the same span is still useful;
        # the two relations challenge different surface representations.
        if expression_edit is None:
            expression_edit = _choose_equivalent_expression_edit(original)
        if expression_edit is not None:
            transformed = expression_edit.apply(original)
            view = NumericMetamorphicView(
                relation_id=f"g{len(views)}",
                relation_type="invariant",
                relation_subtype="equivalent_quantity",
                original_question=original,
                transformed_question=transformed,
                certification="single_reversible_integer_identity_expression",
                edit=expression_edit,
            )
            if not view.verify():
                raise NumericRelationError("equivalent-quantity view failed certification")
            views.append(view)

    # Relation IDs are assigned after eligibility decisions, so they are always
    # contiguous g0, g1, ... and stable for a fixed question.
    if any(not view.verify() for view in views):
        raise NumericRelationError("at least one generated view failed certification")
    return tuple(views)


def relation_coverage(questions: Iterable[str]) -> dict[str, int]:
    """Return a small CPU-only coverage audit over a collection of questions."""

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
        for key in ("identity", "layout_wrapper", "number_representation", "equivalent_quantity"):
            counts[key] += int(key in subtypes)
        counts["four_view_questions"] += int(len(views) == 4)
    return counts
