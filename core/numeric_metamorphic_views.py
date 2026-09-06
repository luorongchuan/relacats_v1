"""Certified metamorphic relation views for scalar math word problems.

This module targets GSM8K and SVAMP. Every emitted transformation is an
input-side deterministic relation whose correct scalar answer is invariant:
``phi_g(T) = T``.

Two profiles are supported:

``legacy``
    The original diagnostic profile: identity, layout wrapper, number
    representation, and equivalent quantity expression.

``entity_rename``
    A reasoning-structure-preserving profile: identity plus up to three
    certified person-name renamings. Numeric tokens, quantities, units, and
    arithmetic structure are left untouched. If no conservatively recognised
    person name is present, the profile falls back to identity only rather
    than inventing an unsafe transformation.

The entity-renaming profile is deliberately conservative. It uses a curated
name lexicon, preserves coarse name gender so nearby pronouns remain coherent,
renames every occurrence of a selected name consistently, and verifies exact
reversibility before emitting a view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Mapping


class NumericRelationError(ValueError):
    """Raised when a numeric relation cannot be certified."""


NUMERIC_RELATION_PROFILES = ("legacy", "entity_rename")

# Standalone non-negative integer. Currency is handled separately.
_INTEGER_RE = re.compile(r"(?<![\w.$%/:-])\d+(?![\w.%/:-])")
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

# Conservative source lexicon. Ambiguous English words/month names such as
# May, April, June, Will, Bill, Mark, Rose, Hope and Grace are intentionally
# omitted even though some can also be personal names.
_MALE_NAMES = (
    "Aaron", "Adam", "Adrian", "Albert", "Alex", "Alexander", "Andrew",
    "Anthony", "Arthur", "Benjamin", "Billy", "Bob", "Brandon", "Brian",
    "Caleb", "Calvin", "Carl", "Charles", "Charlie", "Chris",
    "Christopher", "Daniel", "David", "Dennis", "Donald", "Douglas",
    "Edward", "Eric", "Ethan", "Frank", "Fred", "Gabriel", "Gary",
    "George", "Greg", "Harold", "Henry", "Howard", "Isaac", "Jack",
    "Jacob", "James", "Jason", "Jeff", "Jeremy", "Jerry", "Jesse", "Jim",
    "Joe", "John", "Jonathan", "Joseph", "Josh", "Joshua", "Justin",
    "Keith", "Ken", "Kenneth", "Kevin", "Larry", "Leo", "Leonard",
    "Lucas", "Martin", "Matthew", "Michael", "Mike", "Nathan", "Nicholas",
    "Noah", "Oliver", "Oscar", "Owen", "Patrick", "Paul", "Peter",
    "Philip", "Ralph", "Randy", "Raymond", "Richard", "Robert", "Roger",
    "Ronald", "Ryan", "Samuel", "Scott", "Sean", "Stephen", "Steven",
    "Thomas", "Tim", "Timothy", "Toby", "Tom", "Trevor", "Victor",
    "Walter", "Wayne", "William", "Zachary",
)
_FEMALE_NAMES = (
    "Abigail", "Alice", "Amanda", "Amber", "Amy", "Angela", "Anna", "Anne",
    "Ashley", "Barbara", "Betty", "Brenda", "Brittany", "Carla", "Carol",
    "Carolyn", "Catherine", "Charlotte", "Cheryl", "Christina", "Christine",
    "Cynthia", "Deborah", "Debra", "Denise", "Diana", "Donna", "Dorothy",
    "Elizabeth", "Emily", "Emma", "Evelyn", "Frances", "Heather", "Helen",
    "Isabella", "Janet", "Jennifer", "Jessica", "Joan", "Joyce", "Judith",
    "Julie", "Karen", "Kathleen", "Katherine", "Kelly", "Kimberly", "Laura",
    "Lauren", "Linda", "Lisa", "Lori", "Madison", "Margaret", "Maria",
    "Marie", "Marilyn", "Martha", "Mary", "Megan", "Melanie", "Melissa",
    "Michelle", "Nancy", "Natalia", "Nicole", "Olivia", "Pamela", "Patricia",
    "Rachel", "Rebecca", "Ruth", "Samantha", "Sandra", "Sara", "Sarah",
    "Sharon", "Sophia", "Stephanie", "Susan", "Teresa", "Tina", "Victoria",
    "Virginia", "Wendy",
)
_NAME_GENDER = {name: "male" for name in _MALE_NAMES} | {
    name: "female" for name in _FEMALE_NAMES
}
_NAME_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(name) for name in sorted(_NAME_GENDER, key=len, reverse=True)
    ) + r")\b"
)

# Three disjoint target banks. Targets are selected only if absent from the
# original question and from all source names, so reverse substitution is
# exact and collision-free.
_ENTITY_TARGET_BANKS: tuple[dict[str, tuple[str, ...]], ...] = (
    {
        "male": (
            "Adrian", "Caleb", "Ethan", "Felix", "Nolan", "Victor", "Owen",
            "Julian", "Simon", "Vincent", "Marcus", "Derek",
        ),
        "female": (
            "Clara", "Daphne", "Elena", "Fiona", "Nora", "Sophie", "Audrey",
            "Bianca", "Cecilia", "Julia", "Madeline", "Zoe",
        ),
    },
    {
        "male": (
            "Benjamin", "Lucas", "Samuel", "Theodore", "Dominic", "Gabriel",
            "Isaac", "Sebastian", "Xavier", "Colin", "Miles", "Nathaniel",
        ),
        "female": (
            "Amelia", "Chloe", "Evelyn", "Hannah", "Isabella", "Lily",
            "Naomi", "Penelope", "Ruby", "Sylvia", "Valerie", "Irene",
        ),
    },
    {
        "male": (
            "Arthur", "Daniel", "Frederick", "Henry", "Jonathan", "Leonard",
            "Matthew", "Oliver", "Patrick", "Quentin", "Russell", "Thomas",
        ),
        "female": (
            "Alice", "Caroline", "Diana", "Emily", "Georgia", "Katherine",
            "Laura", "Monica", "Natalie", "Rachel", "Teresa", "Vanessa",
        ),
    },
)


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
    """One exact reversible substring edit used to certify a legacy view."""

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


def _replace_name_mapping(text: str, mapping: Mapping[str, str]) -> str:
    if not mapping:
        return text
    pattern = re.compile(
        r"\b(?:" + "|".join(
            re.escape(name) for name in sorted(mapping, key=len, reverse=True)
        ) + r")\b"
    )
    return pattern.sub(lambda match: mapping[match.group(0)], text)


def recognized_person_names(question: str) -> tuple[str, ...]:
    """Return conservative recognised person names in first-occurrence order."""

    seen: set[str] = set()
    names: list[str] = []
    for match in _NAME_PATTERN.finditer(str(question)):
        name = match.group(0)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def _build_entity_mapping(
    question: str,
    source_names: tuple[str, ...],
    variant_index: int,
) -> tuple[tuple[str, str], ...]:
    if not 0 <= variant_index < len(_ENTITY_TARGET_BANKS):
        raise NumericRelationError(f"invalid entity variant index: {variant_index}")
    bank = _ENTITY_TARGET_BANKS[variant_index]
    source_set = set(source_names)
    used_targets: set[str] = set()
    mapping: list[tuple[str, str]] = []

    for source in source_names:
        gender = _NAME_GENDER[source]
        target = None
        for candidate in bank[gender]:
            if candidate in source_set or candidate in used_targets:
                continue
            if re.search(rf"\b{re.escape(candidate)}\b", question):
                continue
            target = candidate
            break
        if target is None:
            return ()
        mapping.append((source, target))
        used_targets.add(target)
    return tuple(mapping)


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
    entity_mapping: tuple[tuple[str, str], ...] = ()
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
        if self.relation_subtype.startswith("entity_renaming_"):
            if not self.entity_mapping:
                return False
            forward = dict(self.entity_mapping)
            reverse = {target: source for source, target in self.entity_mapping}
            if len(reverse) != len(self.entity_mapping):
                return False
            transformed = _replace_name_mapping(self.original_question, forward)
            if transformed != self.transformed_question:
                return False
            return _replace_name_mapping(self.transformed_question, reverse) == self.original_question
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
            "entity_mapping": [
                {"source": source, "target": target}
                for source, target in self.entity_mapping
            ],
            "relation_verified": self.verify(),
        }


def _identity_view(original: str) -> NumericMetamorphicView:
    return NumericMetamorphicView(
        relation_id="g0",
        relation_type="identity",
        relation_subtype="identity",
        original_question=original,
        transformed_question=original,
        certification="exact_identity",
    )


def generate_entity_renaming_views(
    question: str,
    *,
    num_variants: int = 3,
) -> tuple[NumericMetamorphicView, ...]:
    """Generate identity + certified name-renaming views without changing numbers."""

    original = str(question).strip()
    if not original:
        raise NumericRelationError("question must not be empty")
    if not 0 <= num_variants <= len(_ENTITY_TARGET_BANKS):
        raise NumericRelationError(
            f"num_variants must be in [0, {len(_ENTITY_TARGET_BANKS)}]"
        )

    views: list[NumericMetamorphicView] = [_identity_view(original)]
    source_names = recognized_person_names(original)
    if not source_names:
        return tuple(views)

    for variant_index in range(num_variants):
        mapping = _build_entity_mapping(original, source_names, variant_index)
        if not mapping:
            continue
        transformed = _replace_name_mapping(original, dict(mapping))
        if transformed == original:
            continue
        view = NumericMetamorphicView(
            relation_id=f"g{len(views)}",
            relation_type="invariant",
            relation_subtype=f"entity_renaming_{variant_index + 1}",
            original_question=original,
            transformed_question=transformed,
            certification="gender_preserving_exact_reversible_person_name_renaming",
            entity_mapping=mapping,
        )
        if not view.verify():
            raise NumericRelationError("entity-renaming view failed certification")
        views.append(view)

    return tuple(views)


def _is_currency_left_context(question: str, start: int) -> bool:
    return re.search(r"\$\s*$", question[:start]) is not None


def _candidate_integer_spans(question: str) -> list[tuple[int, int, int, str]]:
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
    for start, end, value, token in _candidate_integer_spans(question):
        if 0 <= value <= 999:
            replacement = integer_to_english(value)
            if replacement != token:
                return CertifiedEdit(start, end, token, replacement)
    for start, end, value, token in _candidate_currency_spans(question):
        if 0 <= value <= 999:
            return CertifiedEdit(start, end, token, _currency_words(value))
    return None


def _choose_equivalent_expression_edit(
    question: str,
    *,
    avoid_span: tuple[int, int] | None = None,
) -> CertifiedEdit | None:
    for start, end, value, token in _candidate_integer_spans(question):
        if avoid_span is not None and (start, end) == avoid_span:
            continue
        if 2 <= value <= 9999:
            left = value - 1
            return CertifiedEdit(start, end, token, f"({left} + 1)")
    for start, end, value, token in _candidate_currency_spans(question):
        if avoid_span is not None and (start, end) == avoid_span:
            continue
        if 2 <= value <= 9999:
            left = value - 1
            unit = "dollar" if value == 1 else "dollars"
            return CertifiedEdit(start, end, token, f"({left} + 1) {unit}")
    return None


def _generate_legacy_views(
    original: str,
    *,
    include_layout: bool,
    include_number_words: bool,
    include_equivalent_expression: bool,
) -> tuple[NumericMetamorphicView, ...]:
    views: list[NumericMetamorphicView] = [_identity_view(original)]

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
        expression_edit = _choose_equivalent_expression_edit(original, avoid_span=avoid)
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

    return tuple(views)


def generate_numeric_metamorphic_views(
    question: str,
    *,
    profile: str = "legacy",
    include_layout: bool = True,
    include_number_words: bool = True,
    include_equivalent_expression: bool = True,
) -> tuple[NumericMetamorphicView, ...]:
    """Generate certified relation views under the requested profile."""

    original = str(question).strip()
    if not original:
        raise NumericRelationError("question must not be empty")
    if profile not in NUMERIC_RELATION_PROFILES:
        raise NumericRelationError(
            f"unknown numeric relation profile {profile!r}; expected one of "
            f"{NUMERIC_RELATION_PROFILES}"
        )

    if profile == "entity_rename":
        views = generate_entity_renaming_views(original, num_variants=3)
    else:
        views = _generate_legacy_views(
            original,
            include_layout=include_layout,
            include_number_words=include_number_words,
            include_equivalent_expression=include_equivalent_expression,
        )

    if any(not view.verify() for view in views):
        raise NumericRelationError("at least one generated view failed certification")
    return tuple(views)


def relation_coverage(
    questions: Iterable[str],
    *,
    profile: str = "legacy",
) -> dict[str, int]:
    """Return a CPU-only coverage audit over a collection of questions."""

    if profile == "legacy":
        counts = {
            "questions": 0,
            "identity": 0,
            "layout_wrapper": 0,
            "number_representation": 0,
            "equivalent_quantity": 0,
            "four_view_questions": 0,
        }
    elif profile == "entity_rename":
        counts = {
            "questions": 0,
            "identity": 0,
            "entity_renaming_1": 0,
            "entity_renaming_2": 0,
            "entity_renaming_3": 0,
            "rename_eligible_questions": 0,
            "identity_only_fallback_questions": 0,
            "four_view_questions": 0,
        }
    else:
        raise NumericRelationError(f"unknown profile: {profile}")

    for question in questions:
        views = generate_numeric_metamorphic_views(question, profile=profile)
        counts["questions"] += 1
        subtypes = {view.relation_subtype for view in views}
        counts["identity"] += int("identity" in subtypes)
        for subtype in subtypes:
            if subtype in counts and subtype != "identity":
                counts[subtype] += 1
        counts["four_view_questions"] += int(len(views) == 4)
        if profile == "entity_rename":
            eligible = any(s.startswith("entity_renaming_") for s in subtypes)
            counts["rename_eligible_questions"] += int(eligible)
            counts["identity_only_fallback_questions"] += int(not eligible)
    return counts
