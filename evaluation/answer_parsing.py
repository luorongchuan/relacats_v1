"""Dataset answer extraction used by the RelaCaTS-v1 evaluation stages.

The original CaTS handlers remain the source of truth for dataset loading and
answer checking.  MathQA is the one exception in the *wire format* emitted by
the evaluator: its upstream handler stores option letters but its generic
multiple-choice prompt demonstrates ``Answer: (A)``.  The released MathQA
regular expression only accepts the bare form ``Answer: A``.  This module
keeps that handler untouched and adds the smallest compatibility layer needed
for the paper-facing option-letter prompt.

The parser intentionally does not infer answers from free-form explanations,
Markdown/LaTeX wrappers, or numeric values.  It only makes parentheses around
the explicit option marker optional.  All non-MathQA datasets continue to use
their original handler extraction (with the existing ARC numeric-label
fallback).
"""

from __future__ import annotations

import re
from typing import Any


UPSTREAM_HANDLER_PARSER_VERSION = "upstream-handler-v1"
MATHQA_PARSER_VERSION = "mathqa-option-letter-optional-parentheses-v1"

# Keep the upstream first-explicit-marker behavior.  ``Answer:`` is matched
# inside ``Final Answer:`` as it is by the released handler; the only semantic
# extension is allowing optional whitespace/parentheses around the letter.
_MATHQA_ANSWER_RE = re.compile(
    r"Answer\s*:\s*(?:\(\s*)?([A-Ea-e])(?:\s*\))?",
    flags=re.IGNORECASE,
)

_ARC_ANSWER_RE = re.compile(
    r"Answer\s*:\s*\(?\s*([A-Ea-e1-5])\s*\)?",
    flags=re.IGNORECASE,
)


def parser_version(dataset_name: str) -> str:
    """Return the extraction protocol recorded in an evaluation artifact."""

    if str(dataset_name).strip().lower().replace("-", "_") in {
        "math_qa",
        "mathqa",
    }:
        return MATHQA_PARSER_VERSION
    return UPSTREAM_HANDLER_PARSER_VERSION


def extract_mathqa_option_answer(text: str) -> str | None:
    """Extract the first explicit MathQA option letter.

    Accepted examples are ``Answer:A``, ``Answer: A`` and ``Answer: (A)``
    (case-insensitive).  The function deliberately returns ``None`` when the
    answer marker is absent instead of guessing a letter from the explanation.
    """

    match = _MATHQA_ANSWER_RE.search(str(text))
    return match.group(1).upper() if match else None


def _normalise_arc_label(label: str) -> str:
    label = label.upper()
    if label in {"1", "2", "3", "4", "5"}:
        return chr(ord("A") + int(label) - 1)
    return label


def extract_dataset_answer(dataset_name: str, text: str, handler: Any) -> Any:
    """Extract an answer while isolating the MathQA compatibility policy.

    For every dataset except MathQA this delegates to ``handler.extract_answer``
    exactly as the original evaluation code did.  ARC keeps the existing
    fallback for handlers that return no value for numeric labels.  MathQA is
    parsed locally so ``Answer: (A)`` is not discarded as invalid.
    """

    normalized_name = str(dataset_name).strip().lower()
    if normalized_name == "math_qa":
        return extract_mathqa_option_answer(text)

    extracted = handler.extract_answer(text)
    if extracted is not None:
        if normalized_name in {"arc_challenge", "arc_easy"}:
            return _normalise_arc_label(str(extracted))
        return extracted
    if normalized_name in {"arc_challenge", "arc_easy"}:
        matches = _ARC_ANSWER_RE.findall(str(text))
        if matches:
            return _normalise_arc_label(matches[-1])
    return None


__all__ = [
    "MATHQA_PARSER_VERSION",
    "UPSTREAM_HANDLER_PARSER_VERSION",
    "extract_dataset_answer",
    "extract_mathqa_option_answer",
    "parser_version",
]
