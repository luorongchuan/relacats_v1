from __future__ import annotations

import unittest

from relacats_v1.evaluation.answer_parsing import (
    MATHQA_PARSER_VERSION,
    UPSTREAM_HANDLER_PARSER_VERSION,
    extract_dataset_answer,
    extract_mathqa_option_answer,
    parser_version,
)


class _BareLetterHandler:
    """Small stand-in for an upstream dataset handler."""

    def __init__(self, value=None):
        self.value = value
        self.calls = 0

    def extract_answer(self, text):
        del text
        self.calls += 1
        return self.value


class EvaluationAnswerParsingTests(unittest.TestCase):
    def test_mathqa_accepts_only_explicit_bare_or_parenthesized_letter(self):
        cases = {
            "Answer:A": "A",
            "Answer: (a)": "A",
            "Answer:  ( E )": "E",
            "Final Answer: (c)": "C",
            "The answer is probably A": None,
            "Answer: [A]": None,
            "Answer: 6": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_mathqa_option_answer(text), expected)

    def test_mathqa_dispatch_does_not_call_upstream_handler(self):
        handler = _BareLetterHandler(value="Z")
        self.assertEqual(
            extract_dataset_answer("math_qa", "Explanation\nAnswer: (d)", handler),
            "D",
        )
        self.assertEqual(handler.calls, 0)

    def test_non_mathqa_dispatch_preserves_handler_behavior(self):
        handler = _BareLetterHandler(value=None)
        self.assertIsNone(
            extract_dataset_answer("object_counting", "Answer: (A)", handler)
        )
        self.assertEqual(handler.calls, 1)

    def test_arc_numeric_fallback_remains_local_to_arc(self):
        handler = _BareLetterHandler(value=None)
        self.assertEqual(
            extract_dataset_answer("arc_easy", "Answer: (2)", handler), "B"
        )
        self.assertIsNone(
            extract_dataset_answer("object_counting", "Answer: (2)", handler)
        )

    def test_parser_versions_are_explicit(self):
        self.assertEqual(parser_version("math_qa"), MATHQA_PARSER_VERSION)
        self.assertEqual(parser_version("MathQA"), MATHQA_PARSER_VERSION)
        self.assertEqual(parser_version("arc_easy"), UPSTREAM_HANDLER_PARSER_VERSION)


if __name__ == "__main__":
    unittest.main()
