import unittest

from src.eval.extractor import (
    extract_answer,
    extract_construction,
    parse_response_sections,
    prepare_response,
)


class ExtractorTests(unittest.TestCase):
    def test_parse_construction_response_sections_returns_question_bodies(self):
        response = "\n".join(
            [
                "## Solution to Question 1",
                "This is the answer to q1.",
                "",
                "## Solution to Question 2",
                "Here is q2 reasoning.",
                "<construct>",
                "[[1, 2], [3, 4]]",
                "</construct>",
            ]
        )

        parsed, error = parse_response_sections(response, mode="construction")

        self.assertIsNone(error)
        self.assertEqual("This is the answer to q1.", parsed.question_1)
        self.assertIn("Here is q2 reasoning.", parsed.question_2)
        construction, construction_error = extract_construction(parsed.question_2)
        self.assertIsNone(construction_error)
        self.assertEqual("[[1, 2], [3, 4]]", construction)
        self.assertEqual("This is the answer to q1.", extract_answer(response, mode="construction"))

    def test_prepare_construction_response_keeps_solution_when_construction_is_invalid(self):
        response = "\n".join(
            [
                "## Solution to Question 1",
                "valid q1",
                "",
                "## Solution to Question 2",
                "missing construction tags here",
            ]
        )

        prepared = prepare_response(response, mode="construction")

        self.assertEqual("valid q1", prepared.solution_text)
        self.assertIsNone(prepared.solution_error)
        self.assertEqual("question 2 solution must contain exactly one construction block", prepared.construction_error)

    def test_prepare_construction_response_rejects_construct_in_question_1_without_losing_construction(self):
        response = "\n".join(
            [
                "## Solution to Question 1",
                "<construct>[1]</construct>",
                "",
                "## Solution to Question 2",
                "ok",
                "<construct>",
                "[[1]]",
                "</construct>",
            ]
        )

        prepared = prepare_response(response, mode="construction")

        self.assertEqual("construct block is not allowed in question 1 solution", prepared.solution_error)
        self.assertEqual("[[1]]", prepared.construction)
        self.assertIsNone(prepared.construction_error)
        self.assertIsNone(extract_answer(response, mode="construction"))

    def test_prepare_plain_response_extracts_single_solution(self):
        response = "\n".join(
            [
                "## Solution",
                "This is the only solution.",
            ]
        )

        prepared = prepare_response(response, mode="plain")

        self.assertEqual(response, prepared.solution_text)
        self.assertIsNone(prepared.solution_error)
        self.assertIsNone(prepared.construction)
        self.assertIsNone(prepared.construction_error)

    def test_prepare_plain_response_accepts_missing_solution_heading(self):
        response = "This is a full solution without a markdown heading."

        prepared = prepare_response(response, mode="plain")

        self.assertEqual(response, prepared.solution_text)
        self.assertIsNone(prepared.solution_error)
        self.assertIsNone(prepared.construction)
        self.assertIsNone(prepared.construction_error)

    def test_prepare_plain_response_accepts_content_before_solution_heading(self):
        response = "\n".join(
            [
                "First solve the problem here.",
                "",
                "## Solution",
                "Then state the final answer.",
            ]
        )

        prepared = prepare_response(response, mode="plain")

        self.assertEqual(response, prepared.solution_text)
        self.assertIsNone(prepared.solution_error)
        self.assertIsNone(prepared.construction)
        self.assertIsNone(prepared.construction_error)

    def test_prepare_plain_response_rejects_construct_block(self):
        response = "\n".join(
            [
                "## Solution",
                "reasoning",
                "<construct>",
                "[1]",
                "</construct>",
            ]
        )

        prepared = prepare_response(response, mode="plain")

        self.assertEqual("construct block is not allowed in plain solution", prepared.solution_error)

    def test_extract_construction_requires_exactly_one_question_2_construct(self):
        construction, error = extract_construction(
            "\n".join(
                [
                    "first",
                    "<construct>[1]</construct>",
                    "second",
                    "<construct>[2]</construct>",
                ]
            )
        )

        self.assertIsNone(construction)
        self.assertEqual("question 2 solution must contain exactly one construction block", error)
