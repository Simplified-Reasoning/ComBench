import unittest

from src.eval.extractor import extract_construction, parse_response_sections, prepare_response
from src.models.mock.model import MockResponseModel


class MockResponseModelTests(unittest.TestCase):
    def test_generate_emits_strict_construction_contract_when_record_has_instruction_and_verify_code(self):
        response = MockResponseModel().generate(
            prompt="ignored",
            record={
                "ref_answer": "42",
                "instruction": "Return a matrix",
                "verify_code": "print(True)",
                "ref_construction": "[[1, 2], [3, 4]]",
            },
        )

        parsed, error = parse_response_sections(response, mode="construction")

        self.assertIsNone(error)
        self.assertIn(r"\boxed{42}", parsed.question_1)
        construction, construction_error = extract_construction(parsed.question_2)
        self.assertIsNone(construction_error)
        self.assertEqual("[[1, 2], [3, 4]]", construction)

    def test_generate_emits_plain_solution_when_record_has_no_construction_metadata(self):
        response = MockResponseModel().generate(prompt="ignored", record={})

        prepared = prepare_response(response, mode="plain")

        self.assertIn("No answer available.", prepared.solution_text)
        self.assertIsNone(prepared.solution_error)
        self.assertIsNone(prepared.construction)
