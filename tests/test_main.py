import argparse
import sys
import unittest

from src.main import _parse_args, _parse_csv_tokens, _parse_line_numbers


class ParseLineNumbersTests(unittest.TestCase):
    def test_parse_single_line(self):
        self.assertEqual(_parse_line_numbers("2"), [2])

    def test_parse_multiple_lines(self):
        self.assertEqual(_parse_line_numbers("1,2,3,10"), [1, 2, 3, 10])

    def test_parse_lines_deduplicates_while_preserving_order(self):
        self.assertEqual(_parse_line_numbers("3,1,3,2,1"), [3, 1, 2])

    def test_parse_lines_rejects_non_integer(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_line_numbers("1,a,3")

    def test_parse_lines_rejects_non_positive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_line_numbers("0")

    def test_parse_lines_rejects_empty_token(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_line_numbers("1,,3")

    def test_parse_args_supports_judge_answer(self):
        argv = sys.argv
        try:
            sys.argv = ["main.py", "--dataset", "data/sample.jsonl", "--judge-answer"]
            args = _parse_args()
        finally:
            sys.argv = argv

        self.assertTrue(args.judge_answer)

    def test_parse_args_supports_repeat(self):
        argv = sys.argv
        try:
            sys.argv = ["main.py", "--dataset", "data/sample.jsonl", "--repeat", "4"]
            args = _parse_args()
        finally:
            sys.argv = argv

        self.assertEqual(4, args.repeat)

    def test_parse_args_supports_force_attempt(self):
        argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--dataset",
                "data/sample.jsonl",
                "--force-attempt",
                "id-1__run_2,id-2__run_1,id-1__run_2",
            ]
            args = _parse_args()
        finally:
            sys.argv = argv

        self.assertEqual(["id-1__run_2", "id-2__run_1"], args.force_attempt)

    def test_parse_force_attempt_rejects_empty_token(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_csv_tokens("id-1,,id-2")

    def test_parse_args_defaults_repeat_to_one(self):
        argv = sys.argv
        try:
            sys.argv = ["main.py", "--dataset", "data/sample.jsonl"]
            args = _parse_args()
        finally:
            sys.argv = argv

        self.assertEqual(1, args.repeat)

    def test_parse_args_rejects_non_positive_repeat(self):
        argv = sys.argv
        try:
            sys.argv = ["main.py", "--dataset", "data/sample.jsonl", "--repeat", "0"]
            with self.assertRaises(SystemExit):
                _parse_args()
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
