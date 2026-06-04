import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data_process.check_ref_construction as checker
from src.eval.prime_code_executor import PrimeCodeExecutionResult


class CheckRefConstructionTest(unittest.TestCase):
    def test_check_record_reports_missing_verify_code(self):
        ok, info, record_id, verify_len, construction_len, detail = checker._check_record(
            obj={"id": "case-1", "ref_construction": [1, 2, 3]},
            line_no=1,
            timeout=5,
        )

        self.assertFalse(ok)
        self.assertEqual("missing verify_code", info)
        self.assertEqual("case-1", record_id)
        self.assertEqual(0, verify_len)
        self.assertGreater(construction_len, 0)
        self.assertEqual("missing verify_code", detail["info"])

    @patch("data_process.check_ref_construction.run_prime_code_verifier")
    def test_main_prints_pass_for_successful_record(self, mock_run_prime_code_verifier):
        mock_run_prime_code_verifier.return_value = PrimeCodeExecutionResult(
            passed=True,
            score=1.0,
            info="True",
            metadata={"status": "passed", "score": 1.0},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "sample.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "case-1",
                        "verify_code": "print(True)",
                        "ref_construction": "[1, 2, 3]",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with patch("sys.argv", ["check_ref_construction.py", "--file", str(dataset_path)]):
                with patch("sys.stdout", stdout):
                    checker.main()

        self.assertIn("PASS (True)", stdout.getvalue())

    @patch("data_process.check_ref_construction.run_prime_code_verifier")
    def test_main_prints_failure_details(self, mock_run_prime_code_verifier):
        mock_run_prime_code_verifier.return_value = PrimeCodeExecutionResult(
            passed=False,
            score=0.0,
            info="worker_crash",
            metadata={
                "status": "worker_crash",
                "score": 0.0,
                "error_message": "prime_code worker exited before reporting results",
                "exception_type": "RuntimeError",
                "prime_code_error_location": {
                    "file": "src/prime_code/testing_util.py",
                    "line": 122,
                    "function": "run_test",
                    "code": "_set_timeout_alarm(timeout)",
                },
                "process_exitcode": -9,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "sample.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "case-2",
                        "verify_code": "print(False)",
                        "ref_construction": "[1, 2, 3]",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with patch("sys.argv", ["check_ref_construction.py", "--file", str(dataset_path)]):
                with patch("sys.stdout", stdout):
                    with self.assertRaises(SystemExit) as ctx:
                        checker.main()

        self.assertEqual(1, ctx.exception.code)
        output = stdout.getvalue()
        self.assertIn("FAIL (worker_crash)", output)
        self.assertIn("metadata_summary:", output)
        self.assertIn("prime_code worker exited before reporting results", output)
        self.assertIn("\"exception_type\": \"RuntimeError\"", output)
        self.assertIn("prime_code_error_location: src/prime_code/testing_util.py:122 in run_test", output)
        self.assertIn("prime_code_error_code: _set_timeout_alarm(timeout)", output)
        self.assertIn("\"prime_code_error_location\": {", output)
        self.assertIn("\"process_exitcode\": -9", output)
