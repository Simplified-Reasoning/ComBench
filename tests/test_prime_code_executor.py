import unittest
from unittest.mock import patch

from src.eval.prime_code_executor import PrimeCodeDependencyError, run_prime_code_verifier
from src.prime_code import compute_score


class PrimeCodeModuleTest(unittest.TestCase):
    @patch("src.prime_code._load_check_correctness")
    def test_compute_score_normalizes_wrong_answer_metadata(self, mock_load_check_correctness):
        mock_load_check_correctness.return_value = lambda **kwargs: ([False], [{"error_message": "Wrong Answer"}])

        score, metadata = compute_score(
            completion="```python\nprint('False')\n```",
            test_cases={"inputs": ["payload\n"], "outputs": ["True"]},
            timeout=9,
        )

        self.assertEqual(0.0, score)
        self.assertEqual("wrong_answer", metadata["status"])
        self.assertEqual([False], metadata["raw_results"])
        self.assertEqual(0.0, metadata["score"])
        self.assertEqual("Wrong Answer", metadata["error_message"])

    @patch("src.prime_code._load_check_correctness")
    def test_compute_score_preserves_detailed_runtime_status(self, mock_load_check_correctness):
        mock_load_check_correctness.return_value = lambda **kwargs: (
            [-1],
            [
                {
                    "status": "worker_crash",
                    "error_message": "prime_code worker exited before reporting results",
                    "process_exitcode": -9,
                }
            ],
        )

        score, metadata = compute_score(
            completion="```python\nprint('True')\n```",
            test_cases={"inputs": ["payload\n"], "outputs": ["True"]},
            timeout=9,
        )

        self.assertEqual(0.0, score)
        self.assertEqual("worker_crash", metadata["status"])
        self.assertEqual([-1], metadata["raw_results"])
        self.assertEqual(-9, metadata["process_exitcode"])


class PrimeCodeExecutorTest(unittest.TestCase):
    @patch("src.eval.prime_code_executor.compute_score")
    def test_pass_result(self, mock_compute_score):
        mock_compute_score.return_value = (1.0, {"status": "passed", "score": 1.0})

        result = run_prime_code_verifier("print(True)", "payload\n", timeout=7)

        self.assertTrue(result.passed)
        self.assertEqual(1.0, result.score)
        self.assertEqual("True", result.info)
        self.assertEqual("passed", result.metadata["status"])

    @patch("src.eval.prime_code_executor.compute_score")
    def test_fail_result_uses_metadata_message(self, mock_compute_score):
        mock_compute_score.return_value = (
            0.0,
            {"status": "wrong_answer", "error_message": "Wrong Answer", "score": 0.0},
        )

        result = run_prime_code_verifier("print(False)", "payload\n", timeout=7)

        self.assertFalse(result.passed)
        self.assertEqual(0.0, result.score)
        self.assertEqual("wrong_answer", result.info)

    @patch("src.eval.prime_code_executor.compute_score")
    def test_runtime_error_result(self, mock_compute_score):
        mock_compute_score.return_value = (
            0.0,
            {"status": "runtime_error", "error": "ValueError('boom')", "score": 0.0},
        )

        result = run_prime_code_verifier("raise ValueError('boom')", "payload\n", timeout=7)

        self.assertFalse(result.passed)
        self.assertEqual("runtime_error", result.info)

    @patch("src.eval.prime_code_executor.compute_score")
    def test_missing_dependency_raises_clear_error(self, mock_compute_score):
        error = ModuleNotFoundError("No module named 'pyext'")
        error.name = "pyext"
        mock_compute_score.side_effect = error

        with self.assertRaises(PrimeCodeDependencyError) as ctx:
            run_prime_code_verifier("print(True)", "payload\n", timeout=7)

        self.assertIn("pyext", str(ctx.exception))
