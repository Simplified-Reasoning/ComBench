import unittest
from unittest.mock import patch

from src.eval.evaluator import EvaluationSuite
from src.eval.prime_code_executor import PrimeCodeExecutionResult


class FakeClient:
    def __init__(self, answer_response, proof_response, answer_usage=None, proof_usage=None):
        self.answer_response = answer_response
        self.proof_response = proof_response
        self.answer_usage = answer_usage
        self.proof_usage = proof_usage
        self.messages = []

    async def chat(self, messages):
        self.messages.append(messages)
        prompt = messages[0]["content"]
        is_proof = "<points>" in prompt
        content = self.proof_response if is_proof else self.answer_response
        usage = self.proof_usage if is_proof else self.answer_usage
        raw = {"choices": [{"message": {"content": content}}]}
        if usage is not None:
            raw["usage"] = usage
        return raw


class EvaluationSuiteTest(unittest.TestCase):
    @patch("src.eval.evaluator.run_prime_code_verifier")
    def test_check_scores_all_enabled_evaluators_and_normalizes_total(self, mock_run_prime_code_verifier):
        mock_run_prime_code_verifier.return_value = PrimeCodeExecutionResult(
            passed=True,
            score=1.0,
            info="True",
            metadata={"status": "passed", "score": 1.0},
        )
        client = FakeClient(
            answer_response="<thinking>ok</thinking>\n\\boxed{Correct}",
            proof_response="analysis\n<points>7 out of 7</points>",
        )
        suite = EvaluationSuite(client=client, timeout=6, semaphore=3, judge_answer=True)

        results = suite.check(
            data_list=[
                {
                    "id": "case-1",
                    "query": "q",
                    "instruction": "i",
                    "ref_answer": "a",
                    "grading_guidelines": "full proof gets 7",
                    "verify_code": "print(True)",
                }
            ],
            responses=[
                "\n".join(
                    [
                        "## Solution to Question 1",
                        "The answer is a.",
                        "",
                        "## Solution to Question 2",
                        "Construction below.",
                        "<construct>",
                        "[1, 2, 3]",
                        "</construct>",
                    ]
                )
            ],
        )

        self.assertEqual("scored", results[0]["answer"]["status"])
        self.assertEqual(1, results[0]["answer"]["raw_score"])
        self.assertEqual(1.0, results[0]["answer"]["normalized_score"])
        self.assertEqual("scored", results[0]["proof"]["status"])
        self.assertEqual(7, results[0]["proof"]["raw_score"])
        self.assertEqual(1.0, results[0]["proof"]["normalized_score"])
        self.assertEqual("scored", results[0]["construction"]["status"])
        self.assertEqual(1, results[0]["construction"]["raw_score"])
        self.assertEqual(1.0, results[0]["construction"]["normalized_score"])
        self.assertEqual(1.0, results[0]["total_score"])
        prompts = [item[0]["content"] for item in client.messages]
        answer_prompt = next(prompt for prompt in prompts if "**GOLDEN ANSWER**" in prompt)
        proof_prompt = next(prompt for prompt in prompts if "**SPECIFIC GRADING GUIDELINES**" in prompt)
        self.assertIn("The answer is a.", answer_prompt)
        self.assertIn("The answer is a.", proof_prompt)
        self.assertNotIn("Construction below.", answer_prompt)
        self.assertNotIn("<construct>", answer_prompt)

    @patch("src.eval.evaluator.run_prime_code_verifier")
    def test_llm_evaluators_record_token_usage(self, mock_run_prime_code_verifier):
        mock_run_prime_code_verifier.return_value = PrimeCodeExecutionResult(
            passed=True,
            score=1.0,
            info="True",
            metadata={"status": "passed", "score": 1.0},
        )
        client = FakeClient(
            answer_response="<thinking>ok</thinking>\n\\boxed{Correct}",
            proof_response="analysis\n<points>7 out of 7</points>",
            answer_usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            proof_usage={
                "prompt_tokens": 7,
                "completion_tokens": 11,
                "total_tokens": 18,
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        )
        suite = EvaluationSuite(client=client, timeout=6, semaphore=3, judge_answer=True)

        results = suite.check(
            data_list=[
                {
                    "id": "case-usage",
                    "query": "q",
                    "instruction": "i",
                    "ref_answer": "a",
                    "grading_guidelines": "full proof gets 7",
                    "verify_code": "print(True)",
                }
            ],
            responses=[
                "\n".join(
                    [
                        "## Solution to Question 1",
                        "The answer is a.",
                        "",
                        "## Solution to Question 2",
                        "<construct>",
                        "[1]",
                        "</construct>",
                    ]
                )
            ],
        )

        self.assertEqual(
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            results[0]["answer"]["details"]["usage"],
        )
        self.assertEqual(
            {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
            {
                key: value
                for key, value in results[0]["proof"]["details"]["usage"].items()
                if key != "completion_tokens_details"
            },
        )
        self.assertEqual(
            {
                "input_tokens": 7,
                "output_tokens": 11,
                "reasoning_tokens": 4,
                "response_tokens": 7,
                "total_tokens": 18,
                "raw": results[0]["proof"]["details"]["usage"],
            },
            results[0]["proof"]["details"]["usage_normalized"],
        )

    @patch("src.eval.evaluator.run_prime_code_verifier")
    def test_answer_judge_disabled_skips_answer_and_averages_proof_and_construction(self, mock_run_prime_code_verifier):
        mock_run_prime_code_verifier.return_value = PrimeCodeExecutionResult(
            passed=True,
            score=1.0,
            info="True",
            metadata={"status": "passed", "score": 1.0},
        )
        client = FakeClient(
            answer_response="<thinking>ok</thinking>\n\\boxed{Incorrect}",
            proof_response="analysis\n<points>6 out of 7</points>",
        )
        suite = EvaluationSuite(client=client, timeout=6, semaphore=3, judge_answer=False)

        results = suite.check(
            data_list=[
                {
                    "id": "case-2",
                    "query": "q",
                    "instruction": "i",
                    "ref_answer": "a",
                    "grading_guidelines": "full proof gets 7",
                    "verify_code": "print(True)",
                }
            ],
            responses=[
                "\n".join(
                    [
                        "## Solution to Question 1",
                        "partial proof",
                        "",
                        "## Solution to Question 2",
                        "<construct>",
                        "[[1]]",
                        "</construct>",
                    ]
                )
            ],
        )

        self.assertEqual("skipped", results[0]["answer"]["status"])
        self.assertEqual(6 / 7, results[0]["proof"]["normalized_score"])
        self.assertEqual(1.0, results[0]["construction"]["normalized_score"])
        self.assertEqual((6 / 7 + 1.0) / 2, results[0]["total_score"])
        mock_run_prime_code_verifier.assert_called_once()

    @patch("src.eval.evaluator.run_prime_code_verifier")
    def test_plain_mode_skips_construction_judge(self, mock_run_prime_code_verifier):
        client = FakeClient(
            answer_response="<thinking>ok</thinking>\n\\boxed{Incorrect}",
            proof_response="analysis\n<points>1 out of 7</points>",
        )
        suite = EvaluationSuite(client=client, timeout=6, semaphore=3, judge_answer=False)

        results = suite.check(
            data_list=[
                {
                    "id": "case-plain",
                    "query": "q",
                    "ref_answer": "a",
                    "grading_guidelines": "partial progress gets 1",
                }
            ],
            responses=[
                "\n".join(
                    [
                        "## Solution",
                        "plain solution",
                    ]
                )
            ],
        )

        self.assertEqual("skipped", results[0]["answer"]["status"])
        self.assertEqual("scored", results[0]["proof"]["status"])
        self.assertEqual("skipped", results[0]["construction"]["status"])
        self.assertEqual(1 / 7, results[0]["total_score"])
        mock_run_prime_code_verifier.assert_not_called()

    @patch("src.eval.evaluator.run_prime_code_verifier")
    def test_invalid_construction_scores_zero_without_blocking_solution_grading(self, mock_run_prime_code_verifier):
        client = FakeClient(
            answer_response="<thinking>ok</thinking>\n\\boxed{Incorrect}",
            proof_response="analysis\n<points>1 out of 7</points>",
        )
        suite = EvaluationSuite(client=client, timeout=6, semaphore=3, judge_answer=False)

        results = suite.check(
            data_list=[
                {
                    "id": "case-3",
                    "query": "q",
                    "instruction": "i",
                    "ref_answer": "a",
                    "grading_guidelines": "partial progress gets 1",
                    "verify_code": "print(True)",
                }
            ],
            responses=[
                "\n".join(
                    [
                        "## Solution to Question 1",
                        "partial proof",
                        "",
                        "## Solution to Question 2",
                        "no construction tags here",
                    ]
                )
            ],
        )

        self.assertEqual(1 / 7, results[0]["proof"]["normalized_score"])
        self.assertEqual(0.0, results[0]["construction"]["normalized_score"])
        self.assertEqual(
            "question 2 solution must contain exactly one construction block",
            results[0]["construction"]["reason"],
        )
        self.assertEqual((1 / 7 + 0.0) / 2, results[0]["total_score"])
        mock_run_prime_code_verifier.assert_not_called()

    @patch("src.eval.evaluator.run_prime_code_verifier")
    def test_missing_proof_fields_returns_error_and_no_total(self, mock_run_prime_code_verifier):
        mock_run_prime_code_verifier.return_value = PrimeCodeExecutionResult(
            passed=True,
            score=1.0,
            info="True",
            metadata={"status": "passed", "score": 1.0},
        )
        client = FakeClient(
            answer_response="<thinking>ok</thinking>\n\\boxed{Correct}",
            proof_response="analysis\n<points>7 out of 7</points>",
        )
        suite = EvaluationSuite(client=client, timeout=6, semaphore=3, judge_answer=False)

        results = suite.check(
            data_list=[
                {
                    "id": "case-4",
                    "query": "q",
                    "instruction": "i",
                    "ref_answer": "a",
                    "verify_code": "print(True)",
                }
            ],
            responses=[
                "\n".join(
                    [
                        "## Solution to Question 1",
                        "answer",
                        "",
                        "## Solution to Question 2",
                        "<construct>",
                        "[[1]]",
                        "</construct>",
                    ]
                )
            ],
        )

        self.assertEqual("error", results[0]["proof"]["status"])
        self.assertEqual("missing required field: grading_guidelines", results[0]["proof"]["reason"])
        self.assertIsNone(results[0]["total_score"])
        mock_run_prime_code_verifier.assert_called_once()

    @patch("src.eval.evaluator.run_prime_code_verifier")
    def test_invalid_answer_or_proof_output_is_an_error(self, mock_run_prime_code_verifier):
        mock_run_prime_code_verifier.return_value = PrimeCodeExecutionResult(
            passed=False,
            score=0.0,
            info="False",
            metadata={"status": "failed", "score": 0.0},
        )
        client = FakeClient(
            answer_response="not boxed",
            proof_response="analysis\n<points>2 out of 7</points>",
        )
        suite = EvaluationSuite(client=client, timeout=6, semaphore=3, judge_answer=True)

        results = suite.check(
            data_list=[
                {
                    "id": "case-5",
                    "query": "q",
                    "instruction": "i",
                    "ref_answer": "a",
                    "grading_guidelines": "full proof gets 7",
                    "verify_code": "print(True)",
                }
            ],
            responses=[
                "\n".join(
                    [
                        "## Solution to Question 1",
                        "answer",
                        "",
                        "## Solution to Question 2",
                        "<construct>",
                        "[[1]]",
                        "</construct>",
                    ]
                )
            ],
        )

        self.assertEqual("error", results[0]["answer"]["status"])
        self.assertEqual("invalid answer score output", results[0]["answer"]["reason"])
        self.assertEqual("error", results[0]["proof"]["status"])
        self.assertEqual("invalid proof score output", results[0]["proof"]["reason"])
        self.assertIsNone(results[0]["total_score"])
