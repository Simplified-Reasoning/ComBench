import unittest

from src.llm.token_usage import normalize_token_usage, sum_normalized_token_usage


class TokenUsageTests(unittest.TestCase):
    def test_normalizes_openai_usage_with_reasoning_tokens(self):
        usage = normalize_token_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 12},
            }
        )

        self.assertEqual(10, usage["input_tokens"])
        self.assertEqual(20, usage["output_tokens"])
        self.assertEqual(12, usage["reasoning_tokens"])
        self.assertEqual(8, usage["response_tokens"])
        self.assertEqual(30, usage["total_tokens"])

    def test_normalizes_gemini_usage_with_thought_tokens(self):
        usage = normalize_token_usage(
            {
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 9,
                    "thoughtsTokenCount": 6,
                    "totalTokenCount": 13,
                }
            }
        )

        self.assertEqual(4, usage["input_tokens"])
        self.assertEqual(9, usage["output_tokens"])
        self.assertEqual(6, usage["reasoning_tokens"])
        self.assertEqual(3, usage["response_tokens"])
        self.assertEqual(13, usage["total_tokens"])

    def test_sums_normalized_usage(self):
        usage = sum_normalized_token_usage(
            [
                {"prompt_tokens": 2, "completion_tokens": 8, "completion_tokens_details": {"reasoning_tokens": 5}},
                {"prompt_tokens": 3, "completion_tokens": 7},
            ]
        )

        self.assertEqual(
            {
                "input_tokens": 5,
                "output_tokens": 15,
                "reasoning_tokens": 5,
                "response_tokens": 10,
                "total_tokens": 20,
            },
            usage,
        )


if __name__ == "__main__":
    unittest.main()
