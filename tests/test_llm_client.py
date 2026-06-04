import os
import unittest
from unittest.mock import Mock, patch

import requests

from src.llm.client import DEFAULT_MAX_RETRIES, LlmClient, LlmConfig, MalformedStreamResponseError
from src.models.llm.model import build_llm_client


class LlmClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_api_key = os.environ.get("TEST_OPENAI_API_KEY")
        os.environ["TEST_OPENAI_API_KEY"] = "dummy-test-key"

    def tearDown(self):
        if self._old_api_key is None:
            os.environ.pop("TEST_OPENAI_API_KEY", None)
        else:
            os.environ["TEST_OPENAI_API_KEY"] = self._old_api_key

    async def test_build_llm_client_uses_default_max_retries(self):
        client = build_llm_client(
            model_name="gpt-test",
            api_key_env="TEST_OPENAI_API_KEY",
            base_url="https://example.com/v1",
        )

        self.assertEqual(DEFAULT_MAX_RETRIES, client.config.max_retries)

    async def test_build_llm_client_accepts_stream_options(self):
        client = build_llm_client(
            model_name="gpt-test",
            api_key_env="TEST_OPENAI_API_KEY",
            base_url="https://example.com/v1",
            stream=True,
            stream_options={"include_usage": True},
        )

        self.assertTrue(client.config.stream)
        self.assertEqual({"include_usage": True}, client.config.stream_options)

    async def test_chat_retries_until_success_within_budget(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        config = LlmConfig(
            api_key_env="TEST_OPENAI_API_KEY",
            model_name="gpt-test",
            base_url_env="IGNORED",
            base_url="https://example.com/v1",
            temperature=0.0,
            timeout=5,
            max_retries=3,
        )
        client = LlmClient(config)

        with patch(
            "src.llm.client.requests.post",
            side_effect=[
                requests.RequestException("first"),
                requests.RequestException("second"),
                response,
            ],
        ) as mock_post:
            raw = await client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual("ok", raw["choices"][0]["message"]["content"])
        self.assertEqual(3, mock_post.call_count)

    async def test_chat_stream_combines_chunks_and_preserves_usage(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            (
                'data: {"choices":[],'
                '"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'
            ),
            "data: [DONE]",
        ]

        config = LlmConfig(
            api_key_env="TEST_OPENAI_API_KEY",
            model_name="gpt-test",
            base_url_env="IGNORED",
            base_url="https://example.com/v1",
            temperature=0.0,
            timeout=5,
            max_retries=1,
            stream=True,
            stream_options={"include_usage": True},
        )
        client = LlmClient(config)

        with patch("src.llm.client.requests.post", return_value=response) as mock_post:
            raw = await client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual("hello", raw["choices"][0]["message"]["content"])
        self.assertEqual(
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            raw["usage"],
        )
        _, kwargs = mock_post.call_args
        self.assertTrue(kwargs["stream"])
        self.assertIn('"stream": true', kwargs["data"])
        self.assertIn('"stream_options": {"include_usage": true}', kwargs["data"])

    async def test_chat_stream_records_reasoning_debug_when_content_is_empty(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"","role":"assistant"},"index":0}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"thinking"},"index":0}]}',
            'data: {"choices":[{"delta":{"reasoning_content":" more"},"index":0}]}',
            (
                'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],'
                '"usage":{"prompt_tokens":4,"completion_tokens":8,"total_tokens":12}}'
            ),
            "data: [DONE]",
        ]

        config = LlmConfig(
            api_key_env="TEST_OPENAI_API_KEY",
            model_name="gpt-test",
            base_url_env="IGNORED",
            base_url="https://example.com/v1",
            temperature=0.0,
            timeout=5,
            max_retries=1,
            stream=True,
        )
        client = LlmClient(config)

        with patch("src.llm.client.requests.post", return_value=response):
            raw = await client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual("", raw["choices"][0]["message"]["content"])
        self.assertEqual(
            {
                "saw_content": False,
                "saw_reasoning_content": True,
                "reasoning_content_characters": len("thinking more"),
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
            },
            raw["stream_debug"],
        )

    async def test_chat_stream_retries_malformed_json_chunk_then_succeeds(self):
        malformed = Mock()
        malformed.raise_for_status.return_value = None
        malformed.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"reasoning_content":"broken}',
        ]
        successful = Mock()
        successful.raise_for_status.return_value = None
        successful.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]

        config = LlmConfig(
            api_key_env="TEST_OPENAI_API_KEY",
            model_name="gpt-test",
            base_url_env="IGNORED",
            base_url="https://example.com/v1",
            temperature=0.0,
            timeout=5,
            max_retries=2,
            stream=True,
        )
        client = LlmClient(config)

        with patch("src.llm.client.requests.post", side_effect=[malformed, successful]) as mock_post:
            raw = await client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual("ok", raw["choices"][0]["message"]["content"])
        self.assertEqual(2, mock_post.call_count)

    async def test_chat_stream_raises_malformed_error_with_details_after_retries(self):
        malformed = Mock()
        malformed.raise_for_status.return_value = None
        malformed.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"reasoning_content":"broken}',
        ]

        config = LlmConfig(
            api_key_env="TEST_OPENAI_API_KEY",
            model_name="gpt-test",
            base_url_env="IGNORED",
            base_url="https://example.com/v1",
            temperature=0.0,
            timeout=5,
            max_retries=2,
            stream=True,
        )
        client = LlmClient(config)

        with patch("src.llm.client.requests.post", return_value=malformed) as mock_post:
            with self.assertRaises(MalformedStreamResponseError) as exc_info:
                await client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(2, mock_post.call_count)
        details = exc_info.exception.details
        self.assertTrue(details["malformed_stream"])
        self.assertIn("Unterminated string", details["json_error"])
        self.assertIn("data:", details["raw_line_preview"])
        self.assertIn("reasoning_content", details["data_preview"])
        self.assertFalse(details["saw_content"])
        self.assertFalse(details["saw_reasoning_content"])

    async def test_chat_raises_last_request_exception_after_max_retries(self):
        config = LlmConfig(
            api_key_env="TEST_OPENAI_API_KEY",
            model_name="gpt-test",
            base_url_env="IGNORED",
            base_url="https://example.com/v1",
            temperature=0.0,
            timeout=5,
            max_retries=2,
        )
        client = LlmClient(config)

        with patch(
            "src.llm.client.requests.post",
            side_effect=[
                requests.RequestException("first"),
                requests.HTTPError("last"),
            ],
        ) as mock_post:
            with self.assertRaises(requests.HTTPError) as exc_info:
                await client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual("last", str(exc_info.exception))
        self.assertEqual(2, mock_post.call_count)

    async def test_config_rejects_invalid_max_retries(self):
        with self.assertRaises(ValueError):
            LlmClient(
                LlmConfig(
                    api_key_env="TEST_OPENAI_API_KEY",
                    model_name="gpt-test",
                    base_url_env="IGNORED",
                    base_url="https://example.com/v1",
                    temperature=0.0,
                    timeout=5,
                    max_retries=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
