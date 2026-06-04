import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

DEFAULT_MAX_RETRIES = 8
STREAM_DEBUG_PREVIEW_CHARS = 500


class MalformedStreamResponseError(requests.RequestException):
    def __init__(self, message: str, details: Dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class LlmConfig:
    api_key_env: str
    model_name: str
    base_url_env: str
    base_url: str | None
    temperature: float
    timeout: int
    max_retries: int = DEFAULT_MAX_RETRIES
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: Dict[str, Any] | None = None
    extra: Dict[str, Any] | None = None

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"missing API key env var: {self.api_key_env}")
        return key

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        env_val = os.environ.get(self.base_url_env)
        if not env_val:
            raise ValueError(f"missing base URL env var: {self.base_url_env}")
        return env_val

    def payload_defaults(self) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.stream:
            payload["stream"] = True
            payload["stream_options"] = self.stream_options or {"include_usage": True}
        if self.extra:
            payload.update(self.extra)
        return payload

    def validate(self) -> None:
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")


class LlmClient:
    def __init__(self, config: LlmConfig) -> None:
        config.validate()
        self.config = config

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(self._chat_sync, messages, kwargs)

    def _chat_sync(self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        url = self.config.resolved_base_url().rstrip("/") + "/chat/completions"
        payload = self.config.payload_defaults()
        payload["messages"] = messages
        if kwargs:
            payload.update({k: v for k, v in kwargs.items() if v is not None})
        headers = {
            "Authorization": f"Bearer {self.config.api_key()}",
            "Content-Type": "application/json",
        }
        last_exc: requests.RequestException | None = None
        for _ in range(self.config.max_retries):
            try:
                if payload.get("stream"):
                    return self._chat_stream_sync(url, headers, payload)
                return self._chat_non_stream_sync(url, headers, payload)
            except requests.RequestException as exc:
                last_exc = exc
                continue

        assert last_exc is not None
        raise last_exc

    def _chat_non_stream_sync(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _chat_stream_sync(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.config.timeout,
            stream=True,
        )
        response.raise_for_status()

        content_parts: List[str] = []
        usage: Dict[str, Any] | None = None
        saw_content = False
        saw_reasoning_content = False
        reasoning_content_characters = 0
        finish_reason: str | None = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                details = _stream_debug_payload(
                    saw_content=saw_content,
                    saw_reasoning_content=saw_reasoning_content,
                    reasoning_content_characters=reasoning_content_characters,
                    finish_reason=finish_reason,
                    usage=usage,
                )
                details.update(
                    {
                        "malformed_stream": True,
                        "json_error": str(exc),
                        "raw_line_preview": line[:STREAM_DEBUG_PREVIEW_CHARS],
                        "data_preview": data[:STREAM_DEBUG_PREVIEW_CHARS],
                    }
                )
                raise MalformedStreamResponseError("malformed stream JSON chunk", details) from exc
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice.get("finish_reason"))
            delta = choice.get("delta")
            if isinstance(delta, dict) and delta.get("content") is not None:
                content = str(delta.get("content"))
                if content:
                    saw_content = True
                    content_parts.append(content)
                if delta.get("reasoning_content") is None:
                    continue
            if isinstance(delta, dict) and delta.get("reasoning_content") is not None:
                reasoning_content = str(delta.get("reasoning_content"))
                if reasoning_content:
                    saw_reasoning_content = True
                    reasoning_content_characters += len(reasoning_content)
                continue
            message = choice.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                content = str(message.get("content"))
                if content:
                    saw_content = True
                    content_parts.append(content)

        content = "".join(content_parts)
        stream_debug = _stream_debug_payload(
            saw_content=saw_content,
            saw_reasoning_content=saw_reasoning_content,
            reasoning_content_characters=reasoning_content_characters,
            finish_reason=finish_reason,
            usage=usage,
        )
        return {
            "choices": [{"message": {"content": content}}],
            "usage": usage,
            "stream_debug": stream_debug,
        }


def _stream_debug_payload(
    *,
    saw_content: bool,
    saw_reasoning_content: bool,
    reasoning_content_characters: int,
    finish_reason: str | None,
    usage: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "saw_content": saw_content,
        "saw_reasoning_content": saw_reasoning_content,
        "reasoning_content_characters": reasoning_content_characters,
        "finish_reason": finish_reason,
        "usage": usage,
    }
