from dataclasses import dataclass
from typing import Any, Dict, List

from src.llm.client import DEFAULT_MAX_RETRIES, LlmClient, LlmConfig


@dataclass(frozen=True)
class GenerationResult:
    content: str
    usage: Dict[str, Any] | None = None
    stream_debug: Dict[str, Any] | None = None


class LlmResponseModel:
    def __init__(self, client: LlmClient) -> None:
        self.client = client

    async def generate(self, prompt: str, record: Dict[str, Any]) -> GenerationResult:
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": prompt},
        ]
        raw = await self.client.chat(messages)
        return GenerationResult(
            content=raw["choices"][0]["message"]["content"],
            usage=raw.get("usage"),
            stream_debug=raw.get("stream_debug"),
        )


def build_llm_model(
    model_name: str,
    api_key_env: str,
    base_url: str | None = None,
    temperature: float | None = None,
    base_url_env: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stream: bool = False,
    stream_options: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> LlmResponseModel:
    return LlmResponseModel(
        build_llm_client(
            model_name=model_name,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            base_url_env=base_url_env,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            stream_options=stream_options,
            **kwargs,
        )
    )


def build_llm_client(
    model_name: str,
    api_key_env: str,
    base_url: str | None = None,
    temperature: float | None = None,
    base_url_env: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stream: bool = False,
    stream_options: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> LlmClient:
    config = LlmConfig(
        api_key_env=api_key_env,
        model_name=model_name,
        base_url_env=base_url_env or "OPENAI_BASE_URL",
        base_url=base_url,
        temperature=temperature or 0.0,
        timeout=60 if timeout is None else timeout,
        max_retries=max_retries if max_retries is not None else DEFAULT_MAX_RETRIES,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream,
        stream_options=stream_options,
        extra=kwargs or None,
    )
    return LlmClient(config)
