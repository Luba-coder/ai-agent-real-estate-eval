"""
Custom Ollama LLM Client for eval_lib.

Использует OpenAI-совместимый API локальной Ollama.
"""

import os
from typing import Optional

from openai import AsyncOpenAI
from eval_lib import CustomLLMClient


class OllamaLLMClient(CustomLLMClient):
    """Custom LLM client for local Ollama."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        embedding_model: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.default_temperature = temperature

        ollama_base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")

        # AsyncOpenAI ожидает OpenAI-compatible endpoint /v1.
        self.base_url = (
            ollama_base_url
            if ollama_base_url.endswith("/v1")
            else f"{ollama_base_url}/v1"
        )

        self.client = AsyncOpenAI(
            api_key="ollama",
            base_url=self.base_url,
        )

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> tuple[str, Optional[float]]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )

            text = (response.choices[0].message.content or "").strip()
            return text, None

        except Exception as error:
            raise RuntimeError(f"Ollama LLM error: {error}") from error

    async def get_embeddings(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> tuple[list[list[float]], Optional[float]]:
        try:
            response = await self.client.embeddings.create(
                model=model or self.embedding_model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]
            return embeddings, None

        except Exception as error:
            raise RuntimeError(f"Ollama embeddings error: {error}") from error

    def get_model_name(self) -> str:
        return f"ollama:{self.model}"


def create_proxy_llm(
    model: str = "qwen2.5:7b",
    embedding_model: str = "nomic-embed-text",
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> OllamaLLMClient:
    """
    Совместимое со старым импортом имя factory-функции.

    Хотя функция называется create_proxy_llm, она создаёт клиент Ollama.
    """
    return OllamaLLMClient(
        model=model,
        embedding_model=embedding_model,
        base_url=base_url,
        temperature=temperature,
    )