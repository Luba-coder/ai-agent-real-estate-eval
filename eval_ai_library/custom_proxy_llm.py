"""
Custom Proxy LLM Client for eval_lib
Implements CustomLLMClient interface for proxy server usage
"""

import os
from typing import Optional
from openai import AsyncOpenAI
from eval_lib import CustomLLMClient


class ProxyLLMClient(CustomLLMClient):
    """Custom LLM client for proxy server"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0
    ):
        self.model = model
        self.default_temperature = temperature
        self.api_key = api_key or os.getenv("PROXY_API_KEY")
        self.base_url = base_url or os.getenv("PROXY_BASE_URL")

        if not self.api_key:
            raise ValueError("PROXY_API_KEY must be set")
        if not self.base_url:
            raise ValueError("PROXY_BASE_URL must be set")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float
    ) -> tuple[str, Optional[float]]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            cost = None
            return text, cost
        except Exception as e:
            raise Exception(f"Proxy LLM error: {str(e)}")

    async def get_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small"
    ) -> tuple[list[list[float]], Optional[float]]:
        """Get embeddings from proxy server"""
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            cost = None
            return embeddings, cost
        except Exception as e:
            raise Exception(f"Proxy embeddings error: {str(e)}")

    def get_model_name(self) -> str:
        """Return model name"""
        return f"proxy:{self.model}"


def create_proxy_llm(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0
) -> ProxyLLMClient:

    return ProxyLLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )
