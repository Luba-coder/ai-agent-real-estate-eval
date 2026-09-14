"""
Кастомная модель DeepEval для локального запуска через Ollama.
"""

import os
from typing import Optional, Union

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM


class OllamaLLM(DeepEvalBaseLLM):
    """Локальная Ollama-модель в роли LLM-as-a-judge для DeepEval."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: Optional[int] = None,
    ):
        self._model_name = model or os.getenv("DEEPEVAL_OLLAMA_MODEL", os.getenv(
            "OLLAMA_MODEL", "qwen2.5:7b"
        ))
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        self.temperature = temperature
        self.num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "12288"))

        self.client = ChatOllama(
            model=self._model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )

        self.model = self.client

    def load_model(self):
        """Возвращает LangChain-клиент локальной Ollama."""
        return self.client

    def generate(
        self,
        prompt: str,
        schema: Optional[type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """
        Генерирует ответ judge-модели.

        Без schema возвращает строку.
        Со schema возвращает Pydantic-объект — это требуется части метрик DeepEval.
        """
        if schema is not None:
            structured_client = self.client.with_structured_output(schema)
            return structured_client.invoke(prompt)

        response = self.client.invoke(prompt)
        return response.content

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """Асинхронная генерация через Ollama."""
        if schema is not None:
            structured_client = self.client.with_structured_output(schema)
            return await structured_client.ainvoke(prompt)

        response = await self.client.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        """Название judge-модели для логов DeepEval."""
        return f"ollama:{self._model_name}"


def create_ollama_model(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
) -> OllamaLLM:
    """Фабрика локальной модели для DeepEval."""
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    judge_model = create_ollama_model()
    prompt = "Объясни одним коротким предложением, что такое RAG."

    print("Тестирование локальной Ollama judge-модели")
    print(f"Модель: {judge_model.get_model_name()}")
    print(f"Ollama URL: {judge_model.base_url}")
    print(f"Ответ: {judge_model.generate(prompt)}")