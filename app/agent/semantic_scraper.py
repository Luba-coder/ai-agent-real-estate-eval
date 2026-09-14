import os
import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langsmith import traceable

from .models import Offer, ExtractionResult
from .http_tools import http_get, strip_boilerplate, absolutize


MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

llm = ChatOllama(
    model=MODEL,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.0,
    num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "12288")),
)

parser = PydanticOutputParser(pydantic_object=ExtractionResult)

# Structured-output обёртка Ollama.
# Она передаёт ExtractionResult как JSON Schema и возвращает Pydantic-объект.
structured_llm = llm.with_structured_output(
    ExtractionResult,
    method="json_schema",
)


SYSTEM = (
    "Ты — ИИ-парсер объявлений о жилье. На вход даётся HTML страницы. "
    "Найди карточки объявлений и извлеки для каждой: заголовок, цену "
    "(целое число), валюту и ссылку. "
    "Если ссылка относительная — верни её как есть. "
    "Если объявлений нет — верни пустой список offers."
)


PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    (
        "human",
        "Базовый URL: {base_url}\n"
        "HTML (обрезано):\n{html}\n"
        "Требуемая схема JSON:\n{schema}\n"
        "Ограничения:\n"
        " - Цена только числом без пробелов.\n"
        " - Валюта как в исходном тексте (₽, RUB, $, USD, €, EUR).\n"
        " - Не более {limit} объявлений.\n"
        "Ответь строго согласно схеме JSON без комментариев.",
    ),
])


def extract_offers_from_html(
    base_url: str,
    html: str,
    limit: int = 50,
) -> ExtractionResult:
    """Извлекает список объявлений из HTML с помощью локальной Ollama-модели."""
    messages = PROMPT.format_messages(
        base_url=base_url,
        html=html,
        schema=parser.get_format_instructions(),
        limit=limit,
    )

    try:
        # Основной надёжный путь: Ollama JSON Schema → Pydantic ExtractionResult.
        data = structured_llm.invoke(messages)

    except Exception:
        # Fallback на прежнюю схему преподавателя:
        # обычный текстовый ответ + PydanticOutputParser.
        response = llm.invoke(messages)
        text = response.content

        try:
            data = parser.parse(text)

        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise

            data = parser.parse(match.group(0))

    data.offers = data.offers[:limit]

    data.offers = [
        Offer(
            title=offer.title,
            price=offer.price,
            currency=offer.currency,
            url=absolutize(base_url, offer.url),
        )
        for offer in data.offers
    ]

    return data


@traceable(name="semantic_scrape")
def extract_offers(url: str, limit: int = 50) -> ExtractionResult:
    """Загружает HTML, очищает его и извлекает объявления."""
    raw_html = http_get(url)
    cleaned_html = strip_boilerplate(raw_html)

    return extract_offers_from_html(
        base_url=url,
        html=cleaned_html,
        limit=limit,
    )