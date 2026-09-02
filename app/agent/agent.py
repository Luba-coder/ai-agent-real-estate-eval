import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from .semantic_scraper import extract_offers
from .tools_ext import compute_stats, filter_offers, normalize_offers_currency

_SESSION_STATE = {"offers": []}

SYSTEM = (
    """Ты — многоцелевой агент мониторинга цен на жильё. 
    Если пользователь дал URL — сначала извлеки объявления (title, price, currency, url). 
    Дальше по вопросу выбери нужные действия: подсчитать статистику, отфильтровать по цене/ключевым словам, 
    Всегда вызывай инструменты именованными аргументами строго по их JSON-схеме
    если 'offers' не указан, инструмент использует последние извлечённые объявления.
    нормализовать валюту (в RUB, USD, EUR), сравнить несколько ссылок, собрать общий отчёт. 
    Отвечай кратко и по-русски, при необходимости показывай 3–5 примеров ссылок."""
)


@observe(name="extract_offers")
def _extract_offers_tool(url: str, limit: int = 50) -> list[dict]:
    res = extract_offers(url, limit=limit)
    out = [o.dict() for o in res.offers]
    _SESSION_STATE["offers"] = out
    return out


@observe(name="filter_offers")
def _filter_offers_tool(offers: Optional[List[Dict[str, Any]]] = None, min_price: Optional[int] = None,
                        max_price: Optional[int] = None,
                        text_contains: str = "") -> List[dict]:
    if offers is None:
        offers = _SESSION_STATE.get("offers", [])
    return filter_offers(offers, min_price, max_price, text_contains)


@observe(name="normalize_currency")
def _normalize_offers_currency_tool(offers: Optional[List[Dict[str, Any]]] = None,
                                    target_currency: str = "RUB") -> List[dict]:
    if offers is None:
        offers = _SESSION_STATE.get("offers", [])
    return normalize_offers_currency(offers, target_currency)


@observe(name="compute_stats")
def _compute_stats_tool(prices: Optional[List[int]] = None,
                        offers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    if prices is None and offers is None:
        offers = _SESSION_STATE.get("offers", [])
    if prices is None and offers is not None:
        prices = [int(o.get("price", 0)) for o in offers]
    return compute_stats(prices or [])


class ExtractOffersArgs(BaseModel):
    url: str = Field(..., description="Страница с объявлениями")
    limit: int = Field(50, description="Максимум объявлений")


class FilterOffersArgs(BaseModel):
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="Список объявлений {title, price, currency, url}. Если не задан — берём последние извлечённые.")
    min_price: Optional[int] = Field(None, description="Минимальная цена")
    max_price: Optional[int] = Field(None, description="Максимальная цена")
    text_contains: Optional[str] = Field(
        "", description="Подстрока в заголовке, например 'однокомнатная'")


class NormalizeCurrencyArgs(BaseModel):
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="Список объявлений. Если не задан — берём последние извлечённые.")
    target_currency: str = Field(
        "RUB", description="Целевая валюта: RUB|USD|EUR")


class ComputeStatsArgs(BaseModel):
    prices: Optional[List[int]] = Field(None, description="Список цен")
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="Можно передать объявления вместо списка цен")


tools = [
    StructuredTool.from_function(
        func=_extract_offers_tool,
        name="extract_offers",
        args_schema=ExtractOffersArgs,
        description="Извлечь объявления (title, price, currency, url) с указанной страницы."
    ),
    StructuredTool.from_function(
        func=_compute_stats_tool,
        name="compute_stats",
        args_schema=ComputeStatsArgs,
        description="Подсчитать статистику по списку цен или по объявлениям: {min,max,avg,median}."
    ),
    StructuredTool.from_function(
        func=_filter_offers_tool,
        name="filter_offers",
        args_schema=FilterOffersArgs,
        description="Отфильтровать объявления по min_price/max_price и/или по подстроке в заголовке."
    ),
    StructuredTool.from_function(
        func=_normalize_offers_currency_tool,
        name="normalize_offers_currency",
        args_schema=NormalizeCurrencyArgs,
        description="Сконвертировать цены объявлений в целевую валюту (RUB|USD|EUR)."
    ),
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Задача пользователя: {question}\n"
     "Ссылки (может быть пусто): {urls}\n"
     "План действий:"
     "1) Если есть ссылки — извлеки объявления с каждой страницы (extract_offers)."
     "2) Если просили фильтровать — применяй filter_offers."
     "3) Если просили в другой валюте — normalize_offers_currency."
     "4) Если нужна статистика — compute_stats."
     "5) Сформируй краткий отчёт и добавь 3–5 примеров ссылок."
     "Отвечай по-русски. Если данных нет — скажи об этом явно."),
    MessagesPlaceholder("agent_scratchpad"),
])

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.0)
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=False,
    return_intermediate_steps=True,
)


@observe(name="run_question")
def run_question(question: str, urls: list[str], max_items: int = 50,
                 session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    langfuse_client = get_client()

    handler = CallbackHandler()

    config = {
        "callbacks": [handler],
        "metadata": {},
        "tags": []
    }

    # Добавляем session_id и user_id в metadata для LangSmith
    if session_id:
        config["metadata"]["session_id"] = session_id
        config["tags"].append(f"session:{session_id}")
    if user_id:
        config["metadata"]["user_id"] = user_id
        config["tags"].append(f"user:{user_id}")

    # Начинаем трассу / наблюдение
    with langfuse_client.start_as_current_observation(
        as_type="span", name="agent_execution"
    ):
        # Пропагируем session_id и user_id (если они заданы)
        attrs: Dict[str, str] = {}
        if session_id:
            attrs["session_id"] = session_id
        if user_id:
            attrs["user_id"] = user_id

        if attrs:
            with propagate_attributes(**attrs):
                result = executor.invoke(
                    {"question": question, "urls": urls, "max_items": max_items},
                    config=config
                )
        else:
            result = executor.invoke(
                {"question": question, "urls": urls, "max_items": max_items},
                config=config
            )
            # В этот момент в result содержатся и вывод, и промежуточные шаги
    tools_used: List[str] = []
    intermediate_steps = result.get("intermediate_steps") or []
    for step in intermediate_steps:
        # Шаг имеет формат (AgentAction, tool_output)
        try:
            action = step[0]
            tool_name = getattr(action, "tool", None)
            if tool_name:
                tools_used.append(tool_name)
        except Exception:
            continue

    return {
        "output": result.get("output"),
        "tools_used": tools_used,
    }
