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

_SESSION_STATE = {
    "offers": [],
    "current_offers": [],
}
SYSTEM = """
Ты — многоцелевой агент мониторинга цен на жильё.

Правила работы с URL и объявлениями:
1. Если в поле "Ссылки" передан полный URL, сначала вызови extract_offers
   для каждого URL.
2. Вызывай extract_offers только с полным URL, начинающимся с http://
   или https://.
3. Не вызывай extract_offers с текстом "по этой ссылке", "эта ссылка",
   "ссылка выше" или с пустой строкой.
4. Если поле "Ссылки" пустое, используй ранее сохранённые объявления.
   Не проси URL повторно, если в текущем запуске уже есть сохранённые offers.

Правила выбора tools:
5. Если пользователь просит найти/показать объявления дешевле или дороже
   указанной цены — обязательно вызови filter_offers.
6. Если пользователь просит найти объявления по слову, типу жилья,
   району или другому текстовому признаку — обязательно вызови filter_offers.
7. Если пользователь просит перевести/конвертировать цены в RUB, USD или EUR —
   обязательно вызови normalize_offers_currency. Никогда не конвертируй
   валюту самостоятельно в финальном тексте.
8. Если пользователь просит минимум, максимум, среднее, медиану
   или статистику — обязательно вызови compute_stats.
9. Для сложного запроса вызывай tools последовательно в нужном порядке:
   filter_offers → normalize_offers_currency → compute_stats.
10. Всегда используй результат последнего tool call в следующем шаге
    и в финальном ответе.
11. Не добавляй в итоговый ответ объявления, которых нет в результате
    последнего tool call.
12. Не утверждай, что объект дешевле/дороже лимита, если не сравнил
    его цену с этим лимитом. Не добавляй объектов, не прошедших filter_offers.
13. Не вызывай compute_stats, если пользователь явно не просил
    статистику, минимум, максимум, среднее или медиану.
14. Если filter_offers вернул пустой список, сообщи, что подходящих
    объявлений не найдено. Не вызывай normalize_offers_currency
    и compute_stats для пустого списка, если пользователь не попросил
    это явно.

    Аргументы filter_offers должны иметь такой JSON-вид:
    {{"max_price": 1800}}
    {{"min_price": 1000}}
    {{"text_contains": "pet friendly"}}

    Для max_price и min_price передавай только число без валюты:
    1800, а не "1800 USD", не {{"description": "1800 USD"}}
    и не любой другой объект.

КРИТИЧЕСКИЕ ПРАВИЛА ДОСТОВЕРНОСТИ:

- Используй в финальном ответе только объявления, которые вернул
  последний релевантный tool call.
- Никогда не создавай, не угадывай и не подставляй примерные ссылки,
  URL, названия квартир, цены или валюты.
- URL вида example.com, example1.com, example2.com, example3.com
  запрещены: их нельзя показывать пользователю.
- Если после filter_offers получился пустой список, ответь ровно:
  «Подходящих объявлений не найдено.»
  Не добавляй примеры, альтернативы, ссылки, статистику или цены.
- Если список пустой, не сообщай, что выполнялась конвертация цен:
  конвертировать нечего.
- Выводи статистику только если пользователь прямо попросил статистику,
  min/max/average/median.
- Значения статистики можно брать только из compute_stats, а цены —
  только из normalize_offers_currency или filter_offers. 

Всегда вызывай инструменты именованными аргументами строго по JSON-схеме.
filter_offers, normalize_offers_currency и compute_stats всегда работают
с текущим серверным набором объявлений. Не передавай в них списки offers
или prices: в JSON-схеме таких полей нет.

Отвечай кратко и по-русски. Не придумывай объявления, цены, валюты,
ссылки и результаты конвертации.
"""


@observe(name="extract_offers")
def _extract_offers_tool(url: str, limit: int = 10) -> Any:
    """Извлекает объявления по HTTP(S)-URL."""
    url = (url or "").strip()

    if not url.startswith(("http://", "https://")):
        return {
            "error": "Для extract_offers нужен полный URL, начинающийся с http:// или https://.",
            "offers": [],
        }

    result = extract_offers(url, limit=limit)
    offers = [offer.dict() for offer in result.offers]
    _SESSION_STATE["offers"] = list(offers)
    _SESSION_STATE["current_offers"] = list(offers)

    return {
        "offers": offers,
    }

@observe(name="filter_offers")
def _filter_offers_tool(
    max_price: Optional[int] = None,
    min_price: Optional[int] = None,
    text_contains: Optional[str] = None,
) -> list[dict]:
    offers = _SESSION_STATE.get("current_offers", [])

    result = filter_offers(
        offers=offers,
        max_price=max_price,
        min_price=min_price,
        text_contains=text_contains,
    )

    _SESSION_STATE["current_offers"] = list(result)
    return result


@observe(name="normalize_currency")
def _normalize_offers_currency_tool(
    target_currency: str = "RUB",
) -> list[dict]:
    offers = _SESSION_STATE.get("current_offers", [])

    result = normalize_offers_currency(
        offers=offers,
        target_currency=target_currency,
    )

    _SESSION_STATE["current_offers"] = result
    return result


@observe(name="compute_stats")
def _compute_stats_tool() -> dict:
    offers = _SESSION_STATE.get("current_offers", [])

    prices = [
        offer["price"]
        for offer in offers
        if isinstance(offer, dict) and offer.get("price") is not None
    ]

    if not prices:
        return {
            "min": None,
            "max": None,
            "avg": None,
            "median": None,
            "count": 0,
        }

    stats = compute_stats(prices)

    return {
        **stats,
        "count": len(prices),
        "currency": offers[0].get("currency", "USD"),
    }


class ExtractOffersArgs(BaseModel):
    url: str = Field(..., description="Страница с объявлениями")
    limit: int = Field(10, ge=1, le=10, description="Максимум объявлений: 1–10")


class FilterOffersArgs(BaseModel):
    min_price: Optional[int] = Field(
        None,
        description="Целое число без валюты, например 1800.",
    )
    max_price: Optional[int] = Field(
        None,
        description="Целое число без валюты, например 1800.",
    )
    text_contains: Optional[str] = Field(
        None,
        description='Подстрока в title, например "pet friendly".',
    )


class NormalizeCurrencyArgs(BaseModel):
    target_currency: str = Field(
        "RUB",
        description="Целевая валюта: RUB, USD или EUR.",
    )


class ComputeStatsArgs(BaseModel):
    pass


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

tools_without_extract = [
    tool
    for tool in tools
    if tool.name != "extract_offers"
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Задача пользователя: {question}\n"
     "Ссылки (может быть пусто): {urls}\n"
     "План действий:"
     "1) Если в поле «Ссылки» есть полный URL — извлеки объявления "
     "с каждой страницы через extract_offers.\n"
     "2) Если поле «Ссылки» пустое — работай с ранее сохранёнными "
     "объявлениями и не пытайся извлекать новые.\n"
     "3) Если просили фильтровать — применяй filter_offers."
     "4) Если просили в другой валюте — normalize_offers_currency."
     "5) Если нужна статистика — compute_stats."
     "6) Сформируй краткий отчёт. "
    "Добавляй ссылки только из результата последнего tool call. "
    "Если результатов нет — не добавляй ссылок."
     "Отвечай по-русски. Если данных нет — скажи об этом явно."),
    MessagesPlaceholder("agent_scratchpad"),
])

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.0,
    num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "12288")),)
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    handle_tool_error=True,
    verbose=False,
    return_intermediate_steps=True,
)

agent_without_extract = create_tool_calling_agent(
    llm,
    tools_without_extract,
    prompt,
)

executor_without_extract = AgentExecutor(
    agent=agent_without_extract,
    tools=tools_without_extract,
    handle_parsing_errors=True,
    handle_tool_error=True,
    verbose=False,
    return_intermediate_steps=True,
)

@observe(name="run_question")
def run_question(question: str, urls: list[str], max_items: int = 10,
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

    # Каждый новый вопрос начинает работу с объявлениями warmup.
    # list(...) создаёт отдельную копию списка.
    _SESSION_STATE["current_offers"] = list(
        _SESSION_STATE.get("offers", [])
    )

    active_executor = executor if urls else executor_without_extract

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
                result = active_executor.invoke(
                    {"question": question, "urls": urls, "max_items": max_items},
                    config=config
                )
        else:
            result = active_executor.invoke(
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

    langfuse_client.flush()

    return {
        "output": result.get("output", ""),
        "tools_used": tools_used,
    }