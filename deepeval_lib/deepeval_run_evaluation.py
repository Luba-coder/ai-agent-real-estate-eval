"""
Интеграционный тест для оценки агента
Метрики: AnswerRelevancy, ToolCorrectness, TaskCompletion
"""

import sys
import os
import time # noqa: E402

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval_lib.deepeval_custom_llm import create_ollama_model, OllamaLLM
from agent_connector import AgentConnector  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    TaskCompletionMetric
)  # noqa: E402
from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402
from typing import List, Union, Optional   # noqa: E402


def convert_tools_to_toolcalls(tools_list: List[str]) -> List[ToolCall]:
    """Преобразует список названий тулов в ToolCall объекты"""
    return [ToolCall(name=tool_name) for tool_name in tools_list]


def evaluate_agent(
    excel_path: str,
    agent_connector: AgentConnector,
    model: Union[str, OllamaLLM],
    urls: Optional[List[str]] = None,
    threshold: float = 0.7,
    sleep_time: float = 0.5,
    max_cases: Optional[int] = None,
    use_warmup: bool = True,
):
    """
    Оценка агента по трем метрикам
    """

    print("=" * 70)
    print("🚀 ИНТЕГРАЦИОННЫЙ ТЕСТ АГЕНТА")
    print("=" * 70)

    # ============= ШАГ 1: Загрузка датасета =============
    print("\n📂 Шаг 1: Загрузка датасета...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Ошибка загрузки датасета")
        return

    # Показываем превью
    parser.preview_dataset(df, n=2)

    # ============= ШАГ 2: Получение данных =============
    print("\n📝 Шаг 2: Извлечение данных из датасета...")

    pairs = parser.get_question_response_pairs(df)

    if max_cases is not None:
        pairs = pairs[:max_cases]
        print(f"🧪 Тестовый режим: запускается {len(pairs)} кейс(ов)")

    print(f"✅ Получено {len(pairs)} пар вопрос-ответ-тулы")

    if urls:
        print(f"🔗 URLs для агента: {urls}")

    # ============= ШАГ 3: Запросы к агенту =============
    # ============= ШАГ 3: Warmup и запросы к агенту =============
    print("\n🤖 Шаг 3: Получение ответов от агента...")

    test_cases = []

    if use_warmup and urls:
        print("\n🔄 Подготовка: извлекаем объявления по URL один раз...")

        warmup_response = agent_connector.query(
            question=(
                "Извлеки объявления с указанных ссылок и сохрани "
                "их для следующих запросов."
            ),
            urls=urls,
        )

        if warmup_response.get("error"):
            print(f"❌ Ошибка на этапе подготовки: {warmup_response['error']}")
            return

        warmup_tools = warmup_response.get("tools_used", [])

        print(f"✅ Warmup завершён. Tools: {warmup_tools}")

        if "extract_offers" not in warmup_tools:
            print(
                "⚠️ Предупреждение: warmup не вызвал extract_offers. "
                "Последующие кейсы могут работать с пустыми данными."
            )

        time.sleep(sleep_time)

    # Убираем extraction-only кейсы: их работу выполнил warmup.
    if use_warmup:
        pairs = [
            pair
            for pair in pairs
            if pair.get("expected_tools") != ["extract_offers"]
        ]
        print(f"ℹ️ После исключения extraction-only кейсов: {len(pairs)}")

    for i, pair in enumerate(pairs, 1):
        case_id = pair.get("case_id", f"case_{i:03d}")
        question = pair["question"]
        expected_response = pair["expected_response"]
        expected_tools = pair["expected_tools"]

        if use_warmup:
            expected_tools = [
                tool_name
                for tool_name in expected_tools
                if tool_name != "extract_offers"
            ]

        # Техническая подсказка добавляется только при warmup-режиме.
        # В Excel вопрос остаётся исходным — коллегам ничего менять не нужно.
        evaluation_question = question

        if use_warmup:
            evaluation_question = (
                "Контекст: список объявлений уже находится в памяти. "
                "Не извлекай объявления повторно и не запрашивай URL. "
                "Выполни исходный запрос пользователя с помощью нужных tools. "
                "Если в запросе есть ограничение цены — используй filter_offers. "
                "Если нужна валюта — используй normalize_offers_currency. "
                "Если нужна статистика — используй compute_stats.\n\n"
                f"{question}"
            )

        print(f"\n[{i}/{len(pairs)}] {case_id}: {question[:60]}...")

        response = agent_connector.query(
            question=evaluation_question,
            urls=None if use_warmup else urls,
        )

        if response.get("error"):
            print(f"   ❌ Ошибка: {response['error']}")
            continue

        answer = response.get("output", "")
        tools_used = response.get("tools_used", [])

        print(f"   Ответ: {answer}")
        print(f"   Тулы: {tools_used}")

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_response if expected_response else None,
            tools_called=convert_tools_to_toolcalls(tools_used),
            expected_tools=convert_tools_to_toolcalls(expected_tools),
        )

        test_cases.append(test_case)

        if i < len(pairs):
            time.sleep(sleep_time)

    print(f"\n✅ Создано {len(test_cases)} тест кейсов")

    if not test_cases:
        print("\n❌ Нет успешных ответов агента: DeepEval не запущен.")
        return

    # ============= ШАГ 4: Создание метрик =============
    print("\n📊 Шаг 4: Инициализация метрик...")

    metrics = [
        AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=True
        ),
        ToolCorrectnessMetric(
            threshold=threshold,
            include_reason=True,
            should_exact_match=True,  # только полное совпадение
            should_consider_ordering=True  # учитываем порядок вызова тулов
        ),
        TaskCompletionMetric(
            threshold=threshold,
            model=model,
            include_reason=True
        )
    ]

    print("✅ Метрики:")
    print("   • AnswerRelevancyMetric")
    print("   • ToolCorrectnessMetric")
    print("   • TaskCompletionMetric")

    # ============= ШАГ 5: Запуск оценки =============
    print("\n🧪 Шаг 5: Запуск оценки...")
    print("=" * 70)

    try:
        evaluate(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(
                run_async=False,
                max_concurrent=1,
            ),
        )

        print("\n" + "=" * 70)
        print("✅ ОЦЕНКА ЗАВЕРШЕНА")
        print("=" * 70)
        print("\n🌐 Результаты: https://app.confident-ai.com")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    agent = AgentConnector(
        endpoint_url=os.getenv(
            "AGENT_ENDPOINT_URL",
            "http://127.0.0.1:8004/ask",
        ),
        api_key=os.getenv("AGENT_API_KEY", "local-dev-key"),
        user_id=os.getenv("AGENT_USER_ID", "lubov_konkina"),
        session_id=os.getenv(
            "EVALUATION_SESSION_ID",
            "real-estate-eval-v1",
        ),
        timeout=int(os.getenv("AGENT_TIMEOUT_SECONDS", "900")),
    )

    print(f"⏱️ Таймаут AgentConnector: {agent.timeout} сек.")
    print(f"🌐 Endpoint: {agent.endpoint_url}")

    judge_model = create_ollama_model(
        model=os.getenv("DEEPEVAL_OLLAMA_MODEL", os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:7b",
        )),
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),
        temperature=0.0,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "12288")),
    )

    excel_path = os.getenv(
        "EVALUATION_DATASET_PATH",
        "data/evaluation_dataset.xlsx",
    )

    test_urls = [
        os.getenv(
            "EVALUATION_TEST_URL",
            "https://www.rentalads.com/for-rent/fl/miami/",
        )
    ]

    evaluate_agent(
        excel_path=excel_path,
        agent_connector=agent,
        model=judge_model,
        urls=test_urls,
        threshold=float(os.getenv("EVALUATION_THRESHOLD", "0.7")),
        sleep_time=float(os.getenv("EVALUATION_SLEEP_SECONDS", "1.0")),
        max_cases=None,
        use_warmup=True,
    )