"""
Интеграционный тест для оценки агента с eval_lib
Метрики: AnswerRelevancy, ToolCorrectness, TaskSuccessRate
"""

import sys
import os

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio  # noqa: E402
import time  # noqa: E402
from typing import List, Optional  # noqa: E402
from eval_lib import (
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    TaskSuccessRateMetric,
    CustomLLMClient
)  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from agent_connector import AgentConnector  # noqa: E402
from custom_proxy_llm import create_proxy_llm  # noqa: E402


async def evaluate_agent_with_eval_lib(
    excel_path: str,
    agent_connector: AgentConnector,
    metrics_list: List[str],
    urls: Optional[List[str]] = None,
    model: str | CustomLLMClient = "gpt-4o-mini",
    threshold: float = 0.7,
    temperature: float = 0.5,
    sleep_time: float = 0.5,
    max_cases: Optional[int] = None,
    use_warmup: bool = True,
    verbose: bool = True,
    show_dashboard: bool = False,
    session_name: str = "Agent Evaluation"
):
    """
    Оценка агента с использованием eval_lib

    Args:
        excel_path: путь к датасету
        agent_connector: коннектор к агенту
        metrics_list: список метрик для оценки
        urls: список URL для передачи агенту
        model: модель для метрик
        threshold: порог для метрик
        temperature: температура для LLM
        sleep_time: задержка между запросами
        verbose: детальный вывод
        show_dashboard: показать dashboard
        session_name: название сессии
    """

    print("\n" + "=" * 70)
    print(f"🚀 {session_name.upper()}")
    print("=" * 70)

    # ============= ШАГ 1: Загрузка датасета =============
    print("\n📂 Шаг 1: Загрузка датасета...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Ошибка загрузки датасета")
        return None

    # Показываем информацию
    info = parser.validate_dataset(df)
    print(f"\n📊 Информация о датасете:")
    print(f"   • Всего строк: {info['total_rows']}")
    print(f"   • Валидных пар: {info['valid_pairs']}")
    if info['has_expected_tools_column']:
        print(f"   • Среднее количество тулов: {info['avg_tools_count']:.1f}")
        print(f"   • Уникальных тулов: {len(info['unique_tools'])}")

    # Превью
    parser.preview_dataset(df, n=2)

    # ============= ШАГ 2: Извлечение данных =============
    print("\n📝 Шаг 2: Извлечение данных из датасета...")

    pairs = parser.get_question_response_pairs(df)
    print(f"✅ Получено {len(pairs)} пар")

    if max_cases is not None:
        pairs = pairs[:max_cases]
        print(f"🧪 Тестовый режим: запускается {len(pairs)} кейс(ов)")

    if urls:
        print(f"🔗 URLs для агента: {urls}")

    # ============= ШАГ 3: Запросы к агенту =============
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
            print(
                f"❌ Ошибка на этапе подготовки: "
                f"{warmup_response['error']}"
            )
            return None

        warmup_tools = warmup_response.get("tools_used", [])
        print(f"✅ Warmup завершён. Tools: {warmup_tools}")

        if "extract_offers" not in warmup_tools:
            print(
                "⚠️ Warmup не вызвал extract_offers. "
                "Последующие кейсы могут работать с пустыми данными."
            )

        time.sleep(sleep_time)

    if use_warmup:
        pairs = [
            pair
            for pair in pairs
            if pair.get("expected_tools") != ["extract_offers"]
        ]
        print(f"ℹ️ После исключения extraction-only кейсов: {len(pairs)}")

    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        expected_response = pair["expected_response"]
        expected_tools = list(pair["expected_tools"])

        if use_warmup:
            expected_tools = [
                tool_name
                for tool_name in expected_tools
                if tool_name != "extract_offers"
            ]

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

        print(f"\n[{i}/{len(pairs)}] {question[:60]}...")

        # Запрос к агенту с URLs
        response = agent_connector.query(
            question=evaluation_question,
            urls=None if use_warmup else urls,
        )

        if response.get('error'):
            print(f"   ❌ {response['error']}")
            continue

        # Извлекаем данные
        answer = response.get('output', '')
        tools_used = response.get('tools_used', [])

        print(f"   ✅ Ответ: {answer[:50]}...")
        print(f"   🔧 Тулы: {tools_used}")

        # Создаем EvalTestCase
        test_case = EvalTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_response,
            tools_called=tools_used,  # список строк
            expected_tools=expected_tools  # список строк
        )

        test_cases.append(test_case)

        if i < len(pairs):
            time.sleep(sleep_time)

    print(f"\n✅ Создано {len(test_cases)} тест кейсов")

    # ============= ШАГ 4: Создание метрик =============
    print("\n📊 Шаг 4: Создание метрик...")
    print(f"   Метрики: {metrics_list}")
    print(f"   Порог: {threshold}")
    print(f"   Температура: {temperature}")

    # Определяем тип модели
    model_name = model.get_model_name() if isinstance(
        model, CustomLLMClient) else model
    print(f"   Модель: {model_name}")

    metrics = []
    metric_classes = {
        'answer_relevancy': AnswerRelevancyMetric,
        'tool_correctness': ToolCorrectnessMetric,
        'task_success_rate': TaskSuccessRateMetric
    }

    for metric_name in metrics_list:
        if metric_name == 'tool_correctness':
            metric = ToolCorrectnessMetric(
                threshold=threshold,
                verbose=verbose,
                exact_match=True,
                check_ordering=True
            )
            metrics.append(metric)
            print(f"   ✅ ToolCorrectnessMetric")
        elif metric_name in metric_classes:
            metric = metric_classes[metric_name](
                model=model,
                threshold=threshold,
                temperature=temperature,
                verbose=verbose
            )
            metrics.append(metric)
            print(f"   ✅ {metric_name}")

    print(f"\n✅ Создано {len(metrics)} метрик")

    # ============= ШАГ 5: Запуск оценки =============
    print("\n🧪 Шаг 5: Запуск оценки...")
    print(f"   Тест кейсов: {len(test_cases)}")
    print(f"   Метрик: {len(metrics)}")
    print("=" * 70)

    try:
        # Запускаем оценку
        results = await evaluate(
            test_cases=test_cases,
            metrics=metrics,
            verbose=verbose,
            show_dashboard=show_dashboard,
            session_name=session_name
        )

        print("\n" + "=" * 70)
        print("✅ ОЦЕНКА ЗАВЕРШЕНА")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ Ошибка при оценке: {e}")
        import traceback
        traceback.print_exc()
        return None


async def scenario_1():
    """Сценарий: оценка агента локальной Ollama-моделью."""

    print("\n" + "=" * 70)
    print("📋 СЦЕНАРИЙ: Оценка с локальной Ollama")
    print("=" * 70)

    agent = AgentConnector(
        endpoint_url=os.getenv(
            "AGENT_ENDPOINT_URL",
            "http://127.0.0.1:8004/ask",
        ),
        api_key=os.getenv("AGENT_API_KEY", "local-dev-key"),
        user_id=os.getenv("AGENT_USER_ID", "lubov_konkina"),
        session_id=os.getenv(
            "EVALUATION_SESSION_ID",
            "real-estate-eval-lib-v1",
        ),
        timeout=int(os.getenv("AGENT_TIMEOUT_SECONDS", "900")),
    )

    ollama_model = create_proxy_llm(
        model=os.getenv(
            "EVALLIB_OLLAMA_MODEL",
            os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        ),
        embedding_model=os.getenv(
            "OLLAMA_EMBED_MODEL",
            "nomic-embed-text",
        ),
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),
        temperature=0.0,
    )

    excel_path = os.getenv(
        "EVALUATION_DATASET_PATH",
        "data/evaluation_dataset.xlsx",
    )

    test_urls = [
        os.getenv(
            "EVALUATION_TEST_URL",
            "https://www.rentalads.com/apartments-for-rent/ny/new-york/",
        )
    ]

    metrics_to_use = [
        "answer_relevancy",
        "tool_correctness",
        "task_success_rate",
    ]

    return await evaluate_agent_with_eval_lib(
        excel_path=excel_path,
        agent_connector=agent,
        metrics_list=metrics_to_use,
        urls=test_urls,
        model=ollama_model,
        threshold=float(os.getenv("EVALUATION_THRESHOLD", "0.7")),
        temperature=0.0,
        sleep_time=float(os.getenv("EVALUATION_SLEEP_SECONDS", "1.0")),
        max_cases=None,
        use_warmup=True,
        verbose=True,
        show_dashboard=False,
        session_name="Agent Evaluation - Local Ollama",
    )


if __name__ == "__main__":

    asyncio.run(scenario_1())
