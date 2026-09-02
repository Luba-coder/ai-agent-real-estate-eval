"""
Интеграционный тест для оценки агента
Метрики: AnswerRelevancy, ToolCorrectness, TaskCompletion
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval_custom_llm import create_proxy_model, ProxyLLM  # noqa: E402
from agent_connector import AgentConnector  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    TaskCompletionMetric
)  # noqa: E402
from deepeval import evaluate  # noqa: E402
from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402
from typing import List, Union, Optional   # noqa: E402
import time  # noqa: E402


def convert_tools_to_toolcalls(tools_list: List[str]) -> List[ToolCall]:
    """Преобразует список названий тулов в ToolCall объекты"""
    return [ToolCall(name=tool_name) for tool_name in tools_list]


def evaluate_agent(
    excel_path: str,
    agent_connector: AgentConnector,
    model: Union[str, ProxyLLM],
    urls: Optional[List[str]] = None,
    threshold: float = 0.7,
    sleep_time: float = 0.5
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
    print(f"✅ Получено {len(pairs)} пар вопрос-ответ-тулы")

    if urls:
        print(f"🔗 URLs для агента: {urls}")

    # ============= ШАГ 3: Запросы к агенту =============
    print("\n🤖 Шаг 3: Получение ответов от агента...")

    test_cases = []

    for i, pair in enumerate(pairs, 1):
        question = pair['question']
        expected_response = pair['expected_response']
        expected_tools = pair['expected_tools']

        print(f"\n[{i}/{len(pairs)}] {question[:60]}...")

        # Запрос к агенту
        response = agent_connector.query(question, urls=urls)

        if response.get('error'):
            print(f"   ❌ Ошибка: {response['error']}")
            continue

        # Извлекаем данные
        answer = response.get('output', '')
        tools_used = response.get('tools_used', [])

        print(f"   Ответ: {answer}")
        print(f"   Тулы: {tools_used}")

        # Создаем тест кейс
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_response if expected_response else None,
            tools_called=convert_tools_to_toolcalls(tools_used),
            expected_tools=convert_tools_to_toolcalls(expected_tools)
        )

        test_cases.append(test_case)

        if i < len(pairs):
            time.sleep(sleep_time)

    print(f"\n✅ Создано {len(test_cases)} тест кейсов")

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
            metrics=metrics
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

    # ============= КОНФИГУРАЦИЯ =============

    # Агент
    agent = AgentConnector(
        endpoint_url="http://5.11.83.110:8004/ask",
        api_key="80456142-5441-4469-b97f-1d72b7802a93",
        user_id="Luba Kon",
        session_id="a83d8b36-4ce6-4116-933f-f3b88303b5a2"
    )

    # Модель для метрик
    proxy_model = create_proxy_model(
        model="gpt-4o-mini",
        api_key="sk-proxy-d1yiB13gt0pyxdbeoaA5ZIv_hc0VJq4iBexunU2rdQg", 
        base_url="http://5.11.83.110:8000"
    )

    # Датасет
    excel_path = "data/evaluation_dataset.xlsx"

    # URLs для передачи агенту
    test_urls = [
        "https://www.rentalads.com/for-rent/fl/miami/"
    ]

    # ============= ЗАПУСК =============

    evaluate_agent(
        excel_path=excel_path,
        agent_connector=agent,
        model=proxy_model,  # или просто "gpt-4o-mini"
        urls=test_urls,
        threshold=0.7,
        sleep_time=0.5
    )
