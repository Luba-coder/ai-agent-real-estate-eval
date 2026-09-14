# agent_connector.py
import requests
import time
import uuid
from typing import List, Dict, Any, Optional


class AgentConnector:
    """Коннектор к AI агенту с поддержкой тулов"""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        user_id: str = "lubov_konkina",
        session_id: Optional[str] = None,
        timeout: int = 120
    ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.user_id = user_id
        self.session_id = session_id or self._generate_session_id()
        self.timeout = timeout

    def _generate_session_id(self) -> str:
        """Генерирует UUID для сессии"""
        return str(uuid.uuid4())

    def _build_headers(self) -> Dict[str, str]:
        """Формирует заголовки для запроса"""
        return {
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'X-Session-Id': self.session_id,
            'X-User-Id': self.user_id,
            'X-API-Key': self.api_key
        }

    def query(
        self,
        question: str,
        urls: Optional[List[str]] = None,
        max_items: int = 10,
    ) -> Dict[str, Any]:
        """
        Отправляет запрос к агенту.

        Returns:
            Dict с полями:
            - output: текст ответа агента
            - tools_used: список использованных tools
            - raw_response: полный ответ API
        """
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

        payload = {
            "question": question,
            "urls": urls or [],
            "max_items": max_items,
            "session_id": self.session_id,
            "user_id": self.user_id,
        }

        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()
            data = response.json()

            return {
                "output": data.get("output", ""),
                "tools_used": data.get("tools_used", []),
                "raw_response": data,
            }

        except requests.exceptions.Timeout:
            return {
                "error": "Request timeout",
                "output": "",
                "tools_used": [],
            }

        except requests.exceptions.RequestException as error:
            return {
                "error": f"Request failed: {error}",
                "output": "",
                "tools_used": [],
            }

        except ValueError as error:
            return {
                "error": f"Invalid JSON in API response: {error}",
                "output": "",
                "tools_used": [],
            }

        except Exception as error:
            return {
                "error": f"Unexpected error: {error}",
                "output": "",
                "tools_used": [],
            }

    def batch_query(
        self,
        questions: List[str],
        urls_list: Optional[List[List[str]]] = None,
        expected_answers: Optional[List[str]] = None,
        delay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Батчевая обработка вопросов

        Args:
            questions: список вопросов
            urls_list: список списков URL для каждого вопроса (опционально)
            expected_answers: ожидаемые ответы для оценки (опционально)
            delay: задержка между запросами в секундах
        """
        results = []

        for i, question in enumerate(questions):
            print(f"📝 {i+1}/{len(questions)}: {question[:60]}...")

            # Получаем URLs для текущего вопроса
            urls = urls_list[i] if urls_list and i < len(urls_list) else None

            # Запрос к агенту
            agent_response = self.query(question, urls)

            # Формируем результат
            result = {
                'question': question,
                'output': agent_response.get('output', ''),
                'tools_used': agent_response.get('tools_used', []),
                'urls': urls,
                'error': agent_response.get('error')
            }

            # Выводим информацию об использованных тулах
            if result['tools_used']:
                tools_str = ', '.join(result['tools_used'])
                print(f"   🔧 Tools: {tools_str}")

            results.append(result)

            # Пауза между запросами
            if i < len(questions) - 1:
                time.sleep(delay)

        # Статистика
        successful = sum(1 for r in results if not r.get('error'))
        print(f"\n✅ Успешно: {successful}/{len(results)}")

        # Статистика по тулам
        all_tools = []
        for r in results:
            all_tools.extend(r.get('tools_used', []))
        if all_tools:
            unique_tools = set(all_tools)
            print(
                f"🔧 Использовано тулов: {len(all_tools)} (уникальных: {len(unique_tools)})")
            print(f"   Список: {', '.join(unique_tools)}")

        return results

    def new_session(self):
        """Создает новую сессию"""
        self.session_id = self._generate_session_id()
        print(f"🔄 Новая сессия: {self.session_id}")
