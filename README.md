# 🏠 Semantic Price Scraper Agent

AI-агент для мониторинга цен на жильё.  
Он извлекает объявления со страницы, фильтрует их, конвертирует цены и считает статистику. Агент работает локально через Ollama и использует LangChain, а также интегрирован с Langfuse и LangSmith для трейсинга.
---

## 📖 Возможности

- 🔎 Извлечение объявлений с сайтов (title, price, currency, url)
- 🧹 Фильтрация объявлений по цене и ключевым словам
- 💱 Конвертация цен в разные валюты (USD, RUB, EUR)
- 📊 Подсчёт статистики по ценам (min, max, avg, median)
- 💾 Запоминание последнего результата для повторного анализа
- 📈 **Полный трейсинг с Langfuse/LangSmith** - отслеживание всех вызовов LLM и инструментов
- 🔄 **Управление сессиями** - группировка связанных запросов
- 👥 **Поддержка user_id** - аналитика по пользователям
-Интеграционная оценка через DeepEval и eval_lib с локальной LLM-судьёй

---
## Требования

- Python 3.12 (проект проверялся с Python 3.12)
- Ollama
- Git
- Доступ к публичной странице с объявлениями

Для локальных evaluation-прогонов нужны модели Ollama:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

`qwen2.5:7b` используется агентом и LLM-as-a-Judge. `nomic-embed-text` нужен для метрик `eval_lib`, которым требуются embeddings.

---
## Установка

### 1. Клонируйте репозиторий

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd ai-agent-real-estate-eval
```

### 2. Создайте и активируйте виртуальное окружение

```bash
python -m venv .venv
source .venv/Scripts/activate
```

После активации в начале строки терминала появится:

```text
(.venv)
```

> Важно: активируйте это же окружение отдельно в каждом терминале, где запускаете Uvicorn или evaluation-скрипты.

### 3. Установите зависимости

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Создайте `.env`

Скопируйте пример:

```bash
cp .env.example .env
```

Для Langfuse/LangSmith добавьте в `.env` собственные ключи и URL.

## Запуск агента

Ollama должна быть запущена. Проверьте доступность моделей:

```bash
ollama list
```

### Терминал 1: FastAPI-агент

Откройте терминал в корне проекта, активируйте окружение и запустите API:

```bash
source .venv/Scripts/activate
uvicorn app.main:api --host 127.0.0.1 --port 8004 --reload --reload-dir app
```

После старта ожидается:

```text
INFO: Application startup complete.
```

Параметр `--reload-dir app` важен: изменения в `deepeval_lib/`, `eval_ai_library/` и `data/` не будут перезапускать сервер во время оценки.

Документация Swagger доступна по адресу:

```text
http://127.0.0.1:8004/docs
```

---

## 🛠 Используемые инструменты (Tools)

Агент использует 4 встроенных инструмента, каждый из которых обёрнут в Langfuse `@observe` для полного трейсинга:

### 1. `extract_offers`
Извлекает объявления с указанной страницы.  
**Вход:** URL, limit  
**Выход:** массив объявлений `{title, price, currency, url}`

### 2. `filter_offers`
Фильтрует объявления по цене и ключевым словам.  
**Вход:** список объявлений, min/max цена, текстовый фильтр  
**Выход:** отфильтрованный список

### 3. `normalize_offers_currency`
Конвертирует цены объявлений в выбранную валюту (RUB / USD / EUR).  
**Вход:** объявления + целевая валюта  
**Выход:** обновлённые объявления

### 4. `compute_stats`
Строит статистику по ценам.  
**Вход:** либо список цен, либо список объявлений  
**Выход:** `{min, max, avg, median}`

Агент автоматически сохраняет последний список объявлений в сессию и подставляет его в инструменты, если пользователь не передаёт `offers` вручную.

Набор объявлений хранится на стороне сервера:

```text
_SESSION_STATE["offers"]          — базовый результат извлечения
_SESSION_STATE["current_offers"]  — рабочая выборка текущего вопроса
```

После `filter_offers` и `normalize_offers_currency` обновляется `current_offers`. `compute_stats` считает статистику по ценам из текущей рабочей выборки.

---

### 🔄 Как работает агент

1. Клиент отправляет `POST /ask` с вопросом, списком `urls` при необходимости, а также опциональными `session_id` и `user_id`.

2. `run_question()` создаёт trace/observation для Langfuse и передаёт LangChain callbacks для связанного трейсинга.

3. Если `urls` содержит полный HTTP(S)-адрес, агенту доступен tool `extract_offers`. После извлечения базовый набор сохраняется в:

   ```text
   _SESSION_STATE["offers"]
   ```

4. Каждый новый вопрос начинает работу с копии базового набора:

   ```text
   _SESSION_STATE["current_offers"]
   ```

   Tools `filter_offers` и `normalize_offers_currency` изменяют именно текущую рабочую выборку. 
   `compute_stats` считает статистику по её ценам.

5. AgentExecutor возвращает финальный текст ответа и список фактически вызванных tools. API отдаёт их клиенту:

   ```json
   {
     "output": "ответ агента",
     "tools_used": [
       "filter_offers",
       "normalize_offers_currency"
     ]
   }
   ```

6. После завершения запроса Langfuse flush’ит накопившиеся trace events.
В Langfuse/LangSmith можно фильтровать и анализировать runs по `session_id`, `user_id`, tools и output.

---

## Пример workflow

Один URL извлекается в первом запросе. Последующие запросы используют сохранённые объявления и не должны повторно вызывать scraper.

```python
import requests

base_url = "http://127.0.0.1:8004"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "local-dev-key",
}

common = {
    "session_id": "demo-session",
    "user_id": "demo-user",
    "max_items": 10,
}

# 1. Извлечь и сохранить объявления.
requests.post(
    f"{base_url}/ask",
    headers=headers,
    json={
        **common,
        "question": "Извлеки объявления с указанной страницы.",
        "urls": ["https://www.rentalads.com/apartments-for-rent/ny/new-york/"],
    },
)

# 2. Отфильтровать сохранённые объявления.
requests.post(
    f"{base_url}/ask",
    headers=headers,
    json={
        **common,
        "question": "Покажи квартиры дешевле 2500 долларов.",
        "urls": [],
    },
)

# 3. Выполнить комбинированный запрос.
requests.post(
    f"{base_url}/ask",
    headers=headers,
    json={
        **common,
        "question": "Найди квартиры дешевле 2500 долларов и переведи цены в рубли.",
        "urls": [],
    },
)
```

## Evaluation

В репозитории есть два интеграционных сценария:

- `deepeval_lib/deepeval_run_evaluation.py` — оценка через DeepEval
- `eval_ai_library/eval_lib_run_evaluation.py` — оценка через `eval_lib`

Оба сценария:

1. Загружают `data/evaluation_dataset.xlsx`.
2. Выполняют warmup: извлекают объявления по URL один раз.
3. Исключают extraction-only кейс из основной оценки.
4. Передают последующие вопросы без URL, используя сохранённые объявления.
5. Оценивают финальные ответы и порядок вызова tools.

### Терминал 1: агент

```bash
source .venv/Scripts/activate
uvicorn app.main:api --host 127.0.0.1 --port 8004 --reload --reload-dir app
```

### Терминал 2: DeepEval

Откройте второй терминал в том же корне проекта и отдельно активируйте окружение:

```bash
source .venv/Scripts/activate
python deepeval_lib/deepeval_run_evaluation.py
```

DeepEval использует:

```text
AnswerRelevancyMetric
ToolCorrectnessMetric
TaskCompletionMetric
```

### Терминал 2: eval_lib

Не запускайте DeepEval и `eval_lib` одновременно, если обе оценки используют одну и ту же локальную Ollama на CPU: они будут конкурировать за модель и выполнятся значительно медленнее.

```bash
source .venv/Scripts/activate
python eval_ai_library/eval_lib_run_evaluation.py
```

`eval_lib` использует:

```text
AnswerRelevancyMetric
ToolCorrectnessMetric
TaskSuccessRateMetric
```

### Интерпретация результатов

- `ToolCorrectnessMetric` проверяет, совпадает ли фактическая последовательность tools с `expected_tools` из Excel.
- `AnswerRelevancyMetric` оценивает соответствие финального ответа вопросу.
- `TaskCompletionMetric`/`TaskSuccessRateMetric` оценивают, достигнута ли цель пользователя.
- Для кейсов с пустой выборкой LLM-as-a-Judge может занизить score, если судье не передан результат `filter_offers = []`. Это ограничение reference-free оценки: отсутствие подходящих объектов может быть корректным, но судья знает только вопрос и итоговый ответ.

---

## Langfuse и LangSmith

Вызовы `run_question()` и tools обёрнуты в Langfuse `@observe`; LangChain callbacks передают связанные observations в trace. В API можно передать `session_id` и `user_id`, чтобы группировать и фильтровать trace’ы.

Для демонстрации online evaluation в Langfuse полезно показать:

- три созданные online metrics/evaluators;
- custom LLM-as-a-Judge evaluator с собственным prompt;
- trace с входом, ответом, tools и score;
- `user_id`/metadata автора trace.

---

## 📡 API эндпоинты

### 1. `/ask` — задать вопрос агенту с трейсингом
Если передан URL, агент может вызвать `extract_offers`. Если `urls` пустой, агент использует сохранённый набор объявлений.

**Пример запроса:**
```json
{
  "question": "Найди квартиры дешевле 2500 долларов и переведи цены в рубли.",
  "urls": ["https://www.rentalads.com/apartments-for-rent/ny/new-york/"],
  "max_items": 10,
  "session_id": "demo-session",
  "user_id": "demo-user"
}
```

**Пример ответа:**
```json
{
  "output": "Квартиры дешевле 2500 долларов: ...",
  "tools_used": [
    "extract_offers",
    "filter_offers",
    "normalize_offers_currency"
  ]
}
```

### 2. `/session/new` — создать новую сессию

**GET**
```bash
curl http://localhost:8000/session/new
```

Ответ:
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Используйте этот ID в хедере X-Session-Id для группировки запросов"
}
```

### 3. `/monitor` — прямое извлечение с сайта

**POST**
```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: your-session-id" \
  -d '{
    "url": "https://www.rentalads.com/apartments-for-rent/ny/new-york/",
    "max_items": 50
  }'
```

---

## 🎯 Работа с сессиями

### Пример полного workflow с одной сессией:
```python
import requests
import json

# 1. Создаем новую сессию
session_resp = requests.get("http://localhost:8000/session/new")
session_id = session_resp.json()["session_id"]
print(f"Сессия создана: {session_id}")

headers = {
    "Content-Type": "application/json",
    "X-Session-Id": session_id,
    "X-User-Id": "user123"
}

# 2. Извлекаем объявления
resp1 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Извлеки квартиры с сайта",
        "urls": ["https://www.rentalads.com/apartments-for-rent/ny/new-york/"]
    }
)

# 3. Фильтруем по цене (в той же сессии)
resp2 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Покажи только квартиры дешевле 2000 долларов"
    }
)

# 4. Конвертируем в рубли (в той же сессии)
resp3 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Переведи цены в рубли"
    }
)

# 5. Получаем статистику (в той же сессии)
resp4 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Покажи статистику по ценам"
    }
)

## 🐳 Docker Compose варианты

### Cloud Langfuse (рекомендуется):
```
bash
docker compose up -d --build
```

---

## ⚠️ Ограничения

- Агент работает только с открытыми страницами (без авторизации)
- HTML структура сайтов может меняться, что сломает парсинг
- Валютные курсы фиксированные и предназначены для воспроизводимого учебного примера, а не для финансовых решений.
- Агент использует глобальный in-memory state и не рассчитан на параллельную multi-user нагрузку.
- Локальная Ollama на CPU может выполнять agent/evaluation медленно, особенно при длинных prompts и нескольких метриках.
- LLM-as-a-Judge не гарантирует идеальный результат и может некорректно оценить допустимый empty-result case без дополнительного контекста.

## 📚 Полезные ссылки

- [Ollama](https://ollama.com/)
- [Ollama: OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse LangChain integration](https://langfuse.com/docs/integrations/langchain)
- [LangChain documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DeepEval Documentation](https://deepeval.com/docs)