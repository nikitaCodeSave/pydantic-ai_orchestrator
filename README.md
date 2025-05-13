# Общая архитектура проекта

Проект представляет собой асинхронную Python-систему, организованную вокруг концепции LLM-агентов (Large Language Model Agents), которые взаимодействуют друг с другом через оркестратор. Основные модули:

- **`config.py`**: настраивает логирование и провайдера LLM (Ollama через локальный OpenAIProvider), хранит ключи API.

- **`models.py`**: описывает все Pydantic и dataclass модели, которые передаются между агентами и оркестратором: идентификаторы агентов, контекст оркестрации, структуры ответов.

- **`agents.py`**: определяет пять агентов:
  - `router_agent` – анализирует, какой агент вызывать дальше.
  - `weather_agent`, `translator_agent`, `tavily_agent` – выполняют специализированные задачи: получение погоды, перевод текста, веб-поиск.
  - `finalizer_agent` – собирает всё в единый ответ.

- **`orchestrator.py`**: реализует логику цикла оркестрации: принимает пользовательский запрос, последовательно вызывает router_agent → нужные агенты → finalizer_agent, аккумулируя результаты в `OrchestrationContext`.

Вся система работает на asyncio, что позволяет эффективно ждать ответов LLM и HTTP-запросов без блокировок.

## Описание ключевых функций и процессов

### 1. Оркестратор (`orchestrate_agents`)

- Создаёт `OrchestrationContext` с исходным запросом.
- В цикле:
  - Вызывает `router_agent.run`, передавая последний запрос и контекст.
  - На основе возвращённого `OrchestratorAction`:
    - Если `action_type == FINALIZER` → вызывает `finalizer_agent`, завершает цикл.
    - Иначе → вызывает соответствующий агент (weather, translator, tavily), добавляет результат в `context.accumulated_data` и продолжает цикл.
  - Если цикл завершился без finalizer (ошибка/зацикливание) → fallback-финализация.

```python
while not context.processing_complete:
    router_result = await router_agent.run(current_query_for_router, deps=context)
    action = router_result.output
    if action.action_type == AgentIdentifier.FINALIZER:
        return await finalizer_agent.run(...).output
    elif action.action_type == AgentIdentifier.WEATHER:
        # вызов weather_agent...
    # аналогично translator и tavily
```
### 2. Router-агент

**Задача**: решить, какой агент нужен далее, основываясь на:
- Исходном запросе пользователя
- Накопленных данных

**Выход**: `OrchestratorAction` с полями:
- `thought` – почему выбран именно этот шаг.
- `action_type` – один из `["weather_agent","translator_agent","tavily_agent","finalizer_agent"]`.
- `query_for_next_agent` – конкретный подзапрос.

**Ключевой момент**: в системе подсказок проверяются правила (например, не делать запрос погоды, если она уже есть) через `config`.

### 3. Специализированные агенты

#### WeatherAgent:
- Использует `duckduckgo_search_tool(max_results=1)` для получения актуальной погоды.
- Парсит температуру, условия, формирует `WeatherResponse`.

#### TranslatorAgent:
- Локальный LLM-перевод без внешних инструментов.
- Возвращает `TranslationResponse`, включая оригинал, перевод, языки.

#### TavilyAgent:
- Использует `tavily_search_tool(api_key=TAVILY_API_KEY)` для общего веб-поиска.
- Собирает «context» – текст найденной информации.

### 4. Финализирующий агент
- Собирает всё из `OrchestrationContext.accumulated_data`, составляет связный ответ.
- Возвращает `FinalResponse`, где:
  - `answer` – текст ответа.
  - `sources` – список агентов, чьи данные вошли в ответ.

## Технологический стек и обоснование

- **Python 3.10+** с **asyncio** для неблокирующей работы с сетью и LLM.
- **Pydantic** для строгой валидации моделей запросов/ответов.
- **dataclasses** для простого хранения контекста.
- **pydantic_ai** (внутренний фреймворк) для декларативного описания агентов и их системных подсказок.
- **OllamaProvider** (локальный LLM-сервер на qwen2.5:32b) вместо OpenAI API для приватности и оффлайн-разработки.
- **DuckDuckGo & Tavily** как «external tools» для веб-поиска, расширяя возможности LLM без прямого скрейпинга.

## Взаимодействие компонентов

| Компонент | Роль | Взаимодействие |
|-----------|------|----------------|
| `orchestrator.py` | Главный цикл, «дирижёр» | Вызывает всех агентов, хранит `OrchestrationContext` |
| `router_agent` | Решает, какой агент нужен дальше | Читает `context.original_user_query` и `accumulated_data` |
| `weather`/`translator`/`tavily` | Специализированные задачи | Возвращают валидированные ответы, накапливаются в `context` |
| `finalizer_agent` | Собирает и формирует итоговый ответ | Читает весь `context` и выдаёт `FinalResponse` |
| `config.py` | Настройка LLM-провайдера и логирования | Экспортирует `ollama_model`, `logger`, ключи API |

