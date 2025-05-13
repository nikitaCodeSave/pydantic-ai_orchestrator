from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool # Updated import to the factory function
from pydantic_ai.common_tools.tavily import tavily_search_tool
from config import ollama_model, logger, TAVILY_API_KEY # Relative import for config
from models import (
    AgentIdentifier,
    OrchestrationContext,
    OrchestratorAction,
    WeatherResponse,
    TranslationResponse,
    TavilyResponse,
    FinalResponse
) # Relative import for models

###############################################################################
# Определение агентов
###############################################################################
router_agent = Agent(
    model=ollama_model,
    deps_type=OrchestrationContext,
    name=AgentIdentifier.ROUTER,
    end_strategy="early",
    model_settings={"temperature": 0.2},
    output_type=OrchestratorAction,
    tools=[],  # router_agent не вызывает инструменты
    retries=3,
)


@router_agent.system_prompt
def router_system_prompt_func(ctx: RunContext[OrchestrationContext]) -> str:
    accumulated_data_str = ""
    if ctx.deps and ctx.deps.accumulated_data:
        accumulated_data_str = "\nСобранные данные:\n"
        for item in ctx.deps.accumulated_data:
            src = item.get("source_agent", "unknown")
            rslt = item.get("result", {})
            accumulated_data_str += f"- От {src}: {rslt}\n"
            
    logger.info("@router_agent.system_prompt: %s", accumulated_data_str)

    user_question = ctx.deps.original_user_query if ctx.deps else "Неизвестный запрос"

    # Improved prompt based on Step 003 design
    base_prompt = (
        "Вы — Router-агент. Ваша задача — проанализировать исходный запрос пользователя и уже собранные данные, "
        "чтобы решить, какой агент должен быть вызван следующим, или пора ли передать всё finalizer_agent.\n\n"
        f"Исходный запрос пользователя: {user_question}\n"
        f"{accumulated_data_str}\n"
        "Правила принятия решений:\n"
        "1. Внимательно изучите исходный запрос пользователя, чтобы понять общую цель."
        "2. Проверьте 'Собранные данные', чтобы увидеть, какая информация уже имеется."
        "3. Если для ответа на исходный запрос требуется информация о погоде (например, 'какая погода в X?'), и такой информации ещё нет в 'Собранных данных' для указанного места, то action_type должен быть 'weather_agent'. В 'query_for_next_agent' укажите только ту часть запроса, которая относится к погоде (например, 'погода в X')."
        "4. Если для ответа на исходный запрос требуется перевод текста (например, 'переведи Y на Z') или если собранные данные содержат текст НЕ на языке Исходного запроса пользователя, и такой перевод ещё не выполнен (или выполнен для другого текста/языка), то action_type должен быть 'translator_agent'. В 'query_for_next_agent' укажите текст для перевода и целевой язык (например, 'Y на Z')."
        "5. Если для ответа на исходный запрос требуется информация из интернета, и такой информации ещё нет в 'Собранных данных', то action_type должен быть 'tavily_agent'. В 'query_for_next_agent' укажите текст запроса для поиска (например, 'найди информацию о X')."
        "6. Если вся необходимая информация для ответа на исходный запрос пользователя уже собрана, или если ни weather_agent, ни translator_agent, ни tavily_agent не могут больше предоставить полезной информации, то action_type должен быть 'finalizer_agent'. В 'query_for_next_agent' передайте исходный запрос пользователя."
        "7. Не запрашивайте одну и ту же информацию повторно, если она уже есть в 'Собранных данных'.\n\n"
        "Вы должны вывести JSON-объект OrchestratorAction с обязательными полями:\n"
        "- 'thought': Ваши мысли о том, почему вы выбрали данный action_type."
        "- 'action_type': Одно из следующих значений: 'weather_agent', 'translator_agent', 'tavily_agent', 'finalizer_agent'."
        "- 'query_for_next_agent': Конкретный запрос для следующего агента, основанный на правилах выше.\n"
        "Нельзя вызывать инструменты напрямую.\n"
    )
    return base_prompt

weather_agent = Agent(
    model=ollama_model,
    name=AgentIdentifier.WEATHER,
    model_settings={"temperature": 0.3},
    output_type=WeatherResponse,
    tools=[duckduckgo_search_tool(max_results=1)], # Use the factory function
)

@weather_agent.system_prompt
def weather_system_prompt() -> str:
    return (
        "Вы — Weather-агент. Ваша задача — предоставить актуальную информацию о погоде для указанного местоположения. "
        "Используйте инструмент DuckDuckGoSearch, чтобы найти текущие погодные условия (температуру, описание погоды, например, солнечно, облачно, дождь). "
        "После получения результатов поиска, извлеките из них необходимую информацию и верните её в формате WeatherResponse. "
        "Поля для заполнения: location (местоположение, указанное в запросе), temperature (например, '25°C' или '77°F'), conditions (например, 'Солнечно', 'Облачно с прояснениями', 'Дождь'). "
        "Если возможно, добавьте краткую дополнительную информацию в 'additional_info', например, скорость ветра или влажность, если она легко доступна из результатов поиска. "
        "Убедитесь, что вы указываете местоположение в ответе. Если точные данные найти не удалось, укажите это в 'additional_info'."
    )


translator_agent = Agent(
    model=ollama_model,
    name=AgentIdentifier.TRANSLATOR,
    model_settings={"temperature": 0.1},
    output_type=TranslationResponse,
    tools=[],
)


@translator_agent.system_prompt
def translator_system_prompt() -> str:
    return (
        "Вы — Translation-агент. Переведите полученный текст на требуемый язык, "
        "вернув TranslationResponse. Определите язык-источник, если он не указан. Укажите исходный текст, переведенный текст, исходный язык и целевой язык в ответе."
    )


tavily_agent = Agent(
    model=ollama_model,
    name=AgentIdentifier.TAVILY,
    model_settings={"temperature": 0.5},
    output_type=TavilyResponse,
    tools=[tavily_search_tool(api_key=TAVILY_API_KEY)],
)

@tavily_agent.system_prompt
def tavily_system_prompt() -> str:
    return (
        "Вы — Tavily-агент. Найдите информацию для ответа на исходный запрос. Используйте инструмент TavilySearch, чтобы найти ответ в интернете. "
        "После получения результатов поиска, извлеките из них необходимую информацию и верните её в формате TavilyResponse. "
        "Поля для заполнения: context (текст, полученный из результата поиска). "
    )

finalizer_agent = Agent(
    model=ollama_model,
    deps_type=OrchestrationContext,
    name=AgentIdentifier.FINALIZER,
    model_settings={"temperature": 0.3},
    output_type=FinalResponse,
)

 
@finalizer_agent.system_prompt
def finalizer_system_prompt(ctx: RunContext[OrchestrationContext]) -> str:
    accumulated_data_str = ""
    source_agents_list = []
    if ctx.deps and ctx.deps.accumulated_data:
        accumulated_data_str = "Собранные данные для формирования ответа:\\n"
        for item in ctx.deps.accumulated_data:
            src = item.get("source_agent", "unknown")
            rslt = item.get("result", {})
            if src not in source_agents_list:
                source_agents_list.append(src)
            accumulated_data_str += f"- Информация от {src}: {rslt}\\n"
    logger.info("@finalizer_agent.system_prompt: %s", accumulated_data_str)

    original_query = ctx.deps.original_user_query if ctx.deps else "Неизвестный запрос"

    return (
        "Вы — Финализирующий агент. Ваша задача — собрать всю предоставленную информацию из 'Собранных данных' "
        "и на её основе дать полный и связный ответ на исходный вопрос пользователя. "
        "Ответ должен быть на том же языке, что и исходный вопрос.\\n\\n"
        f"Исходный вопрос пользователя: {original_query}\\n\\n"
        f"{accumulated_data_str}\\n"
        "Сформируйте ответ в виде JSON-объекта FinalResponse с полями 'answer' (текст ответа) и 'sources' (список агентов, предоставивших информацию, например, [\"weather_agent\", \"translator_agent\"])"
    )
