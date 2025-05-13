import asyncio

from config import logger  # Относительный импорт для config
from models import (
    AgentIdentifier,
    OrchestrationContext,
    FinalResponse
)  # Относительный импорт для models
from agents import (
    router_agent,
    weather_agent,
    translator_agent,
    tavily_agent,
    finalizer_agent
)  # Относительный импорт для agents

###############################################################################
# Оркестратор
###############################################################################
async def orchestrate_agents(user_query: str) -> FinalResponse:
    context = OrchestrationContext(original_user_query=user_query)
    current_query_for_router = user_query  # Начальный запрос для маршрутизатора

    while not context.processing_complete:
        # 1. Маршрутизатор решает, что делать
        # Теперь маршрутизатор использует context.original_user_query и context.accumulated_data через свои зависимости
        router_result = await router_agent.run(current_query_for_router, deps=context)
        action = router_result.output

        logger.debug(
            "[Router] action_type=%s thought=%s next_query=%s",
            action.action_type, action.thought, action.query_for_next_agent
        )
        
        # Обновляем счётчик used_agents
        context.used_agents[action.action_type] = context.used_agents.get(action.action_type, 0) + 1

        if action.action_type == AgentIdentifier.FINALIZER:
            final_result = await finalizer_agent.run(
                action.query_for_next_agent, deps=context  # Финализатор также использует полный контекст
            )
            context.processing_complete = True
            # Убедиться, что источники корректно заполняются самим finalizer_agent на основе context.accumulated_data
            return final_result.output

        elif action.action_type == AgentIdentifier.WEATHER:
            weather_result = await weather_agent.run(action.query_for_next_agent)
            context.accumulated_data.append({
                "source_agent": AgentIdentifier.WEATHER,
                "result": weather_result.output.model_dump(),
            })
            # Маршрутизатор определит следующий шаг на основе обновлённого контекста.
            # Нет необходимости изменять current_query_for_router, добавляя результаты здесь.
            # Подсказка маршрутизатора настроена на работу с accumulated_data.
            current_query_for_router = context.original_user_query  # Сбросить к исходному запросу для следующего шага

        elif action.action_type == AgentIdentifier.TRANSLATOR:
            translation_result = await translator_agent.run(action.query_for_next_agent)
            context.accumulated_data.append({
                "source_agent": AgentIdentifier.TRANSLATOR,
                "result": translation_result.output.model_dump(),
            })
            # Аналогично погоде, маршрутизатор будет использовать accumulated_data.
            current_query_for_router = context.original_user_query
            
        elif action.action_type == AgentIdentifier.TAVILY:
            tavily_result = await tavily_agent.run(action.query_for_next_agent)
            # print("Tavily agent result: %s", tavily_result.output)
            context.accumulated_data.append({
                "source_agent": AgentIdentifier.TAVILY,
                "result": tavily_result.output.model_dump(),
            })
            # Аналогично погоде, маршрутизатор будет использовать accumulated_data.
            current_query_for_router = context.original_user_query
            
        else:
            logger.error("Неизвестный action_type %s. Завершаем.", action.action_type)
            context.processing_complete = True
            # Резервный вариант, если каким-то образом получен неизвестный тип действия
            return FinalResponse(
                answer=f"Произошла ошибка: неизвестный тип действия от роутера: {action.action_type}.",
                sources=[d.get("source_agent", "unknown") for d in context.accumulated_data if d.get("source_agent")]
            )

    # Если вышли из цикла без явного вызова finalizer_agent (например, из-за ошибки)
    logger.warning("Цикл завершён без явного вызова finalizer_agent. Попытка финализации...")
    # Пытаемся выполнить финализацию, если цикл завершился неожиданно, но обработка не была отмечена как завершённая финализатором
    # Это может произойти, если маршрутизатор зацикливается или возникает ошибка до вызова финализатора.
    # Для надёжности мы можем вызвать финализатор здесь, если context ещё не завершён.
    if not any(item.get("source_agent") == AgentIdentifier.FINALIZER for item in context.accumulated_data):
        logger.info("Вызов finalizer_agent для завершения.")
        final_fallback_result = await finalizer_agent.run(context.original_user_query, deps=context)
        return final_fallback_result.output
    
    return FinalResponse(
        answer="Произошла непредвиденная ошибка: цикл обработки завершён без успешного вызова финализирующего агента.",
        sources=[d.get("source_agent", "unknown") for d in context.accumulated_data if d.get("source_agent")]
    )


async def main_orchestration() -> None:
    query = "Какая погода в Москве и переведи 'good morning' на русский?"
    # query = "What is the weather in London?"  # Тестовый запрос для погодного агента
    # query = "Какая погода в Берлине?"
    # query = "Переведи 'I love programming' на немецкий."
    # query = "Расскажи мне анекдот."

    logger.info(f"Запускаем оркестрацию для запроса: {query}")
    result = await orchestrate_agents(query)
    print("\n===== ИТОГОВЫЙ РЕЗУЛЬТАТ ОРКЕСТРАЦИИ =====")
    print(f"Ответ: {result.answer}")
    if result.sources:
        print(f"Источники: {', '.join(result.sources)}")
    else:
        print("Источники: не указаны")
    print("=========================================")


async def main_orchestration() -> None:
    query = "Какие новости в мире AI на сегодня?"
    logger.info(f"Запускаем оркестрацию для запроса: {query}")
    result = await orchestrate_agents(query)
    print("\n===== ИТОГОВЫЙ РЕЗУЛЬТАТ ОРКЕСТРАЦИИ =====")
    print(f"Ответ: {result.answer}")
    if result.sources:
        print(f"Источники: {', '.join(result.sources)}")
    else:
        print("Источники: не указаны")
    print("=========================================")


if __name__ == "__main__":
    asyncio.run(main_orchestration())
