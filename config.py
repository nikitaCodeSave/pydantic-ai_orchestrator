import logging
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

###############################################################################
# Настройка логирования и провайдера
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ollama_provider = OpenAIProvider(base_url="http://localhost:11434/v1")
ollama_model = OpenAIModel(model_name="qwen2.5:32b", provider=ollama_provider)

###############################################################################
# Api ключи и другие конфигурации
###############################################################################
TAVILY_API_KEY = <API>  # Замените на ваш реальный API ключ
