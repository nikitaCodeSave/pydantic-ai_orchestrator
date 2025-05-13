from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

###############################################################################
# Модели данных
###############################################################################
@dataclass
class AgentIdentifier:
    ROUTER: str = "router_agent"
    WEATHER: str = "weather_agent"
    TRANSLATOR: str = "translator_agent"
    TAVILY: str = "tavily_agent"
    FINALIZER: str = "finalizer_agent"


class WeatherResponse(BaseModel):
    location: str
    temperature: str
    conditions: str
    additional_info: Optional[str] = None


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str

# Добавил веб поиск
class TavilyResponse(BaseModel):
    context: str

class FinalResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)


@dataclass
class OrchestrationContext:
    original_user_query: str
    accumulated_data: List[Dict[str, Any]] = field(default_factory=list)
    processing_complete: bool = False
    used_agents: Dict[str, int] = field(default_factory=dict) # Added as per design


@dataclass
class OrchestratorAction:
    thought: str
    action_type: str # Should map to AgentIdentifier values
    query_for_next_agent: str

