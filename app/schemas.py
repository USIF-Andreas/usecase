from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class PlanName(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class ModelTier(str, Enum):
    FAST = "gpt-4o-mini"       # Tool formatting, summarization
    FULL = "gpt-4o"            # Reasoning, planning, user-facing generation

class AgentState(BaseModel):
    messages: list[Dict[str, Any]] = Field(default_factory=list, description="List of structured messages.")
    user_id: Optional[str] = Field(None, description="Active user ID.")
    current_plan: Optional[PlanName] = Field(None, description="Validated current plan name.")
    errors: list[Dict[str, Any]] = Field(default_factory=list, description="Tracked failure modes and tool errors.")
    step_count: int = Field(0, description="Tracks current ReAct loop iterations.")
    total_cost_usd: float = Field(0.0, description="Accumulated LLM cost in USD.")
    telemetry: list[Dict[str, Any]] = Field(default_factory=list, description="Per-step latency and token metrics.")
