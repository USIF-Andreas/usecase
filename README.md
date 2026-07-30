```
.
├── README.md
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── agent.py
│   ├── config.py
│   ├── schemas.py
│   └── tools
│       ├── __init__.py
│       ├── subscription.py
│       └── user.py
└── tests
    ├── __init__.py
    └── test_agent.py
```

### app/config.py
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings.
    
    Integrates with the scenario_resource_catalog:
    - model_family: "claude-3-5-sonnet" or "gpt-4o"
    - autonomy_level: "autonomous-within-budget"
    - cost_limit_usd: 0.50 (p95_latency_ms: 1500-8000)
    """
    model_family: str = "gpt-4o"
    autonomy_level: str = "autonomous-within-budget"
    max_steps: int = 12
    cost_limit_usd: float = 0.50

settings = Settings()
```

### app/schemas.py
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class AgentState(BaseModel):
    """
    Represents the state passed between nodes in the LangGraph workflow.
    Uses working (context window) memory to store conversation history and tool outputs.
    """
    messages: list = Field(default_factory=list, description="The list of messages in the conversation.")
    user_id: Optional[str] = Field(None, description="The active user ID for context tracking.")
    current_plan: Optional[str] = Field(None, description="The current subscription plan of the user.")
    errors: list = Field(default_factory=list, description="Tracked failure_modes like silent-compounding-error.")
```

### app/tools/user.py
```python
import httpx
from typing import Dict, Any

def get_user_profile(user_id: str) -> str:
    """
    Retrieves the user profile from the mock external-api-call service.
    
    The try-except block in `app/tools/user.py` inside `get_user_profile` which catches all exceptions and returns a flat string 'Error occurred' instead of structured validation details.
    """
    try:
        # Mock external API call to fetch user profile
        response = httpx.get(f"https://api.mockservice.local/v1/users/{user_id}")
        response.raise_for_status()
        return response.text
    except Exception as e:
        # The try-except block in `app/tools/user.py` inside `get_user_profile` which catches all exceptions and returns a flat string 'Error occurred' instead of structured validation details.
        return "Error occurred"
```

### app/tools/subscription.py
```python
import httpx
from pydantic import BaseModel, Field

class UpdateSubscriptionInput(BaseModel):
    """
    Input schema for updating a user's subscription plan.
    
    The function signature and Pydantic schema for `update_subscription` in `app/tools/subscription.py` where `plan_name: str` has no validation or enum constraints.
    """
    user_id: str = Field(..., description="The unique identifier of the user.")
    plan_name: str = Field(..., description="The name of the plan to subscribe to, e.g., free, pro, or enterprise.")

def update_subscription(user_id: str, plan_name: str) -> str:
    """
    Mutates the user's subscription state in the external database-query system.
    
    The `update_subscription` function signature in `app/tools/subscription.py` which lacks any idempotency token or transaction verification logic.
    
    The function signature and Pydantic schema for `update_subscription` in `app/tools/subscription.py` where `plan_name: str` has no validation or enum constraints.
    """
    # The `update_subscription` function signature in `app/tools/subscription.py` which lacks any idempotency token or transaction verification logic.
    payload = {"plan": plan_name}
    response = httpx.post(f"https://api.mockservice.local/v1/users/{user_id}/subscription", json=payload)
    response.raise_for_status()
    return response.json().get("status", "success")
```

### app/agent.py
```python
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.schemas import AgentState
from app.tools.user import get_user_profile
from app.tools.subscription import update_subscription

class LangGraphAssistant:
    """
    Manages the autonomous-loop (ReAct) workflow pattern using LangGraph.
    Coordinates routing and tool execution while tracking failure_modes like infinite-loop / no-termination.
    """
    
    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self._build_graph()
        
    def _build_graph(self):
        """Initializes nodes, edges, and routers for the assistant."""
        self.workflow.add_node("agent", self.call_model)
        self.workflow.add_node("get_user_profile_tool", self.run_get_user_profile)
        self.workflow.add_node("update_subscription_tool", self.run_update_subscription)
        
        self.workflow.set_entry_point("agent")
        
        # Router determines whether to call tools or terminate
        self.workflow.add_conditional_edges(
            "agent",
            self.router,
            {
                "get_profile": "get_user_profile_tool",
                "update_sub": "update_subscription_tool",
                "end": END
            }
        )
        
        self.workflow.add_edge("get_user_profile_tool", "agent")
        self.workflow.add_edge("update_subscription_tool", "agent")
        
        self.app = self.workflow.compile()

    def call_model(self, state: AgentState) -> Dict[str, Any]:
        """
        Queries the configured model_family (e.g., claude or gpt) using the current context window.
        Decides which tool to invoke or returns the final response.
        """
        pass

    def router(self, state: AgentState) -> str:
        """
        Routes the workflow execution path based on the model's last response.
        """
        pass

    def run_get_user_profile(self, state: AgentState) -> Dict[str, Any]:
        """Executes the get_user_profile tool and appends the result to the state."""
        pass

    def run_update_subscription(self, state: AgentState) -> Dict[str, Any]:
        """Executes the update_subscription tool and appends the result to the state."""
        pass
```

### app/main.py
```python
from fastapi import FastAPI, HTTPException
from app.schemas import AgentState
from app.agent import LangGraphAssistant

app = FastAPI(
    title="LangGraph Assistant Service",
    description="FastAPI backend exposing a LangGraph assistant with mock subscription management tools."
)

assistant = LangGraphAssistant()

@app.post("/chat")
async def chat(payload: AgentState):
    """
    Exposes the LangGraph assistant workflow over HTTP.
    
    Accepts the current conversation state and returns the updated state after running
    the autonomous-loop. Tracks realistic_trace_metrics such as latency and cost.
    """
    try:
        updated_state = assistant.app.invoke(payload.dict())
        return updated_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Returns the health status of the service and mock API connectivity."""
    return {"status": "healthy"}
```