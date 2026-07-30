import json
import time
import asyncio
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.schemas import AgentState, PlanName, ModelTier
from app.config import settings
from app.tools.user import get_user_profile, GetUserProfileInput, cache_invalidate
from app.tools.subscription import update_subscription, UpdateSubscriptionInput

logger = logging.getLogger(__name__)

def _run_async(coro):
    """Safely run async coroutine from sync execution context."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return loop.run_in_executor(pool, lambda: asyncio.run(coro))
    except RuntimeError:
        return asyncio.run(coro)

class LangGraphAssistant:
    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self.checkpointer = MemorySaver()
        self._build_graph()

    def _build_graph(self):
        self.workflow.add_node("agent", self.call_model)
        self.workflow.add_node("get_user_profile_tool", self.run_get_user_profile)
        self.workflow.add_node("update_subscription_tool", self.run_update_subscription)
        self.workflow.add_node("parallel_tools_node", self.run_parallel_tools)
        
        self.workflow.set_entry_point("agent")
        
        self.workflow.add_conditional_edges(
            "agent",
            self.router,
            {
                "get_profile": "get_user_profile_tool",
                "update_sub": "update_subscription_tool",
                "parallel": "parallel_tools_node",
                "end": END
            }
        )
        
        self.workflow.add_edge("get_user_profile_tool", "agent")
        self.workflow.add_edge("update_subscription_tool", "agent")
        self.workflow.add_edge("parallel_tools_node", "agent")
        
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _prune_context(self, messages: list[dict], max_turns: int | None = None) -> list[dict]:
        """Adaptive Context Pruning: preserves first message & recent window, truncates old tool outputs."""
        max_turns = max_turns or settings.max_context_turns
        if len(messages) <= max_turns:
            return messages
        
        pruned: list[dict] = []
        for i, msg in enumerate(messages):
            if i == 0 or i >= len(messages) - (max_turns - 1):
                pruned.append(msg)
            elif msg.get("role") == "tool":
                pruned.append({"role": "tool", "content": "[truncated]"})
            else:
                pruned.append(msg)
        return pruned

    def call_model(self, state: AgentState) -> Dict[str, Any]:
        start = time.perf_counter()
        
        pruned = self._prune_context(state.messages)
        context_token_estimate = sum(len(json.dumps(m)) for m in pruned) // 4
        
        new_step_count = state.step_count + 1
        estimated_cost = state.total_cost_usd + 0.02
        
        # Model Tiering: route to FAST model for follow-ups, FULL for initial turn
        model_tier = ModelTier.FAST if new_step_count > 1 else ModelTier.FULL
        
        response_msg = {"role": "assistant", "content": "Evaluating next action..."}
        latency_ms = (time.perf_counter() - start) * 1000
        
        telemetry_entry = {
            "step": new_step_count,
            "latency_ms": round(latency_ms, 2),
            "context_tokens": context_token_estimate,
            "cost_usd": round(estimated_cost, 4),
            "model_tier": model_tier.value,
            "message_count": len(pruned),
        }
        
        logger.info(
            "step=%d latency_ms=%.1f context_tokens=%d cost=%.4f model=%s",
            new_step_count, latency_ms, context_token_estimate, estimated_cost, model_tier.value
        )
        
        return {
            "messages": pruned + [response_msg],
            "step_count": new_step_count,
            "total_cost_usd": estimated_cost,
            "telemetry": state.telemetry + [telemetry_entry],
        }

    def router(self, state: AgentState) -> str:
        if state.step_count >= settings.max_steps or state.total_cost_usd >= settings.cost_limit_usd:
            return "end"
        
        last_message = state.messages[-1] if state.messages else {}
        content = last_message.get("content", "")

        has_profile = "get_profile" in content
        has_sub = "update_sub" in content
        
        if has_profile and has_sub:
            return "parallel"
        elif has_profile:
            return "get_profile"
        elif has_sub:
            return "update_sub"
        
        return "end"

    def run_get_user_profile(self, state: AgentState) -> Dict[str, Any]:
        if not state.user_id:
            return {"errors": state.errors + [{"error": "Missing user_id"}]}
            
        try:
            inp = GetUserProfileInput(user_id=state.user_id)
            res = asyncio.run(get_user_profile(inp))
            if res.get("status") == "error":
                return {"errors": state.errors + [res]}
                
            return {"messages": state.messages + [{"role": "tool", "content": json.dumps(res)}]}
        except Exception as e:
            return {"errors": state.errors + [{"error": "validation_failure", "detail": str(e)}]}

    def run_update_subscription(self, state: AgentState) -> Dict[str, Any]:
        if not state.user_id:
            return {"errors": state.errors + [{"error": "Missing user_id"}]}

        try:
            inp = UpdateSubscriptionInput(user_id=state.user_id, plan_name=PlanName.PRO)
            res = asyncio.run(update_subscription(inp))
            
            if res.get("status") == "error":
                return {"errors": state.errors + [res]}
                
            cache_invalidate(state.user_id)
            
            return {
                "messages": state.messages + [{"role": "tool", "content": json.dumps(res)}],
                "current_plan": PlanName.PRO
            }
        except Exception as e:
            return {"errors": state.errors + [{"error": "validation_failure", "detail": str(e)}]}

    def run_parallel_tools(self, state: AgentState) -> Dict[str, Any]:
        """Executes get_user_profile and update_subscription in parallel via asyncio.gather."""
        if not state.user_id:
            return {"errors": state.errors + [{"error": "Missing user_id"}]}

        async def _run_both():
            profile_inp = GetUserProfileInput(user_id=state.user_id)
            sub_inp = UpdateSubscriptionInput(user_id=state.user_id, plan_name=PlanName.PRO)
            return await asyncio.gather(
                get_user_profile(profile_inp),
                update_subscription(sub_inp),
                return_exceptions=True
            )

        try:
            results = asyncio.run(_run_both())
            new_messages, new_errors = [], []
            new_plan = state.current_plan
            
            for res in results:
                if isinstance(res, Exception):
                    new_errors.append({"error": "parallel_execution", "detail": str(res)})
                elif isinstance(res, dict) and res.get("status") == "error":
                    new_errors.append(res)
                elif isinstance(res, dict):
                    new_messages.append({"role": "tool", "content": json.dumps(res)})
                    if res.get("idempotency_key"):
                        new_plan = PlanName.PRO
                        cache_invalidate(state.user_id)
            
            result = {"messages": state.messages + new_messages}
            if new_errors:
                result["errors"] = state.errors + new_errors
            if new_plan != state.current_plan:
                result["current_plan"] = new_plan
            return result
            
        except Exception as e:
            return {"errors": state.errors + [{"error": "parallel_dispatch_failure", "detail": str(e)}]}
