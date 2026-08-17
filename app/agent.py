import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.schemas import AgentState, PlanName, ModelTier
from app.config import settings
from app.tools.user import get_user_profile, GetUserProfileInput, cache_invalidate
from app.tools.subscription import update_subscription, UpdateSubscriptionInput

logger = logging.getLogger(__name__)

TOOL_INTENT_KEYWORDS = {
    "get_profile": ["get_profile", "profile", "user_info", "view plan", "my profile", "account info"],
    "update_sub": ["update_sub", "upgrade", "change plan", "subscribe", "subscription"],
}

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

    def _extract_requested_plan(self, content: str) -> Optional[PlanName]:
        content_lower = content.lower()
        if "enterprise" in content_lower:
            return PlanName.ENTERPRISE
        elif "free" in content_lower:
            return PlanName.FREE
        elif "pro" in content_lower or "upgrade" in content_lower or "change plan" in content_lower:
            return PlanName.PRO
        return None

    def _extract_user_intent(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _detect_intent_from_message(self, content: str) -> tuple[bool, bool]:
        content_lower = content.lower()
        has_profile = any(k in content_lower for k in TOOL_INTENT_KEYWORDS["get_profile"])
        has_sub = any(k in content_lower for k in TOOL_INTENT_KEYWORDS["update_sub"])
        return has_profile, has_sub

    async def call_model(self, state: AgentState) -> Dict[str, Any]:
        start = time.perf_counter()
        
        pruned = self._prune_context(state.messages)
        context_token_estimate = sum(len(json.dumps(m)) for m in pruned) // 4
        
        new_step_count = state.step_count + 1
        
        model_tier = ModelTier.FAST if new_step_count > 1 else ModelTier.FULL
        step_cost = 0.005 if model_tier == ModelTier.FAST else 0.02
        estimated_cost = state.total_cost_usd + step_cost
        
        user_intent = self._extract_user_intent(pruned)
        
        has_tool_result = False
        if pruned and pruned[-1].get("role") == "tool":
            has_tool_result = True
        
        if has_tool_result:
            response_msg = {
                "role": "assistant",
                "content": "Task completed. Here is the result."
            }
        else:
            has_profile, has_sub = self._detect_intent_from_message(user_intent)
            if has_profile and has_sub:
                intent_hint = "get_profile and update_sub"
            elif has_profile:
                intent_hint = "get_profile"
            elif has_sub:
                intent_hint = "update_sub"
            else:
                intent_hint = "end"
            response_msg = {
                "role": "assistant",
                "content": f"Executing: {intent_hint}"
            }
        
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
        
        for msg in reversed(state.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                has_profile, has_sub = self._detect_intent_from_message(content)
                
                if has_profile and has_sub:
                    return "parallel"
                elif has_profile:
                    return "get_profile"
                elif has_sub:
                    return "update_sub"
                else:
                    return "end"
        
        return "end"

    async def run_get_user_profile(self, state: AgentState) -> Dict[str, Any]:
        if not state.user_id:
            return {"errors": state.errors + [{"error": "Missing user_id", "code": "INVALID_STATE"}]}
            
        try:
            inp = GetUserProfileInput(user_id=state.user_id)
            res = await get_user_profile(inp)
            if res.get("status") == "error":
                return {"errors": state.errors + [res]}
                
            return {"messages": state.messages + [{"role": "tool", "content": json.dumps(res)}]}
        except Exception as e:
            return {"errors": state.errors + [{"error": "execution_failure", "detail": str(e)}]}

    async def run_update_subscription(self, state: AgentState) -> Dict[str, Any]:
        if not state.user_id:
            return {"errors": state.errors + [{"error": "Missing user_id", "code": "INVALID_STATE"}]}

        try:
            last_user_msg = next((m.get("content", "") for m in reversed(state.messages) if m.get("role") == "user"), "")
            target_plan = self._extract_requested_plan(last_user_msg)
            
            if target_plan is None:
                return {
                    "errors": state.errors + [{
                        "error": "no_plan_specified",
                        "code": "AMBIGUOUS_REQUEST",
                        "error_type": "MISSING_PLAN",
                        "message": "No target plan specified in the request. Please specify 'free', 'pro', or 'enterprise'.",
                        "retryable": False,
                        "suggestion": "Ask the user which plan they want to switch to."
                    }]
                }
            
            inp = UpdateSubscriptionInput(user_id=state.user_id, plan_name=target_plan)
            res = await update_subscription(inp)
            
            if res.get("status") == "error":
                return {"errors": state.errors + [res]}
                
            cache_invalidate(state.user_id)
            
            return {
                "messages": state.messages + [{"role": "tool", "content": json.dumps(res)}],
                "current_plan": target_plan
            }
        except Exception as e:
            return {"errors": state.errors + [{"error": "execution_failure", "detail": str(e)}]}

    async def run_parallel_tools(self, state: AgentState) -> Dict[str, Any]:
        if not state.user_id:
            return {"errors": state.errors + [{"error": "Missing user_id", "code": "INVALID_STATE"}]}

        last_user_msg = next((m.get("content", "") for m in reversed(state.messages) if m.get("role") == "user"), "")
        target_plan = self._extract_requested_plan(last_user_msg)

        profile_inp = GetUserProfileInput(user_id=state.user_id)
        sub_inp = UpdateSubscriptionInput(user_id=state.user_id, plan_name=target_plan or PlanName.PRO)

        try:
            results = await asyncio.gather(
                get_user_profile(profile_inp),
                update_subscription(sub_inp),
                return_exceptions=True
            )
            
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
                        new_plan = target_plan or PlanName.PRO
                        cache_invalidate(state.user_id)
            
            result = {"messages": state.messages + new_messages}
            if new_errors:
                result["errors"] = state.errors + new_errors
            if new_plan != state.current_plan:
                result["current_plan"] = new_plan
            return result
            
        except Exception as e:
            return {"errors": state.errors + [{"error": "parallel_dispatch_failure", "detail": str(e)}]}