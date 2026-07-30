import asyncio
import pytest
from pydantic import ValidationError

from app.schemas import AgentState, PlanName, ModelTier
from app.tools.user import get_user_profile, GetUserProfileInput, cache_invalidate, cache_clear, _PROFILE_CACHE
from app.tools.subscription import update_subscription, UpdateSubscriptionInput
from app.tools._circuit_breaker import CircuitBreaker
from app.agent import LangGraphAssistant

# --- Schema Validation Tests ---

def test_get_user_profile_input_validation():
    inp = GetUserProfileInput(user_id="user_123")
    assert inp.user_id == "user_123"

def test_update_subscription_enum_validation():
    inp = UpdateSubscriptionInput(user_id="user_123", plan_name=PlanName.PRO)
    assert inp.plan_name == PlanName.PRO

    inp2 = UpdateSubscriptionInput(user_id="user_123", plan_name="enterprise")
    assert inp2.plan_name == PlanName.ENTERPRISE

    with pytest.raises(ValidationError):
        UpdateSubscriptionInput(user_id="user_123", plan_name="invalid_plan_name")

def test_update_subscription_idempotency_key_generated():
    inp = UpdateSubscriptionInput(user_id="user_123", plan_name=PlanName.FREE)
    assert inp.idempotency_key is not None
    assert len(inp.idempotency_key) > 0

def test_update_subscription_idempotency_keys_unique():
    inp1 = UpdateSubscriptionInput(user_id="user_123", plan_name=PlanName.FREE)
    inp2 = UpdateSubscriptionInput(user_id="user_123", plan_name=PlanName.FREE)
    assert inp1.idempotency_key != inp2.idempotency_key

# --- Agent Tests ---

def test_agent_graph_initialization():
    assistant = LangGraphAssistant()
    assert assistant.app is not None

def test_agent_state_schema():
    state = AgentState(messages=[{"role": "user", "content": "Hello"}], user_id="user_123")
    assert state.user_id == "user_123"
    assert state.step_count == 0
    assert state.total_cost_usd == 0.0
    assert state.telemetry == []

def test_agent_state_telemetry_field():
    state = AgentState(
        messages=[],
        telemetry=[{"step": 1, "latency_ms": 5.0, "context_tokens": 100}]
    )
    assert len(state.telemetry) == 1
    assert state.telemetry[0]["step"] == 1

# --- Model Tier Tests ---

def test_model_tier_enum():
    assert ModelTier.FAST.value == "gpt-4o-mini"
    assert ModelTier.FULL.value == "gpt-4o"

# --- Circuit Breaker Tests ---

def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker(threshold=3, cooldown=10.0)
    assert not cb.is_open

def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(threshold=3, cooldown=10.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open

def test_circuit_breaker_resets_on_success():
    cb = CircuitBreaker(threshold=3, cooldown=10.0)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert not cb.is_open
    assert cb.failure_count == 0

def test_circuit_breaker_reset():
    cb = CircuitBreaker(threshold=3, cooldown=10.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open
    cb.reset()
    assert not cb.is_open

# --- Cache Tests ---

def test_cache_invalidation():
    _PROFILE_CACHE["test_user"] = (0.0, {"name": "Test"})
    cache_invalidate("test_user")
    assert "test_user" not in _PROFILE_CACHE

def test_cache_clear():
    _PROFILE_CACHE["u1"] = (0.0, {})
    _PROFILE_CACHE["u2"] = (0.0, {})
    cache_clear()
    assert len(_PROFILE_CACHE) == 0

# --- Context Pruning Tests ---

def test_context_pruning_short_messages():
    assistant = LangGraphAssistant()
    msgs = [{"role": "user", "content": "hi"}]
    result = assistant._prune_context(msgs, max_turns=6)
    assert result == msgs

def test_context_pruning_truncates_old_tool_outputs():
    assistant = LangGraphAssistant()
    msgs = [
        {"role": "system", "content": "You are an assistant"},
        {"role": "tool", "content": '{"big": "data"}'},
        {"role": "tool", "content": '{"more": "data"}'},
        {"role": "user", "content": "question 1"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "tool", "content": '{"result": "data"}'},
        {"role": "user", "content": "question 2"},
        {"role": "assistant", "content": "answer 2"},
    ]
    result = assistant._prune_context(msgs, max_turns=4)
    assert result[0] == msgs[0]
    truncated = [m for m in result if m.get("content") == "[truncated]"]
    assert len(truncated) > 0
    assert result[-1] == msgs[-1]
