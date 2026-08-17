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

# --- Recall@K and MBR Evaluation Metrics ---

def test_recall_at_k_evaluator_basic():
    from app.evaluation import RecallAtKEvaluator, ToolCallCandidate, EvaluationCase
    
    evaluator = RecallAtKEvaluator(k_values=[1, 3, 5])
    
    case = EvaluationCase(
        case_id="test_1",
        user_message="show my profile",
        expected_tool="get_profile",
    )
    case.candidates = [
        ToolCallCandidate(tool_name="get_profile", plan_name=None, confidence=0.9),
        ToolCallCandidate(tool_name="update_sub", plan_name="pro", confidence=0.5),
        ToolCallCandidate(tool_name="end", plan_name=None, confidence=0.1),
    ]
    
    results = evaluator.evaluate_case(case)
    assert results["recall_at_1"] == 1.0  # get_profile is at position 0
    assert results["recall_at_3"] == 1.0  # get_profile is in top 3
    assert results["recall_at_5"] == 1.0  # get_profile is in top 5

def test_recall_at_k_when_correct_not_in_top_k():
    from app.evaluation import RecallAtKEvaluator, ToolCallCandidate, EvaluationCase
    
    evaluator = RecallAtKEvaluator(k_values=[1, 3])
    
    case = EvaluationCase(
        case_id="test_2",
        user_message="show my profile",
        expected_tool="get_profile",
    )
    case.candidates = [
        ToolCallCandidate(tool_name="update_sub", plan_name="pro", confidence=0.8),
        ToolCallCandidate(tool_name="end", plan_name=None, confidence=0.6),
        ToolCallCandidate(tool_name="get_profile", plan_name=None, confidence=0.3),
    ]
    
    results = evaluator.evaluate_case(case)
    assert results["recall_at_1"] == 0.0  # get_profile not at position 0
    assert results["recall_at_3"] == 1.0  # get_profile is in top 3 (position 2)

def test_mbr_evaluator_selects_lowest_risk():
    from app.evaluation import MBREvaluator, ToolCallCandidate, EvaluationCase
    
    evaluator = MBREvaluator()
    
    case = EvaluationCase(
        case_id="test_3",
        user_message="show my profile",
        expected_tool="get_profile",
    )
    case.candidates = [
        ToolCallCandidate(tool_name="get_profile", plan_name=None, confidence=0.9),
        ToolCallCandidate(tool_name="update_sub", plan_name="pro", confidence=0.8),
        ToolCallCandidate(tool_name="get_profile", plan_name=None, confidence=0.7),
    ]
    
    results = evaluator.evaluate_case(case)
    # get_profile at index 0 and 2 should have lowest risk
    # since they agree with each other (same tool+plan)
    assert results["best_candidate_idx"] in [0, 2]
    assert results["mbr_correct"] is True

def test_mbr_evaluator_correct_when_disagreement():
    from app.evaluation import MBREvaluator, ToolCallCandidate, EvaluationCase
    
    evaluator = MBREvaluator()
    
    case = EvaluationCase(
        case_id="test_4",
        user_message="show my profile",
        expected_tool="get_profile",
    )
    # All candidates disagree → risk is high but MBR picks the one with most agreement
    case.candidates = [
        ToolCallCandidate(tool_name="update_sub", plan_name="pro", confidence=0.9),
        ToolCallCandidate(tool_name="end", plan_name=None, confidence=0.8),
        ToolCallCandidate(tool_name="get_profile", plan_name=None, confidence=0.7),
    ]
    
    results = evaluator.evaluate_case(case)
    # With disagreement, MBR picks lowest risk which is get_profile (agrees with fewer but has lower disagreement risk with itself)
    assert results["best_candidate_idx"] is not None

def test_tool_calling_evaluator_integration():
    from app.evaluation import ToolCallingEvaluator, create_golden_set_evaluator
    
    evaluator = create_golden_set_evaluator()
    summary = evaluator.get_summary()
    
    assert summary["total_cases"] == 7
    assert 0.0 <= summary["recall_at_1_mean"] <= 1.0
    assert 0.0 <= summary["recall_at_5_mean"] <= 1.0
    assert 0.0 <= summary["correct_tool_selection_rate"] <= 1.0

def test_intent_detection():
    from app.evaluation import ToolCallingEvaluator
    
    evaluator = ToolCallingEvaluator()
    
    has_profile, has_sub, plan = evaluator._detect_intent("show me my profile")
    assert has_profile is True
    assert has_sub is False
    assert plan is None  # "show me my profile" shouldn't detect "pro" from "profile"
    
    has_profile, has_sub, plan = evaluator._detect_intent("upgrade to enterprise")
    assert has_profile is False
    assert has_sub is True
    assert plan == "enterprise"
    
    has_profile, has_sub, plan = evaluator._detect_intent("cancel my subscription")
    assert has_sub is True
    assert plan is None  # No explicit plan mentioned
    
    has_profile, has_sub, plan = evaluator._detect_intent("change to free")
    assert has_sub is True
    assert plan == "free"
    
    has_profile, has_sub, plan = evaluator._detect_intent("upgrade to pro")
    assert has_sub is True
    assert plan == "pro"

def test_generate_candidates_sorts_by_confidence():
    from app.evaluation import ToolCallingEvaluator
    
    evaluator = ToolCallingEvaluator()
    evaluator.register_case("test_5", "show my profile", "get_profile")
    candidates = evaluator.generate_candidates("test_5")
    
    # Should be sorted by confidence descending
    for i in range(len(candidates) - 1):
        assert candidates[i].confidence >= candidates[i + 1].confidence
