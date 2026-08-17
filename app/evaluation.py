"""
Evaluation module implementing Recall@K and MBR metrics for tool-calling assessment.

These metrics evaluate how well the LangGraph agent ranks and selects the correct
tools for user intents. They complement the existing judge model calibration
(True Positive Rate, Cohen's κ) with retrieval-based and risk-minimization approaches.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

TOOL_KEYWORDS = {
    "get_profile": ["get_profile", "profile", "user_info", "view plan", "my profile", "account info"],
    "update_sub": ["update_sub", "upgrade", "change plan", "change to", "subscribe", "subscription", "downgrade"],
}

class ToolCallOutcome(Enum):
    CORRECT = "correct"
    WRONG_TOOL = "wrong_tool"
    INVALID_PLAN = "invalid_plan"
    MISSING_INTENT = "missing_intent"
    TERMINATED_EARLY = "terminated_early"

@dataclass
class ToolCallCandidate:
    """Represents a candidate tool call with confidence score."""
    tool_name: str
    plan_name: Optional[str]
    confidence: float
    reasoning: str = ""

@dataclass
class EvaluationCase:
    """A single test case for evaluation."""
    case_id: str
    user_message: str
    expected_tool: str
    expected_plan: Optional[str] = None
    description: str = ""
    candidates: List[ToolCallCandidate] = field(default_factory=list)
    ground_truth_outcome: Optional[ToolCallOutcome] = None

@dataclass
class EvaluationResult:
    """Result of evaluating a single case."""
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mbr_risk: float = 0.0
    best_candidate_idx: int = 0
    outcome: ToolCallOutcome = ToolCallOutcome.TERMINATED_EARLY

class RecallAtKEvaluator:
    """
    Recall@K: Measures whether the correct tool is among the top-K
    ranked tool candidates.
    
    Recall@K = 1 if correct tool is in top-K candidates, else 0
    """
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5]
    
    def evaluate_case(self, case: EvaluationCase) -> Dict[str, float]:
        """Evaluate a single case for Recall@K."""
        results = {}
        
        for k in self.k_values:
            top_k = case.candidates[:k]
            found = any(c.tool_name == case.expected_tool for c in top_k)
            results[f"recall_at_{k}"] = 1.0 if found else 0.0
        
        return results

class MBREvaluator:
    """
    Minimum Bayes-Risk (MBR) decoding: Selects the tool call sequence
    that minimizes the expected risk (loss) across candidate sequences.
    
    Risk is computed as the disagreement between candidate sequences
    weighted by their confidence scores.
    """
    
    def __init__(self, risk_weight: float = 1.0):
        self.risk_weight = risk_weight
    
    def compute_pairwise_risk(self, candidates: List[ToolCallCandidate], 
                              expected_tool: str) -> List[float]:
        """
        Compute pairwise risk for each candidate.
        
        Risk(c_i) = sum over c_j of: loss(c_i, c_j) * confidence(c_j)
        where loss = 0 if same tool, 1 if different.
        """
        risks = []
        
        for i, ci in enumerate(candidates):
            risk = 0.0
            for j, cj in enumerate(candidates):
                # Loss function: 0 if same tool+plan, 1 otherwise
                if ci.tool_name != cj.tool_name or ci.plan_name != cj.plan_name:
                    loss = 1.0
                else:
                    loss = 0.0
                
                # Weight by candidate confidence
                risk += loss * cj.confidence
            
            risks.append(risk)
        
        return risks
    
    def evaluate_case(self, case: EvaluationCase) -> Dict[str, Any]:
        """Evaluate a single case using MBR decoding."""
        if not case.candidates:
            return {
                "mbr_risk": float('inf'),
                "best_candidate_idx": -1,
                "mbr_decision": None,
                "mbr_correct": False
            }
        
        risks = self.compute_pairwise_risk(case.candidates, case.expected_tool)
        min_risk_idx = risks.index(min(risks))
        best_candidate = case.candidates[min_risk_idx]
        
        is_correct = best_candidate.tool_name == case.expected_tool
        if case.expected_plan:
            is_correct = is_correct and best_candidate.plan_name == case.expected_plan
        
        return {
            "mbr_risk": risks[min_risk_idx],
            "best_candidate_idx": min_risk_idx,
            "mbr_decision": best_candidate.tool_name,
            "mbr_correct": is_correct,
            "all_risks": risks
        }

class ToolCallingEvaluator:
    """
    Comprehensive evaluator combining Recall@K and MBR metrics.
    
    Usage:
        evaluator = ToolCallingEvaluator()
        evaluator.register_case("case_1", "show my profile", "get_profile")
        evaluator.generate_candidates("case_1")
        results = evaluator.evaluate_all()
    """
    
    def __init__(self, k_values: List[int] = None):
        self.cases: Dict[str, EvaluationCase] = {}
        self.recall_evaluator = RecallAtKEvaluator(k_values)
        self.mbr_evaluator = MBREvaluator()
        self.results: Dict[str, EvaluationResult] = {}
    
    def register_case(self, case_id: str, user_message: str, 
                      expected_tool: str, expected_plan: Optional[str] = None,
                      description: str = "") -> EvaluationCase:
        """Register a new evaluation case."""
        case = EvaluationCase(
            case_id=case_id,
            user_message=user_message,
            expected_tool=expected_tool,
            expected_plan=expected_plan,
            description=description
        )
        self.cases[case_id] = case
        return case
    
    def _detect_intent(self, message: str) -> Tuple[bool, bool, Optional[str]]:
        """
        Detect tool intent from message content.
        
        Returns: (has_profile_intent, has_sub_intent, extracted_plan)
        """
        content_lower = message.lower()
        
        has_profile = any(k in content_lower for k in TOOL_KEYWORDS["get_profile"])
        has_sub = any(k in content_lower for k in TOOL_KEYWORDS["update_sub"])
        
        plan = None
        if "enterprise" in content_lower:
            plan = "enterprise"
        elif "downgrade to free" in content_lower or content_lower == "free" or " to free" in content_lower:
            plan = "free"
        elif " to pro" in content_lower or "upgrade to pro" in content_lower or "change plan to pro" in content_lower:
            plan = "pro"
        
        return has_profile, has_sub, plan
    
    def generate_candidates(self, case_id: str, num_candidates: int = 5) -> List[ToolCallCandidate]:
        """
        Generate candidate tool calls with varying confidence scores.
        
        In a real system, these would come from model sampling or
        beam search. Here we simulate diverse candidates.
        """
        case = self.cases.get(case_id)
        if not case:
            raise KeyError(f"Case {case_id} not registered")
        
        candidates = []
        has_profile, has_sub, plan = self._detect_intent(case.user_message)
        
        # Candidate 1: Full intent match (high confidence)
        if has_profile:
            candidates.append(ToolCallCandidate(
                tool_name="get_profile",
                plan_name=None,
                confidence=0.95,
                reasoning="User explicitly requested profile lookup"
            ))
        
        if has_sub:
            candidates.append(ToolCallCandidate(
                tool_name="update_sub",
                plan_name=plan or "pro",
                confidence=0.85,
                reasoning=f"Subscription change requested, plan={plan}"
            ))
        
        # Candidate 2: Partial match (medium confidence)
        if has_profile and not has_sub:
            candidates.append(ToolCallCandidate(
                tool_name="get_profile",
                plan_name=None,
                confidence=0.70,
                reasoning="Profile-related keywords detected"
            ))
        
        # Candidate 3: Wrong tool (low confidence)
        wrong_tool = "update_sub" if case.expected_tool == "get_profile" else "get_profile"
        candidates.append(ToolCallCandidate(
            tool_name=wrong_tool,
            plan_name=plan or "pro",
            confidence=0.10,
            reasoning="Low confidence alternative"
        ))
        
        # Candidate 4: Missing intent (very low confidence)
        candidates.append(ToolCallCandidate(
            tool_name="end",
            plan_name=None,
            confidence=0.05,
            reasoning="Unclear intent, termination suggested"
        ))
        
        # Candidate 5: Another wrong tool with higher confidence (tests robustness)
        candidates.append(ToolCallCandidate(
            tool_name=wrong_tool,
            plan_name="free" if plan != "free" else "enterprise",
            confidence=0.20,
            reasoning="Alternative interpretation"
        ))
        
        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        case.candidates = candidates
        return candidates
    
    def evaluate_case(self, case_id: str) -> EvaluationResult:
        """Evaluate a single case with both metrics."""
        case = self.cases.get(case_id)
        if not case:
            raise KeyError(f"Case {case_id} not registered")
        
        if not case.candidates:
            return EvaluationResult(outcome=ToolCallOutcome.TERMINATED_EARLY)
        
        # Recall@K evaluation
        recall_results = self.recall_evaluator.evaluate_case(case)
        recall_at_1 = recall_results["recall_at_1"]
        recall_at_5 = recall_results["recall_at_5"]
        
        # MBR evaluation
        mbr_results = self.mbr_evaluator.evaluate_case(case)
        
        # Determine outcome
        if recall_at_1 > 0:
            outcome = ToolCallOutcome.CORRECT
        elif any(c.tool_name != case.expected_tool for c in case.candidates[:5]):
            wrong_in_top5 = any(
                c.tool_name != case.expected_tool 
                for c in case.candidates[:5]
            )
            if wrong_in_top5 and recall_at_5 > 0:
                outcome = ToolCallOutcome.WRONG_TOOL
            else:
                outcome = ToolCallOutcome.MISSING_INTENT
        
        result = EvaluationResult(
            recall_at_1=recall_at_1,
            recall_at_5=recall_at_5,
            mbr_risk=mbr_results["mbr_risk"],
            best_candidate_idx=mbr_results["best_candidate_idx"],
            outcome=outcome
        )
        
        self.results[case_id] = result
        return result
    
    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate all registered cases and return summary metrics."""
        all_results = {}
        
        for case_id in self.cases:
            result = self.evaluate_case(case_id)
            all_results[case_id] = {
                "recall_at_1": result.recall_at_1,
                "recall_at_5": result.recall_at_5,
                "mbr_risk": result.mbr_risk,
                "mbr_best_idx": result.best_candidate_idx,
                "outcome": result.outcome.value,
                "expected_tool": self.cases[case_id].expected_tool,
            }
        
        return all_results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate summary statistics."""
        results = self.evaluate_all()
        
        if not results:
            return {"total_cases": 0}
        
        recall_at_1_values = [r["recall_at_1"] for r in results.values()]
        recall_at_5_values = [r["recall_at_5"] for r in results.values()]
        
        correct_count = sum(1 for r in results.values() if r["outcome"] == ToolCallOutcome.CORRECT.value)
        total_count = len(results)
        
        return {
            "total_cases": total_count,
            "recall_at_1_mean": sum(recall_at_1_values) / len(recall_at_1_values),
            "recall_at_5_mean": sum(recall_at_5_values) / len(recall_at_5_values),
            "correct_tool_selection_rate": correct_count / total_count,
            "cases": results
        }

def create_golden_set_evaluator() -> ToolCallingEvaluator:
    """
    Create an evaluator pre-populated with the 7 golden test cases
    from the project's evaluation plan.
    """
    evaluator = ToolCallingEvaluator()
    
    # Case 1: Happy path - profile read
    evaluator.register_case(
        "case_1", 
        "show me my profile",
        "get_profile",
        description="Happy path — profile read"
    )
    
    # Case 2: Happy path - plan upgrade
    evaluator.register_case(
        "case_2",
        "upgrade my plan to enterprise",
        "update_sub",
        expected_plan="enterprise",
        description="Happy path — plan upgrade"
    )
    
    # Case 3: Missing user (404 scenario - invalid user_id in intent)
    evaluator.register_case(
        "case_3",
        "show me profile user_999",
        "get_profile",
        description="Missing user (404)"
    )
    
    # Case 4: Invalid plan name
    evaluator.register_case(
        "case_4",
        "change my plan to premium",
        "update_sub",
        expected_plan=None,  # "premium" is invalid
        description="Invalid plan ('premium')"
    )
    
    # Case 5: Network error during mutation
    evaluator.register_case(
        "case_5",
        "upgrade to pro subscription",
        "update_sub",
        expected_plan="pro",
        description="Plan upgrade for idempotency test"
    )
    
    # Case 6: Ambiguous status
    evaluator.register_case(
        "case_6",
        "view plan and subscription status",
        "get_profile",
        description="Check current subscription status"
    )
    
    # Case 7: Downstream outage
    evaluator.register_case(
        "case_7",
        "get my profile and upgrade to enterprise",
        "get_profile",
        description="Both profile and subscription update (parallel)"
    )
    
    # Generate candidates for all cases
    for case_id in list(evaluator.cases.keys()):
        evaluator.generate_candidates(case_id)
    
    return evaluator

if __name__ == "__main__":
    import sys
    
    evaluator = create_golden_set_evaluator()
    summary = evaluator.get_summary()
    
    print(json.dumps(summary, indent=2, default=str))
    sys.exit(0)