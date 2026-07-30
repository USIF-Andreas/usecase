1 Restate the brief and locate the LLM in the system
warm_up
15m
In two or three sentences, restate what this subscription-management assistant is for, where the LLM actually sits in the FastAPI request-response lifecycle, and what 'success' means for a resolved user request. Identify the specific API endpoints in the skeleton that trigger the LangGraph agent, and define the success criteria Tarek should use to evaluate if the tool-calling layer is robust.

architecture


2 Identify the primary tool-calling failure modes
warm_up
20m
Review the provided repository skeleton, specifically the tool definitions in `app/tools/subscription.py` and `app/tools/user.py`. List the three planted flaws that prevent this agent from being production-ready. For each flaw, point to the specific file and line ranges where it surfaces, and describe the immediate symptom a user or developer would observe in production.

diagnosis



3 Redesign the get_user_profile tool schema and error handling
core
45m
Rewrite the `get_user_profile` tool interface in `app/tools/user.py` to fix the blind retry storm. Define a strict Pydantic model for the input arguments. Modify the try-except block so that instead of returning a generic 'Error occurred' string, it catches exceptions (like a 404 for a missing user) and returns a structured, machine-readable JSON error response that explicitly tells the LLM what went wrong and how to correct its input.

memory_tools_workflow


4 Harden the update_subscription schema with Enums
core
30m
The current prototype allows the LLM to pass any string to `plan_name` in `update_subscription`. Redesign this tool's input schema using a strict Python Enum of valid plans (e.g., 'free', 'pro', 'enterprise'). Provide the Pydantic field descriptions that explain these options to the model, and explain how this schema-level constraint prevents the LLM from hallucinating invalid plan updates.

memory_tools_workflow


5 Design transaction safety and idempotency keys
core
40m
Address the `flaw_non_idempotent_mutation` in `update_subscription`. Redesign the tool's signature to accept a mandatory or optional idempotency key (such as a UUID). Explain how the backend should track and verify this key in a stateful mock store to prevent duplicate subscription charges or state corruption if the network drops and the agent retries the call.

memory_tools_workflow


6 Specify the tool-calling evaluation metrics
advanced
45m
Design an evaluation plan to measure the tool-calling success rate of the rewritten service. Define a 'golden set' of at least 5 test cases representing both happy paths and edge cases (e.g., invalid user ID, malformed plan name). Specify what metrics you will track to verify that the agent achieves the target 80% test coverage and handles errors gracefully without crashing.

eval_plan



7 Calibrate a judge model for tool validation
advanced
40m
You want to use an LLM-as-judge to evaluate whether the agent's generated responses are grounded and match the mock API's actual state. Describe how you would calibrate this judge against human-labeled traces. How do you measure agreement (e.g., True Positive Rate), and how do you prevent the judge model from hallucinating its own validation criteria?

eval_plan


8 Analyze cost and latency of the tool-calling loop
advanced
45m
A single user interaction that triggers multiple tool calls can quickly blow past latency and cost budgets. Estimate the cost-per-task and p95 latency for a scenario where the agent encounters a validation error, self-corrects, and successfully updates a subscription. Propose two specific optimization levers (e.g., semantic caching, model routing) to keep the p95 latency under 2 seconds.

ops


9 Decide single-agent versus multi-agent pattern
novel
45m
A colleague suggests splitting this assistant into two separate agents: a 'User Profile Specialist' and a 'Subscription Billing Specialist' coordinated by an orchestrator. Evaluate this proposal. Compare the token overhead, latency, and coordination complexity of this multi-agent design against your single-agent LangGraph implementation. Make an explicit, architecture-grounded recommendation.

architecture