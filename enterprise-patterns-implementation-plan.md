# Enterprise Patterns — Implementation Plan

## Current State Assessment

After auditing the live codebase (`app/schemas.py`, `app/tools/user.py`, `app/tools/subscription.py`, `app/agent.py`, `app/main.py`, `app/config.py`), here is the gap analysis:

| # | Pattern | Status in Code | Evidence |
|---|---|---|---|
| 1 | **Tool Memoization (Caching)** | ❌ Not implemented | `get_user_profile` makes a fresh HTTP call every invocation; no `_PROFILE_CACHE`, no TTL logic |
| 2 | **Circuit Breaker** | ❌ Not implemented | No `FAILURE_COUNT`, no threshold, no cooldown gating anywhere in the tool layer |
| 3 | **Retry Policy (Tenacity)** | ❌ Not implemented | No `@retry` decorator, no exponential backoff; single-attempt `try/except` only |
| 4 | **Async HTTP Client** | ❌ Not implemented | Tools use `httpx.get()` / `httpx.post()` (sync); no shared `httpx.AsyncClient` connection pool |
| 5 | **Adaptive Context Pruning** | ❌ Not implemented | `AgentState` grows unbounded; no `_prune_context()` truncation in `call_model()` |
| 6 | **Observability / Tracing** | ❌ Not implemented | No latency tracking, no per-node metrics, no telemetry output in any node |
| 7 | **Parallel Tool Scheduler** | ❌ Not implemented | Graph runs tools sequentially (`agent -> user -> agent -> sub -> agent`); no `asyncio.gather` parallelism |
| 8 | **Async Checkpointer** | ❌ Not implemented | Using `MemorySaver` (process-local, lost on restart); no `AsyncPostgresSaver` / `AsyncRedisSaver` |
| 9 | **SSE Streaming** | ❌ Not implemented | No `/chat/stream` endpoint; only blocking `POST /chat` |
| 10 | **Model Tiering** | ❌ Not implemented | Single model path; no distinction between reasoning vs formatting turns |

**Bottom line:** Zero of the 6 enterprise patterns (or their sub-patterns) are present in the current codebase. The "Pattern Summary Matrix" was aspirational design, not applied code.

---

## Implementation Plan

### Phase 1 — Network & Resilience Layer (Tools)

**Target files:** `app/tools/user.py`, `app/tools/subscription.py`, `app/config.py`

#### 1.1 Shared Async HTTP Client

Create a module-level `httpx.AsyncClient` with connection pooling and timeouts:

```python
# app/tools/_client.py
import httpx
from app.config import settings

_client: httpx.AsyncClient | None = None

async def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.user_api_base_url,
            timeout=httpx.Timeout(10.0, connect=3.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
        )
    return _client
```

**Rationale:** Each `httpx.get()` call creates a new TCP connection (3-way handshake + TLS). With a shared pool, repeated tool calls within a session reuse warm connections, cutting p95 latency by 40-60%.

#### 1.2 Retry Policy with Tenacity

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    before_sleep=lambda retry_state: logger.warning(
        "Retry attempt %d after %s", retry_state.attempt_number, retry_state.outcome.exception()
    ),
)
async def _fetch_profile(url: str) -> dict: ...
```

**Exponential backoff:** 1s → 2s → 4s (jitter optional). Three attempts cover the vast majority of transient failures without unacceptable user-facing latency.

#### 1.3 Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, threshold: int = 5, cooldown: float = 30.0):
        self.failure_count = 0
        self.threshold = threshold
        self.cooldown = cooldown
        self.last_failure = 0.0
        self._lock = threading.Lock()

    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with self._lock:
                if self.failure_count >= self.threshold:
                    if time.time() - self.last_failure < self.cooldown:
                        return {"status": "error", "code": "CIRCUIT_OPEN", ...}
                    self.failure_count = 0  # half-open
            result = await func(*args, **kwargs)
            with self._lock:
                if result.get("status") == "error" and result.get("code", 0) in (500, 502, 503):
                    self.failure_count += 1
                    self.last_failure = time.time()
                else:
                    self.failure_count = 0
            return result
        return wrapper
```

**Why:** Without a circuit breaker, a downstream outage causes cascading timeouts on every request, consuming threads/connections and degrading the entire service. A 30-second cooldown lets the downstream recover without hammering it.

#### 1.4 Tool Memoization (TTL Cache)

```python
from functools import lru_cache
import time

_PROFILE_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 30.0

async def get_user_profile(input_data: GetUserProfileInput) -> dict:
    uid = input_data.user_id
    now = time.time()
    
    cached = _PROFILE_CACHE.get(uid)
    if cached and (now - cached[0]) < CACHE_TTL:
        return {"status": "success", "data": cached[1], "cached": True}
    
    data = await _fetch_and_cache(uid)
    _PROFILE_CACHE[uid] = (now, data)
    return {"status": "success", "data": data, "cached": False}
```

**Why 30s?** User profiles change infrequently. A 30-second TTL eliminates redundant reads during the same conversation without risking stale data across sessions.

---

### Phase 2 — Adaptive State & Observability (Agent)

**Target files:** `app/agent.py`, `app/schemas.py`

#### 2.1 Context Pruning

```python
def _prune_context(self, messages: list[dict], max_turns: int = 6) -> list[dict]:
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
```

**Rationale:** LangGraph's state is passed to the LLM context window on every turn. If tool outputs are large JSON blobs, context grows unbounded → token cost increases → model loses focus on recent turns. Pruning keeps the window fixed at `max_turns + 1`.

Add to config:
```python
max_context_turns: int = 6
```

#### 2.2 Observability Telemetry

```python
import time, logging

def call_model(self, state: AgentState) -> Dict[str, Any]:
    start = time.perf_counter()
    
    pruned = self._prune_context(state.messages)
    context_token_estimate = sum(len(json.dumps(m)) for m in pruned) // 4  # rough token count
    
    # Simulate LLM call
    latency_ms = (time.perf_counter() - start) * 1000
    
    logger.info(
        "step=%d latency_ms=%.1f context_tokens=%d cost=%.4f",
        state.step_count, latency_ms, context_token_estimate, state.total_cost_usd
    )
    
    return {
        "messages": state.messages + [new_msg],
        "step_count": state.step_count + 1,
        "total_cost_usd": state.total_cost_usd + 0.02,
    }
```

**Why:** Without telemetry, production debugging is guesswork. Per-step latency identifies slow tools; token estimates flag context-window pressure before it causes truncation errors.

---

### Phase 3 — Parallel Tool Dispatch (Orchestration)

**Target files:** `app/agent.py`

#### 3.1 Parallel Tool Node

```python
import asyncio

async def run_parallel_tools(self, state: AgentState) -> Dict[str, Any]:
    tasks = []
    
    if "get_profile" in self._last_intent(state):
        inp = GetUserProfileInput(user_id=state.user_id)
        tasks.append(asyncio.to_thread(get_user_profile, inp))
    
    if "update_sub" in self._last_intent(state):
        inp = UpdateSubscriptionInput(user_id=state.user_id, plan_name=PlanName.PRO)
        tasks.append(asyncio.to_thread(update_subscription, inp))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    new_messages, new_errors = [], []
    for res in results:
        if isinstance(res, Exception):
            new_errors.append({"error": "parallel_execution", "detail": str(res)})
        elif res.get("status") == "error":
            new_errors.append(res)
        else:
            new_messages.append({"role": "tool", "content": str(res)})
    
    return {
        "messages": state.messages + new_messages,
        "errors": state.errors + new_errors,
    }
```

**Graph change:** Replace two sequential tool nodes with one `parallel_tools_node`:

```python
self.workflow.add_node("parallel_tools_node", self.run_parallel_tools)
self.workflow.add_conditional_edges("agent", self.router, {
    "tools": "parallel_tools_node",
    "end": END,
})
self.workflow.add_edge("parallel_tools_node", "agent")
```

**Why:** In the current sequential graph, fetching a profile and updating a subscription takes 2 round-trips through the agent node. Running them concurrently cuts wall-clock time in half when they're independent.

#### 3.2 Async Checkpointer Upgrade

```python
# Current (process-local, lost on restart):
self.checkpointer = MemorySaver()

# Production target:
from langgraph.checkpoint.postgres import AsyncPostgresSaver

async def init_checkpointer():
    conn_string = settings.checkpointer_dsn  # "postgresql://..."
    checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)
    await checkpointer.setup()  # creates tables if needed
    return checkpointer
```

**Why:** `MemorySaver` means every pod has its own isolated state. A horizontal scale-out (2+ pods) loses conversation continuity — a request from the same user hits a different pod with no memory. An external checkpointer is necessary for multi-pod deployment.

---

### Phase 4 — API Streaming & Model Tiering (Gateway)

**Target files:** `app/main.py`

#### 4.1 SSE Streaming Endpoint

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
async def chat_stream(payload: AgentState):
    async def event_generator():
        config = {"configurable": {"thread_id": payload.user_id or "default_session"}}
        state_dict = payload.model_dump()
        
        async for event in assistant.app.astream(state_dict, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                yield f"data: {json.dumps({'node': node_name, 'step': update.get('step_count')})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Why:** The current `/chat` blocks until the entire graph finishes (potentially 6+ sequential steps → 10-15s). SSE lets the frontend render partial results (e.g., "Fetching profile...", "Updating plan...") as they happen, improving perceived responsiveness dramatically.

#### 4.2 Model Tiering

```python
class ModelTier(str, Enum):
    FAST = "gpt-4o-mini"      # Tool formatting, summarization
    FULL = "gpt-4o"            # Reasoning, planning, user-facing generation

LLM_ROUTING = {
    "tool_format": ModelTier.FAST,
    "context_summary": ModelTier.FAST,
    "reasoning": ModelTier.FULL,
    "user_response": ModelTier.FULL,
}
```

**Why:** Using the full model for every turn wastes cost. Tool-formatting turns (e.g., "decide which tool to call") are trivial for a mini model. Splitting by intent reduces cost by ~60% without quality loss.

---

## Migration Timeline

| Phase | Patterns | Files Changed | Risk | Effort |
|---|---|---|---|---|
| **P1** | Async client, Retry, Circuit Breaker, Caching | `tools/*`, `config.py` | Low (tools are leaf nodes) | 2-3d |
| **P2** | Context pruning, Observability | `agent.py`, `schemas.py` | Low (read-only state transforms) | 1-2d |
| **P3** | Parallel dispatch, Async checkpointer | `agent.py`, `config.py`, new DB | Medium (changes graph topology) | 3-5d |
| **P4** | SSE streaming, Model tiering | `main.py` | Low (new endpoints, additive) | 1-2d |

**Total: ~7-12 engineering days for a senior engineer.**

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Parallel execution causes race conditions on `state.errors` | Low | Medium | Use `return_exceptions=True` and merge results immutably |
| Circuit breaker stays open due to flapping | Medium | Low | Implement half-open probe (single test request after cooldown) |
| TTL cache serves stale profile data | Low | Low (30s TTL) | Add `cache_invalidate(user_id)` method for explicit invalidation |
| AsyncPostgresSaver adds DB latency per step | Medium | Medium | Use connection pool; benchmark before/after; consider Redis as faster alternative |
| SSE stream drops mid-response | Medium | High | Add `retry: 3000` directive in SSE preamble; client-side reconnection logic |

---

## Decision Record

| Decision | Choice | Alternatives Considered | Rationale |
|---|---|---|---|
| Retry library | **Tenacity** | backoff, custom | Tenacity is battle-tested, LangChain ecosystem uses it, supports async natively |
| Circuit breaker granularity | **Per-tool instance** | Global, per-endpoint | Per-tool prevents one failing endpoint from starving another |
| Cache backend | **In-memory dict** (Phase 1), Redis (Phase 2) | Memcached, file-based | In-memory is zero-infra for MVP; Redis adds persistence & cross-pod sharing |
| Checkpointer | **AsyncPostgresSaver** | AsyncRedisSaver, MongoDB | LangGraph has first-class Postgres support; most teams already run Postgres |
| Streaming format | **SSE (text/event-stream)** | WebSocket, Chunked JSON | SSE is simpler, HTTP/2-compatible, works with standard fetch() EventSource |
| Context pruning strategy | **Drop oldest tool outputs** | Summarize via LLM, semantic chunking | LLM summarization adds latency & cost; dropping is O(1) and sufficient |
