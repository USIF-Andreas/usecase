import asyncio
import json
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
import httpx

from app.schemas import AgentState
from app.agent import LangGraphAssistant
from app.config import settings
from app.tools._client import close_http_client

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LangGraph Assistant Service")
    yield
    await close_http_client()
    logger.info("Shut down — HTTP client pool closed")

app = FastAPI(
    title="LangGraph Assistant Service",
    description="FastAPI backend with enterprise-grade LangGraph agent for subscription management.",
    lifespan=lifespan,
)
assistant = LangGraphAssistant()

@app.post("/chat", response_model=Dict[str, Any])
async def chat(payload: AgentState):
    try:
        loop = asyncio.get_running_loop()
        state_dict = payload.model_dump()
        config = {"configurable": {"thread_id": payload.user_id or "default_session"}}
        updated_state = await loop.run_in_executor(
            None, 
            lambda: assistant.app.invoke(state_dict, config=config)
        )
        return updated_state
    except Exception as e:
        logger.error("Execution error in /chat: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "AgentExecutionError", "message": str(e)}
        )

@app.post("/chat/stream")
async def chat_stream(payload: AgentState):
    """Server-Sent Events (SSE) streaming endpoint emitting real-time node updates."""
    async def event_generator():
        config = {"configurable": {"thread_id": payload.user_id or "default_session"}}
        state_dict = payload.model_dump()
        
        try:
            async for event in assistant.app.astream(state_dict, config=config, stream_mode="updates"):
                for node_name, update in event.items():
                    sse_data = {
                        "node": node_name,
                        "step": update.get("step_count"),
                        "has_errors": bool(update.get("errors")),
                    }
                    yield f"data: {json.dumps(sse_data)}\n\n"
        except Exception as e:
            logger.error("SSE stream error: %s", str(e), exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.user_api_base_url}/health")
            upstream_ok = resp.status_code == 200
    except Exception:
        upstream_ok = False
        
    return {
        "status": "healthy" if upstream_ok else "degraded",
        "mock_api_connected": upstream_ok,
        "config": {
            "max_steps": settings.max_steps,
            "cost_limit_usd": settings.cost_limit_usd,
            "model_family": settings.model_family,
        }
    }
