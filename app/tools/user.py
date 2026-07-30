import time
import logging
from urllib.parse import quote
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

from app.config import settings
from app.tools._client import get_http_client
from app.tools._circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# --- TTL Cache ---
_PROFILE_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL = settings.cache_ttl

def cache_invalidate(user_id: str) -> None:
    """Explicitly invalidate a cached user profile."""
    _PROFILE_CACHE.pop(user_id, None)

def cache_clear() -> None:
    """Clear the entire profile cache."""
    _PROFILE_CACHE.clear()

class GetUserProfileInput(BaseModel):
    user_id: str = Field(
        ...,
        description="The unique identifier of the user (e.g., 'user_123'). Must be a non-empty string."
    )

_user_profile_cb = CircuitBreaker(
    threshold=settings.circuit_breaker_threshold, 
    cooldown=settings.circuit_breaker_cooldown
)

@retry(
    stop=stop_after_attempt(settings.retry_max_attempts),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    before_sleep=lambda rs: logger.warning(
        "Retry attempt %d for _fetch_profile after: %s",
        rs.attempt_number, rs.outcome.exception()
    ),
)
async def _fetch_profile(url: str) -> httpx.Response:
    client = await get_http_client()
    response = await client.get(url)
    response.raise_for_status()
    return response

@_user_profile_cb
async def get_user_profile(input_data: GetUserProfileInput) -> dict:
    """
    Retrieves user profile with async connection pooling, TTL caching, retries, and circuit breaker.
    """
    uid = input_data.user_id
    now = time.time()
    
    cached = _PROFILE_CACHE.get(uid)
    if cached and (now - cached[0]) < CACHE_TTL:
        logger.debug("Cache HIT for user_id=%s", uid)
        return {"status": "success", "data": cached[1], "cached": True}
    
    safe_user_id = quote(uid)
    url = f"/v1/users/{safe_user_id}"
    
    try:
        response = await _fetch_profile(url)
        data = response.json()
        _PROFILE_CACHE[uid] = (now, data)
        return {"status": "success", "data": data, "cached": False}
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        if status_code == 404:
            return {
                "status": "error",
                "code": 404,
                "error_type": "NOT_FOUND",
                "message": f"User with ID '{uid}' was not found in the system.",
                "retryable": False,
                "suggestion": "Prompt the user to verify their user_id or check for typos. Do not retry the request with the same user_id."
            }
        return {
            "status": "error",
            "code": status_code,
            "error_type": "HTTP_ERROR",
            "message": f"HTTP error {status_code}: {e.response.text}",
            "retryable": status_code >= 500,
            "suggestion": "If 5xx, wait and retry once; otherwise check input parameters."
        }
    except (httpx.RequestError, httpx.TimeoutException) as e:
        return {
            "status": "error",
            "code": "NETWORK_ERROR",
            "error_type": "CONNECTION_FAILED",
            "message": f"Failed to connect to user profile service: {str(e)}",
            "retryable": True,
            "suggestion": "Transient network issue. Retry after a short delay."
        }
