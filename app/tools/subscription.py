import uuid
import logging
from urllib.parse import quote
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

from app.config import settings
from app.schemas import PlanName
from app.tools._client import get_http_client
from app.tools._circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class UpdateSubscriptionInput(BaseModel):
    user_id: str = Field(
        ..., 
        description="The unique identifier of the user (e.g., 'user_123'). Must be a non-empty string."
    )
    plan_name: PlanName = Field(
        ..., 
        description="The target subscription plan to update to. Must strictly be one of the Enum values: 'free', 'pro', or 'enterprise'."
    )
    idempotency_key: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique UUID idempotency key to prevent duplicate transactions or double mutation during retries."
    )

_subscription_cb = CircuitBreaker(
    threshold=settings.circuit_breaker_threshold, 
    cooldown=settings.circuit_breaker_cooldown
)

@retry(
    stop=stop_after_attempt(settings.retry_max_attempts),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    before_sleep=lambda rs: logger.warning(
        "Retry attempt %d for _post_subscription after: %s",
        rs.attempt_number, rs.outcome.exception()
    ),
)
async def _post_subscription(url: str, payload: dict, headers: dict) -> httpx.Response:
    client = await get_http_client()
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response

@_subscription_cb
async def update_subscription(input_data: UpdateSubscriptionInput) -> dict:
    """
    Mutates user's subscription asynchronously with idempotency, retry policies, and circuit breaker protection.
    """
    safe_user_id = quote(input_data.user_id)
    url = f"/v1/users/{safe_user_id}/subscription"
    headers = {"Idempotency-Key": input_data.idempotency_key}
    
    plan_val = input_data.plan_name.value if isinstance(input_data.plan_name, PlanName) else str(input_data.plan_name)
    payload = {"plan": plan_val}

    try:
        response = await _post_subscription(url, payload, headers)
        data = response.json()
        
        status_val = data.get("status")
        if status_val != "success":
            return {
                "status": "error",
                "code": "AMBIGUOUS_RESPONSE",
                "error_type": "UNEXPECTED_STATUS",
                "message": f"Backend returned unexpected status: '{status_val}'. State change is unconfirmed.",
                "retryable": False,
                "suggestion": "Call get_user_profile to verify actual current subscription state before retrying."
            }
        return {
            "status": "success", 
            "data": data, 
            "idempotency_key": input_data.idempotency_key
        }
        
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        return {
            "status": "error",
            "code": status_code,
            "error_type": "HTTP_ERROR",
            "message": f"HTTP {status_code}: {e.response.text}",
            "retryable": status_code >= 500,
            "suggestion": "If 5xx, retry with the SAME idempotency_key. If 4xx, correct parameters."
        }
    except (httpx.RequestError, httpx.TimeoutException) as e:
        return {
            "status": "error",
            "code": "NETWORK_ERROR",
            "error_type": "CONNECTION_FAILED",
            "message": f"Network error during subscription update: {str(e)}",
            "retryable": True,
            "suggestion": "Network failed during post. Retry using the EXACT SAME idempotency_key."
        }
