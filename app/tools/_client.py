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

async def close_http_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
