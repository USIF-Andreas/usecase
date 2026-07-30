from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings.
    
    Integrates with the scenario_resource_catalog:
    - model_family: "claude-3-5-sonnet" or "gpt-4o"
    - autonomy_level: "autonomous-within-budget"
    - cost_limit_usd: 0.50 (p95_latency_ms: 1500-8000)
    """
    model_family: str = "gpt-4o"
    autonomy_level: str = "autonomous-within-budget"
    max_steps: int = 12
    cost_limit_usd: float = 0.50
    user_api_base_url: str = "https://api.mockservice.local"

    # Enterprise Resilience Settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown: float = 30.0
    cache_ttl: float = 30.0
    retry_max_attempts: int = 3
    max_context_turns: int = 6

settings = Settings()
