import time
import threading
import functools
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Per-tool circuit breaker with half-open probe support."""
    
    def __init__(self, threshold: int = 5, cooldown: float = 30.0):
        self.failure_count = 0
        self.threshold = threshold
        self.cooldown = cooldown
        self.last_failure: float = 0.0
        self._lock = threading.Lock()
    
    @property
    def is_open(self) -> bool:
        with self._lock:
            if self.failure_count >= self.threshold:
                if time.time() - self.last_failure < self.cooldown:
                    return True
            return False
    
    def record_success(self):
        with self._lock:
            self.failure_count = 0
    
    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure = time.time()
    
    def reset(self):
        with self._lock:
            self.failure_count = 0
            self.last_failure = 0.0
    
    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if self.is_open:
                logger.warning("Circuit OPEN for %s — short-circuiting", func.__name__)
                return {
                    "status": "error",
                    "code": "CIRCUIT_OPEN",
                    "error_type": "CIRCUIT_BREAKER_TRIPPED",
                    "message": f"Circuit breaker is open for '{func.__name__}'. Too many consecutive failures. Will retry after cooldown.",
                    "retryable": False,
                    "suggestion": "The downstream service is experiencing issues. Wait before retrying."
                }
            
            result = await func(*args, **kwargs)
            
            if isinstance(result, dict) and result.get("status") == "error":
                code = result.get("code", 0)
                if isinstance(code, int) and code >= 500:
                    self.record_failure()
                elif code in ("NETWORK_ERROR", "CONNECTION_FAILED"):
                    self.record_failure()
                else:
                    self.record_success()
            else:
                self.record_success()
            
            return result
        return wrapper
