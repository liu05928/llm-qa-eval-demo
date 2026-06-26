import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Tuple


try:
    import redis

    REDIS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on optional package
    redis = None
    REDIS_IMPORT_ERROR = str(exc)


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
AGENT_CACHE_TTL_SECONDS = int(os.getenv("AGENT_CACHE_TTL_SECONDS", "300"))
AGENT_RATE_LIMIT_PER_MINUTE = int(os.getenv("AGENT_RATE_LIMIT_PER_MINUTE", "20"))

_REDIS_CLIENT = None
_REDIS_DISABLED_REASON = ""


def get_redis_client():
    global _REDIS_CLIENT
    global _REDIS_DISABLED_REASON

    if redis is None:
        _REDIS_DISABLED_REASON = f"redis package not available: {REDIS_IMPORT_ERROR}"
        return None

    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=1.0,
        )
        client.ping()
        _REDIS_CLIENT = client
        _REDIS_DISABLED_REASON = ""
        return _REDIS_CLIENT
    except Exception as exc:
        _REDIS_DISABLED_REASON = str(exc)
        return None


def get_runtime_controls_status() -> Dict[str, Any]:
    client = get_redis_client()

    return {
        "redis_available": client is not None,
        "redis_url": REDIS_URL,
        "redis_disabled_reason": "" if client else _REDIS_DISABLED_REASON,
        "cache_ttl_seconds": AGENT_CACHE_TTL_SECONDS,
        "rate_limit_per_minute": AGENT_RATE_LIMIT_PER_MINUTE,
    }


def _json_default(value: Any):
    if hasattr(value, "to_dict"):
        return value.to_dict()

    return str(value)


def build_cache_key(namespace: str, payload: Dict[str, Any]) -> str:
    normalized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    client = get_redis_client()

    if not client:
        return None

    cached = client.get(cache_key)

    if not cached:
        return None

    try:
        return json.loads(cached)
    except json.JSONDecodeError:
        return None


def set_cached_response(
    cache_key: str,
    response: Dict[str, Any],
    ttl_seconds: int = AGENT_CACHE_TTL_SECONDS,
) -> bool:
    client = get_redis_client()

    if not client:
        return False

    client.setex(
        cache_key,
        ttl_seconds,
        json.dumps(response, ensure_ascii=False, default=_json_default),
    )
    return True


def check_rate_limit(
    identifier: str,
    limit: int = AGENT_RATE_LIMIT_PER_MINUTE,
    window_seconds: int = 60,
) -> Tuple[bool, Dict[str, Any]]:
    """Fixed-window API rate limiting. Redis failures are fail-open."""

    client = get_redis_client()

    if not client:
        return True, {
            "enabled": False,
            "allowed": True,
            "reason": _REDIS_DISABLED_REASON or "redis unavailable",
        }

    window = int(time.time() // window_seconds)
    key = f"rate_limit:agent:{identifier}:{window}"

    current = client.incr(key)

    if current == 1:
        client.expire(key, window_seconds)

    allowed = current <= limit
    ttl = client.ttl(key)

    return allowed, {
        "enabled": True,
        "allowed": allowed,
        "current": current,
        "limit": limit,
        "window_seconds": window_seconds,
        "reset_in_seconds": ttl,
    }
