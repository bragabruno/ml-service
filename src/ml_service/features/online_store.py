from __future__ import annotations

import json
from typing import Any

import redis

from ml_service.app.config import get_settings


class OnlineFeatureStore:
    def __init__(self, redis_url: str | None = None) -> None:
        settings = get_settings()
        url = redis_url or settings.redis_url
        self._client = redis.from_url(url, decode_responses=True)  # type: ignore[no-untyped-call]
        self._prefix = "fraud:features"

    def get_user_context(self, user_id: str) -> dict[str, Any]:
        key = f"{self._prefix}:user:{user_id}"
        data = self._client.hgetall(key)
        if not data:
            return {}
        return {k: _parse_value(v) for k, v in data.items()}

    def get_device_context(self, device_id: str) -> dict[str, Any]:
        key = f"{self._prefix}:device:{device_id}"
        data = self._client.hgetall(key)
        if not data:
            return {}
        return {k: _parse_value(v) for k, v in data.items()}

    def get_merchant_context(self, merchant_id: str) -> dict[str, Any]:
        key = f"{self._prefix}:merchant:{merchant_id}"
        data = self._client.hgetall(key)
        if not data:
            return {}
        return {k: _parse_value(v) for k, v in data.items()}

    def set_user_context(self, user_id: str, context: dict[str, Any], ttl: int = 86400) -> None:
        key = f"{self._prefix}:user:{user_id}"
        self._client.hset(key, mapping={k: str(v) for k, v in context.items()})
        self._client.expire(key, ttl)

    def set_device_context(self, device_id: str, context: dict[str, Any], ttl: int = 86400) -> None:
        key = f"{self._prefix}:device:{device_id}"
        self._client.hset(key, mapping={k: str(v) for k, v in context.items()})
        self._client.expire(key, ttl)

    def set_merchant_context(self, merchant_id: str, context: dict[str, Any], ttl: int = 86400) -> None:
        key = f"{self._prefix}:merchant:{merchant_id}"
        self._client.hset(key, mapping={k: str(v) for k, v in context.items()})
        self._client.expire(key, ttl)

    def gather_context(self, user_id: str, device_id: str, merchant_id: str) -> dict[str, Any]:
        user_ctx = self.get_user_context(user_id)
        device_ctx = self.get_device_context(device_id)
        merchant_ctx = self.get_merchant_context(merchant_id)
        return {**user_ctx, **device_ctx, **merchant_ctx}

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.ConnectionError:
            return False


def _parse_value(v: str) -> Any:
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return v


_store: OnlineFeatureStore | None = None


def get_online_store() -> OnlineFeatureStore:
    global _store
    if _store is None:
        _store = OnlineFeatureStore()
    return _store
