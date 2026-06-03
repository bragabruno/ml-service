from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ml_service.app.config import get_settings
from ml_service.app.observability import get_logger

logger = get_logger("event_producer")


@runtime_checkable
class EventProducer(Protocol):
    """Publishes domain events. Mirrors the LLM/case-client factory: mock for offline/test,
    Kafka for production — selected by `get_event_producer()`.
    """

    def publish(self, topic: str, payload: dict[str, Any]) -> None: ...


@dataclass
class _Published:
    topic: str
    payload: dict[str, Any]


@dataclass
class MockEventProducer:
    """Deterministic, offline producer — records in memory and logs. Default provider."""

    published: list[_Published] = field(default_factory=list)

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append(_Published(topic=topic, payload=payload))
        logger.info("event_published", topic=topic, request_id=payload.get("request_id"))


class KafkaEventProducer:
    """Production producer — publishes to Kafka via a short-lived aiokafka producer.

    `aiokafka` is lazy-imported (optional `kafka` extra) so importing this module needs no broker.
    Trigger events are low-volume, so a per-publish connect/send/close is acceptable.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap_servers = bootstrap_servers

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        import asyncio
        import json

        async def _send() -> None:
            from aiokafka import AIOKafkaProducer

            producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await producer.start()
            try:
                await producer.send_and_wait(topic, payload)
            finally:
                await producer.stop()

        asyncio.run(_send())
        logger.info("event_published", topic=topic, request_id=payload.get("request_id"))


_producer: EventProducer | None = None


def get_event_producer() -> EventProducer:
    """Return the configured event producer (mock by default)."""
    global _producer
    if _producer is not None:
        return _producer

    settings = get_settings()
    provider = settings.event_producer_provider.lower()
    if provider == "mock":
        _producer = MockEventProducer()
    elif provider == "kafka":
        _producer = KafkaEventProducer(settings.kafka_bootstrap_servers)
    else:
        raise ValueError(f"Unknown EVENT_PRODUCER_PROVIDER: {provider!r}. Use 'mock' or 'kafka'.")
    return _producer


__all__ = [
    "EventProducer",
    "KafkaEventProducer",
    "MockEventProducer",
    "get_event_producer",
]
