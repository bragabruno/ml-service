from __future__ import annotations

import json

from ml_service.agent.case_integration import handle_review_required
from ml_service.agent.llm.factory import get_llm_client
from ml_service.app.config import get_settings
from ml_service.app.observability import get_logger
from ml_service.events.topics import Topic
from ml_service.integrations.case_management import get_case_client
from ml_service.schemas.investigation import ReviewRequiredEvent

logger = get_logger("review_consumer")


async def run_review_consumer() -> None:
    """Production ingress: consume `fraud.review.required` and auto-draft reports.

    `aiokafka` is imported lazily (optional `kafka` extra) so importing this module never
    requires a broker; tests and the API server do not start this loop. The actual work is
    delegated to the same `handle_review_required` core used by the HTTP endpoint.
    """
    from aiokafka import AIOKafkaConsumer

    settings = get_settings()
    llm = get_llm_client()
    case_client = get_case_client()

    consumer = AIOKafkaConsumer(
        Topic.FRAUD_REVIEW_REQUIRED.value,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="ml-service.review-autodraft",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    await consumer.start()
    logger.info("review_consumer_started", topic=Topic.FRAUD_REVIEW_REQUIRED.value)
    try:
        async for message in consumer:
            try:
                event = ReviewRequiredEvent.model_validate(message.value)
                handle_review_required(event, llm, case_client)
            except Exception as exc:  # noqa: BLE001 - one bad event must not kill the consumer
                logger.warning("review_event_failed", error=str(exc))
    finally:
        await consumer.stop()
        logger.info("review_consumer_stopped")
