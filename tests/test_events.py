from __future__ import annotations

from ml_service.events.topics import ALL_TOPICS, CONSUMES, PRODUCES, Topic


def test_seven_domain_topics() -> None:
    assert len(ALL_TOPICS) == 7
    assert Topic.FRAUD_MODEL_DEPLOYED.value == "fraud.model.deployed"
    assert "transactions.created" in ALL_TOPICS


def test_consume_and_produce_contracts() -> None:
    assert Topic.FRAUD_RETRAINING_REQUESTED in PRODUCES
    assert Topic.FRAUD_CONFIRMED in CONSUMES
    assert Topic.FRAUD_MODEL_DEPLOYED in CONSUMES
    # ml-service consumes review.required to auto-draft investigation reports (FRAUD-191).
    assert Topic.FRAUD_REVIEW_REQUIRED in CONSUMES
