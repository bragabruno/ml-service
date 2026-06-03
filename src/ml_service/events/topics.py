from __future__ import annotations

from enum import Enum


class Topic(str, Enum):
    """The platform's seven Kafka domain topics (docs/diagrams/01-architecture-c4.md).

    The Kafka client itself is provided by the deployment environment (the JVM backend
    owns the broker); this module is the shared topic contract so producers/consumers on
    the Python side reference the exact same names.
    """

    TRANSACTIONS_CREATED = "transactions.created"
    FRAUD_SCORED = "fraud.scored"
    FRAUD_REVIEW_REQUIRED = "fraud.review.required"
    FRAUD_CONFIRMED = "fraud.confirmed"
    FRAUD_FALSE_POSITIVE = "fraud.falsepositive"
    FRAUD_MODEL_DEPLOYED = "fraud.model.deployed"
    FRAUD_RETRAINING_REQUESTED = "fraud.retraining.requested"


# What the ml-service consumes from and produces to (see README "Platform Contracts").
CONSUMES: tuple[Topic, ...] = (
    Topic.FRAUD_REVIEW_REQUIRED,
    Topic.FRAUD_CONFIRMED,
    Topic.FRAUD_FALSE_POSITIVE,
    Topic.FRAUD_MODEL_DEPLOYED,
)
PRODUCES: tuple[Topic, ...] = (Topic.FRAUD_RETRAINING_REQUESTED,)

ALL_TOPICS: tuple[str, ...] = tuple(t.value for t in Topic)
