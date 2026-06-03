from __future__ import annotations

import logging
import time
from typing import Any

import structlog
from prometheus_client import Counter, Histogram
from prometheus_client import generate_latest as _generate_latest

PREDICTION_COUNT = Counter(
    "fraud_predictions_total",
    "Total predictions served",
    ["model_version", "decision"],
)

PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

SCORE_HISTOGRAM = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud scores",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

INVESTIGATION_COUNT = Counter(
    "fraud_investigations_total",
    "Total agent investigations",
    ["disposition"],
)

MODEL_SWAPS = Counter(
    "fraud_model_swaps_total",
    "Total model hot-swaps",
)


def configure_logging(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    from typing import cast

    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))


class Timer:
    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


def metrics_endpoint() -> bytes:
    result: bytes = _generate_latest()
    return result
