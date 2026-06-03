from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ml_service.features.contract import FEATURE_NAMES, NUM_FEATURES

RISK_TIER_MAP = {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.9}


def compute_features(payload: dict[str, Any], context: dict[str, Any] | None = None) -> np.ndarray:
    ctx = context or {}
    amount = float(payload.get("amount", 0.0))
    now = datetime.now(UTC)

    txn_time = payload.get("created_at")
    if isinstance(txn_time, str):
        txn_dt = datetime.fromisoformat(txn_time.replace("Z", "+00:00"))
    elif isinstance(txn_time, datetime):
        txn_dt = txn_time if txn_time.tzinfo else txn_time.replace(tzinfo=UTC)
    else:
        txn_dt = now

    user_created = ctx.get("user_created_at")
    if isinstance(user_created, str):
        user_created_dt = datetime.fromisoformat(user_created.replace("Z", "+00:00"))
        account_age_days = (txn_dt - user_created_dt).total_seconds() / 86400.0
    else:
        account_age_days = 365.0

    device_first_seen = ctx.get("device_first_seen")
    if isinstance(device_first_seen, str):
        device_first_dt = datetime.fromisoformat(device_first_seen.replace("Z", "+00:00"))
        device_age_days = (txn_dt - device_first_dt).total_seconds() / 86400.0
    else:
        device_age_days = 30.0

    merchant_risk_tier = ctx.get("merchant_risk_tier", "MEDIUM")
    country = payload.get("country", "US")
    home_country = ctx.get("home_country", "US")
    mcc = ctx.get("mcc", "5411")

    country_risk_map = ctx.get("country_risk_map", {})
    mcc_risk_map = ctx.get("mcc_risk_map", {})

    avg_amount_30d = ctx.get("avg_amount_30d", amount)
    prev_txn_ts = ctx.get("prev_txn_ts")
    if isinstance(prev_txn_ts, str):
        prev_dt = datetime.fromisoformat(prev_txn_ts.replace("Z", "+00:00"))
        days_since_last = (txn_dt - prev_dt).total_seconds() / 86400.0
    else:
        days_since_last = 30.0

    features: dict[str, float] = {
        "amount": amount,
        "amount_log": math.log1p(amount),
        "amount_usd": amount,
        "txn_hour": txn_dt.hour,
        "txn_day_of_week": txn_dt.weekday(),
        "is_weekend": float(txn_dt.weekday() >= 5),
        "is_night": float(0 <= txn_dt.hour <= 4),
        "velocity_5m": float(ctx.get("velocity_5m", 0)),
        "velocity_1h": float(ctx.get("velocity_1h", 0)),
        "velocity_24h": float(ctx.get("velocity_24h", 0)),
        "amount_5m": float(ctx.get("amount_5m", 0.0)),
        "amount_1h": float(ctx.get("amount_1h", 0.0)),
        "amount_24h": float(ctx.get("amount_24h", 0.0)),
        "user_account_age_days": account_age_days,
        "user_txn_count_30d": float(ctx.get("user_txn_count_30d", 0)),
        "user_failed_attempts_24h": float(payload.get("failed_attempts", 0)),
        "device_age_days": device_age_days,
        "device_is_trusted": float(ctx.get("device_is_trusted", False)),
        "device_is_new": float(device_age_days <= 7),
        "device_txn_count_24h": float(ctx.get("device_txn_count_24h", 0)),
        "merchant_risk_score": RISK_TIER_MAP.get(merchant_risk_tier, 0.5),
        "merchant_txn_volume_24h": float(ctx.get("merchant_txn_volume_24h", 0)),
        "merchant_fraud_rate_30d": float(ctx.get("merchant_fraud_rate_30d", 0.0)),
        "country_risk": float(country_risk_map.get(country, 0.5)),
        "is_foreign_country": float(country != home_country),
        "ip_country_mismatch": float(ctx.get("ip_country_mismatch", False)),
        "mcc_risk": float(mcc_risk_map.get(mcc, 0.3)),
        "days_since_last_txn": days_since_last,
        "avg_amount_30d": float(avg_amount_30d),
        "amount_to_avg_ratio": amount / max(avg_amount_30d, 1e-6),
        "distinct_merchants_24h": float(ctx.get("distinct_merchants_24h", 0)),
        "distinct_countries_24h": float(ctx.get("distinct_countries_24h", 0)),
        "chargeback_count_90d": float(ctx.get("chargeback_count_90d", 0)),
        "has_chargeback_history": float(ctx.get("chargeback_count_90d", 0) > 0),
    }

    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def feature_importance_names(top_indices: list[int]) -> list[str]:
    return [FEATURE_NAMES[i] for i in top_indices if 0 <= i < NUM_FEATURES]
