from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    tool: str
    data: dict[str, Any]
    finding: str


def _seed_rng(transaction_id: str, tool: str) -> random.Random:
    seed = int(hashlib.sha256(f"{transaction_id}:{tool}".encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def get_transaction_features(transaction_id: str) -> ToolResult:
    rng = _seed_rng(transaction_id, "features")
    amount = round(rng.uniform(50, 10000), 2)
    country = rng.choice(["US", "US", "US", "GB", "NG", "BR", "RU", "CN"])
    new_device = rng.random() < 0.35
    failed_attempts = rng.randint(0, 8)
    velocity_5m = rng.randint(0, 4)
    velocity_24h = rng.randint(1, 12)
    account_age_days = rng.randint(7, 1200)
    chargeback_count = rng.randint(0, 5)
    merchant_risk = round(rng.uniform(0.0, 1.0), 2)

    data = {
        "transaction_id": transaction_id,
        "amount": amount,
        "currency": "USD",
        "country": country,
        "new_device": new_device,
        "failed_attempts": failed_attempts,
        "velocity_5m": velocity_5m,
        "velocity_24h": velocity_24h,
        "account_age_days": account_age_days,
        "chargeback_count": chargeback_count,
        "merchant_risk_score": merchant_risk,
    }
    finding = (
        f"amount={amount}, country={country}, new_device={new_device}, "
        f"velocity_24h={velocity_24h}, failed_attempts={failed_attempts}"
    )
    return ToolResult(tool="get_transaction_features", data=data, finding=finding)


def get_velocity(transaction_id: str) -> ToolResult:
    rng = _seed_rng(transaction_id, "velocity")
    v5m = rng.randint(0, 5)
    v1h = rng.randint(0, 10)
    v24h = rng.randint(1, 15)
    amt_24h = round(rng.uniform(100, 25000), 2)

    data = {
        "transaction_id": transaction_id,
        "transactions_5m": v5m,
        "transactions_1h": v1h,
        "transactions_24h": v24h,
        "amount_24h": amt_24h,
    }
    finding = f"{v5m} txns in 5m, {v1h} in 1h, {v24h} in 24h (total ${amt_24h})"
    return ToolResult(tool="get_velocity", data=data, finding=finding)


def get_similar_cases(transaction_id: str) -> ToolResult:
    rng = _seed_rng(transaction_id, "similar")
    n_cases = rng.randint(2, 6)
    cases = []
    for i in range(n_cases):
        outcome = rng.choice(["FRAUD", "LEGITIMATE", "LEGITIMATE"])
        cases.append({
            "case_id": f"CASE-{rng.randint(1000, 9999)}",
            "outcome": outcome,
            "similarity": round(rng.uniform(0.6, 0.98), 2),
        })

    fraud_count = sum(1 for c in cases if c["outcome"] == "FRAUD")
    data = {"transaction_id": transaction_id, "similar_cases": cases}
    finding = f"{n_cases} similar cases found: {fraud_count} fraud, {n_cases - fraud_count} legitimate"
    return ToolResult(tool="get_similar_cases", data=data, finding=finding)


def get_shap_explanation(transaction_id: str) -> ToolResult:
    rng = _seed_rng(transaction_id, "shap")
    features = [
        ("new_device", round(rng.uniform(-0.1, 0.4), 3)),
        ("country_risk", round(rng.uniform(-0.05, 0.35), 3)),
        ("velocity_24h", round(rng.uniform(-0.1, 0.3), 3)),
        ("amount", round(rng.uniform(-0.05, 0.25), 3)),
        ("failed_attempts", round(rng.uniform(-0.05, 0.2), 3)),
        ("account_age_days", round(rng.uniform(-0.2, 0.05), 3)),
        ("merchant_risk_score", round(rng.uniform(-0.05, 0.15), 3)),
    ]
    features.sort(key=lambda x: abs(x[1]), reverse=True)

    data = {"transaction_id": transaction_id, "shap_values": features}
    top3 = ", ".join(f"{f[0]}({f[1]:+.3f})" for f in features[:3])
    finding = f"Top SHAP contributors: {top3}"
    return ToolResult(tool="get_shap_explanation", data=data, finding=finding)


def get_rule_hits(transaction_id: str) -> ToolResult:
    rng = _seed_rng(transaction_id, "rules")
    all_rules = [
        ("NEW_DEVICE_FOREIGN_COUNTRY", "New device from a foreign country"),
        ("HIGH_VELOCITY_5M", "More than 3 transactions in 5 minutes"),
        ("HIGH_VELOCITY_24H", "More than 10 transactions in 24 hours"),
        ("HIGH_AMOUNT", "Transaction amount exceeds $5000"),
        ("HIGH_FAILED_ATTEMPTS", "More than 5 failed login attempts"),
        ("HIGH_RISK_MERCHANT", "Transaction at a high-risk merchant"),
        ("KNOWN_FRAUD_DEVICE", "Device fingerprint linked to prior fraud"),
    ]
    n_hits = rng.randint(0, 4)
    hits = rng.sample(all_rules, min(n_hits, len(all_rules)))

    data = {"transaction_id": transaction_id, "rule_hits": hits}
    if hits:
        finding = f"{len(hits)} rules triggered: {', '.join(h[0] for h in hits)}"
    else:
        finding = "No rules triggered"
    return ToolResult(tool="get_rule_hits", data=data, finding=finding)


def get_customer_history(transaction_id: str) -> ToolResult:
    rng = _seed_rng(transaction_id, "customer")
    account_age = rng.randint(30, 1500)
    total_txns = rng.randint(5, 200)
    chargebacks = rng.randint(0, 8)
    fraud_reports = rng.randint(0, 3)
    avg_amount = round(rng.uniform(50, 2000), 2)

    data = {
        "transaction_id": transaction_id,
        "account_age_days": account_age,
        "total_transactions": total_txns,
        "chargeback_count": chargebacks,
        "fraud_reports": fraud_reports,
        "average_transaction_amount": avg_amount,
    }
    finding = (
        f"account_age={account_age}d, {total_txns} total txns, "
        f"{chargebacks} chargebacks, avg_amt=${avg_amount}"
    )
    return ToolResult(tool="get_customer_history", data=data, finding=finding)


TOOL_REGISTRY: dict[str, Any] = {
    "get_transaction_features": get_transaction_features,
    "get_velocity": get_velocity,
    "get_similar_cases": get_similar_cases,
    "get_shap_explanation": get_shap_explanation,
    "get_rule_hits": get_rule_hits,
    "get_customer_history": get_customer_history,
}
