from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEED = 42
NUM_USERS = 500
NUM_MERCHANTS = 80
NUM_DEVICES_PER_USER = (1, 4)
NUM_TRANSACTIONS = 10_000
FRAUD_RATE = 0.05
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 5, 31, tzinfo=timezone.utc)

COUNTRIES = ["US", "GB", "BR", "DE", "FR", "NG", "RU", "CN", "IN", "JP", "CA", "AU", "MX", "KR", "ZA"]
CURRENCIES = ["USD", "GBP", "BRL", "EUR", "EUR", "NGN", "RUB", "CNY", "INR", "JPY", "CAD", "AUD", "MXN", "KRW", "ZAR"]
COUNTRY_RISK = {"US": 0.1, "GB": 0.1, "BR": 0.3, "DE": 0.1, "FR": 0.1, "NG": 0.7, "RU": 0.8, "CN": 0.5, "IN": 0.3, "JP": 0.1, "CA": 0.1, "AU": 0.1, "MX": 0.4, "KR": 0.2, "ZA": 0.5}
MCC_CODES = ["5411", "5812", "5912", "7011", "5541", "5691", "5944", "7995", "6011", "4111"]
MCC_RISK = {"5411": 0.1, "5812": 0.15, "5912": 0.2, "7011": 0.2, "5541": 0.25, "5691": 0.3, "5944": 0.4, "7995": 0.8, "6011": 0.7, "4111": 0.1}
RISK_TIERS = ["LOW", "MEDIUM", "HIGH"]
DEVICE_TYPES = ["DESKTOP", "MOBILE", "TABLET"]
ROLES = ["ADMIN", "FRAUD_ANALYST", "INVESTIGATOR", "AUDITOR", "SYSTEM_ACCOUNT"]
TXN_STATUSES = ["RECEIVED", "SCORING", "APPROVED", "IN_REVIEW", "DECLINED"]
DECISIONS = ["APPROVE", "REVIEW", "DECLINE"]
CASE_STATUSES = ["OPEN", "ASSIGNED", "IN_REVIEW", "RESOLVED_FRAUD", "RESOLVED_LEGIT", "ESCALATED", "CLOSED"]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def uuid7() -> str:
    return str(uuid.uuid4())


def random_ts(start: datetime, end: datetime) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, delta))


def generate_users(rng: random.Random) -> pd.DataFrame:
    rows = []
    for i in range(NUM_USERS):
        created = random_ts(START_DATE - timedelta(days=730), START_DATE)
        rows.append({
            "id": uuid7(),
            "username": f"user_{i:05d}",
            "email": f"user_{i:05d}@example.com",
            "password_hash": f"$2b$12${rng.randbytes(16).hex()}",
            "role": rng.choice(["FRAUD_ANALYST", "FRAUD_ANALYST", "FRAUD_ANALYST", "ADMIN", "INVESTIGATOR"]),
            "status": rng.choice(["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "DISABLED"]),
            "home_country": rng.choice(COUNTRIES[:8]),
            "created_at": created.isoformat(),
            "updated_at": created.isoformat(),
            "version": 1,
        })
    return pd.DataFrame(rows)


def generate_merchants(rng: random.Random) -> pd.DataFrame:
    rows = []
    for i in range(NUM_MERCHANTS):
        rows.append({
            "id": uuid7(),
            "name": f"Merchant_{i:04d}",
            "mcc": rng.choice(MCC_CODES),
            "risk_tier": rng.choices(RISK_TIERS, weights=[0.6, 0.3, 0.1])[0],
            "country": rng.choice(COUNTRIES),
        })
    return pd.DataFrame(rows)


def generate_devices(users: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    rows = []
    for _, user in users.iterrows():
        n_devices = rng.randint(*NUM_DEVICES_PER_USER)
        for _ in range(n_devices):
            first = random_ts(START_DATE - timedelta(days=365), END_DATE)
            rows.append({
                "id": uuid7(),
                "fingerprint": rng.randbytes(16).hex(),
                "type": rng.choice(DEVICE_TYPES),
                "trusted": rng.random() < 0.3,
                "first_seen": first.isoformat(),
                "last_seen": random_ts(first, END_DATE).isoformat(),
            })
    return pd.DataFrame(rows)


def generate_transactions(
    users: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
    rng: random.Random,
) -> pd.DataFrame:
    user_ids = users["id"].tolist()
    merchant_ids = merchants["id"].tolist()
    device_ids = devices["id"].tolist()

    fraud_indices = set(rng.sample(range(NUM_TRANSACTIONS), int(NUM_TRANSACTIONS * FRAUD_RATE)))
    rows = []

    for i in range(NUM_TRANSACTIONS):
        is_fraud = i in fraud_indices
        user_id = rng.choice(user_ids)

        if is_fraud:
            pattern = rng.choice(["velocity_burst", "new_device_foreign", "high_amount", "chargeback_history"])
            if pattern == "velocity_burst":
                base_ts = random_ts(START_DATE, END_DATE)
                amount = rng.uniform(50, 500)
                country = rng.choice(COUNTRIES[:8])
            elif pattern == "new_device_foreign":
                base_ts = random_ts(START_DATE, END_DATE)
                amount = rng.uniform(200, 5000)
                country = rng.choice(["NG", "RU", "CN", "ZA"])
            elif pattern == "high_amount":
                base_ts = random_ts(START_DATE, END_DATE)
                amount = rng.uniform(5000, 50000)
                country = rng.choice(COUNTRIES)
            else:
                base_ts = random_ts(START_DATE, END_DATE)
                amount = rng.uniform(100, 2000)
                country = rng.choice(COUNTRIES)
        else:
            base_ts = random_ts(START_DATE, END_DATE)
            amount = rng.lognormvariate(4.5, 1.2)
            country = rng.choices(COUNTRIES, weights=[0.4, 0.1, 0.1, 0.05, 0.05, 0.02, 0.01, 0.02, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])[0]

        rows.append({
            "id": uuid7(),
            "user_id": user_id,
            "merchant_id": rng.choice(merchant_ids),
            "device_id": rng.choice(device_ids),
            "amount": round(amount, 2),
            "currency": CURRENCIES[COUNTRIES.index(country)] if country in COUNTRIES else "USD",
            "ip_address": f"{rng.randint(1,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
            "country": country,
            "status": rng.choice(TXN_STATUSES) if not is_fraud else rng.choice(["DECLINED", "IN_REVIEW", "APPROVED"]),
            "idempotency_key": uuid7(),
            "created_at": base_ts.isoformat(),
            "_is_fraud": is_fraud,
        })

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def generate_risk_scores(transactions: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    model_version_id = uuid7()
    rows = []
    for _, txn in transactions.iterrows():
        is_fraud = txn["_is_fraud"]
        if is_fraud:
            ml_score = rng.uniform(0.5, 0.98)
            rules_score = rng.uniform(0.3, 0.9)
        else:
            ml_score = rng.uniform(0.01, 0.5)
            rules_score = rng.uniform(0.01, 0.4)

        agg = 0.6 * ml_score + 0.4 * rules_score
        if agg > 0.7:
            decision = "DECLINE"
        elif agg > 0.4:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        reason_codes = []
        if ml_score > 0.7:
            reason_codes.append("HIGH_ML_SCORE")
        if rules_score > 0.6:
            reason_codes.append("RULE_TRIGGER")
        if txn["amount"] > 5000:
            reason_codes.append("HIGH_AMOUNT")

        rows.append({
            "id": uuid7(),
            "transaction_id": txn["id"],
            "model_version_id": model_version_id,
            "ml_score": round(ml_score, 6),
            "rules_score": round(rules_score, 6),
            "aggregate_score": round(agg, 6),
            "decision": decision,
            "degraded_mode": False,
            "reason_codes": json.dumps(reason_codes),
            "created_at": txn["created_at"].isoformat(),
        })
    return pd.DataFrame(rows)


def generate_fraud_cases(
    transactions: pd.DataFrame,
    risk_scores: pd.DataFrame,
    users: pd.DataFrame,
    rng: random.Random,
) -> pd.DataFrame:
    review_scores = risk_scores[risk_scores["decision"].isin(["REVIEW", "DECLINE"])]
    analysts = users[users["role"] == "FRAUD_ANALYST"]["id"].tolist()
    rows = []
    for _, score in review_scores.iterrows():
        txn = transactions[transactions["id"] == score["transaction_id"]].iloc[0]
        opened = txn["created_at"] + timedelta(minutes=rng.randint(1, 60))
        is_fraud = txn["_is_fraud"]

        if rng.random() < 0.7:
            status = "RESOLVED_FRAUD" if is_fraud else "RESOLVED_LEGIT"
            resolved = opened + timedelta(hours=rng.randint(1, 72))
        else:
            status = rng.choice(["OPEN", "ASSIGNED", "IN_REVIEW"])
            resolved = None

        rows.append({
            "id": uuid7(),
            "transaction_id": score["transaction_id"],
            "risk_score_id": score["id"],
            "assignee_id": rng.choice(analysts) if analysts else uuid7(),
            "status": status,
            "severity": rng.choices(SEVERITIES, weights=[0.2, 0.3, 0.35, 0.15])[0],
            "opened_at": opened.isoformat(),
            "sla_due_at": (opened + timedelta(hours=48)).isoformat(),
            "resolved_at": resolved.isoformat() if resolved else None,
        })
    return pd.DataFrame(rows)


def generate_fraud_labels(
    fraud_cases: pd.DataFrame,
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    rng: random.Random,
) -> pd.DataFrame:
    analysts = users[users["role"] == "FRAUD_ANALYST"]["id"].tolist()
    resolved = fraud_cases[fraud_cases["status"].isin(["RESOLVED_FRAUD", "RESOLVED_LEGIT"])]
    rows = []
    for _, case in resolved.iterrows():
        label = "FRAUD" if case["status"] == "RESOLVED_FRAUD" else "LEGITIMATE"
        rows.append({
            "id": uuid7(),
            "transaction_id": case["transaction_id"],
            "case_id": case["id"],
            "analyst_id": rng.choice(analysts) if analysts else uuid7(),
            "label": label,
            "confidence": round(rng.uniform(0.7, 1.0), 4),
            "reason": f"Analyst review: {label.lower()} confirmed",
            "labeled_at": case["resolved_at"],
        })
    return pd.DataFrame(rows)


def generate_model_versions(rng: random.Random) -> pd.DataFrame:
    rows = []
    for i, (version, status) in enumerate([
        ("v1.0.0", "ARCHIVED"),
        ("v1.1.0", "ARCHIVED"),
        ("v1.2.0", "DEPLOYED"),
    ]):
        created = START_DATE + timedelta(days=30 * i)
        rows.append({
            "id": uuid7(),
            "version": version,
            "mlflow_run_id": uuid7(),
            "metrics": json.dumps({
                "pr_auc": round(rng.uniform(0.7, 0.95), 4),
                "roc_auc": round(rng.uniform(0.85, 0.98), 4),
                "recall": round(rng.uniform(0.7, 0.95), 4),
                "fp_rate": round(rng.uniform(0.01, 0.1), 4),
            }),
            "status": status,
            "deployed_at": created.isoformat() if status == "DEPLOYED" else None,
            "created_at": created.isoformat(),
        })
    return pd.DataFrame(rows)


def generate_audit_events(rng: random.Random) -> pd.DataFrame:
    actions = ["MODEL_DEPLOYED", "CASE_RESOLVED", "LABEL_CREATED", "USER_LOGIN", "CONFIG_CHANGED"]
    rows = []
    for _ in range(200):
        rows.append({
            "id": uuid7(),
            "actor": f"system" if rng.random() < 0.3 else f"user_{rng.randint(0, NUM_USERS-1):05d}",
            "action": rng.choice(actions),
            "target_type": rng.choice(["model_version", "fraud_case", "fraud_label", "user"]),
            "target_id": uuid7(),
            "before": json.dumps({}),
            "after": json.dumps({"status": "changed"}),
            "correlation_id": uuid7(),
            "created_at": random_ts(START_DATE, END_DATE).isoformat(),
        })
    return pd.DataFrame(rows)


def load_to_duckdb(
    conn: duckdb.DuckDBConnection,
    tables: dict[str, pd.DataFrame],
) -> None:
    for name, df in tables.items():
        drop_cols = [c for c in df.columns if c.startswith("_")]
        clean = df.drop(columns=drop_cols) if drop_cols else df
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM clean")
        print(f"  {name}: {len(clean)} rows")


def main() -> None:
    global NUM_TRANSACTIONS, FRAUD_RATE

    parser = argparse.ArgumentParser(description="Generate synthetic fraud data")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-transactions", type=int, default=NUM_TRANSACTIONS)
    parser.add_argument("--fraud-rate", type=float, default=FRAUD_RATE)
    parser.add_argument("--output", type=str, default="./data/fraud.duckdb")
    args = parser.parse_args()

    NUM_TRANSACTIONS = args.num_transactions
    FRAUD_RATE = args.fraud_rate

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f"Generating synthetic data (seed={args.seed}, txns={NUM_TRANSACTIONS}, fraud_rate={FRAUD_RATE})...")

    users = generate_users(rng)
    merchants = generate_merchants(rng)
    devices = generate_devices(users, rng)
    transactions = generate_transactions(users, merchants, devices, rng)
    risk_scores = generate_risk_scores(transactions, rng)
    fraud_cases = generate_fraud_cases(transactions, risk_scores, users, rng)
    fraud_labels = generate_fraud_labels(fraud_cases, transactions, users, rng)
    model_versions = generate_model_versions(rng)
    audit_events = generate_audit_events(rng)

    tables = {
        "users": users,
        "merchants": merchants,
        "devices": devices,
        "transactions": transactions,
        "risk_scores": risk_scores,
        "fraud_cases": fraud_cases,
        "fraud_labels": fraud_labels,
        "model_versions": model_versions,
        "audit_events": audit_events,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(output_path))
    print(f"\nLoading to DuckDB ({output_path}):")
    load_to_duckdb(conn, tables)

    fraud_count = transactions["_is_fraud"].sum()
    print(f"\nSummary: {NUM_TRANSACTIONS} transactions, {fraud_count} fraud ({fraud_count/NUM_TRANSACTIONS:.1%})")
    print(f"  {len(fraud_cases)} cases, {len(fraud_labels)} labels")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
