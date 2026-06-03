from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FeatureType(str, Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: FeatureType
    description: str = ""
    nullable: bool = False


CONTRACT_VERSION = "1.0.0"

FEATURES: tuple[Feature, ...] = (
    Feature("amount", FeatureType.FLOAT, "Transaction amount in base currency"),
    Feature("amount_log", FeatureType.FLOAT, "log1p(amount)"),
    Feature("amount_usd", FeatureType.FLOAT, "Amount converted to USD"),
    Feature("txn_hour", FeatureType.INT, "Hour of day (0-23) UTC"),
    Feature("txn_day_of_week", FeatureType.INT, "Day of week (0=Mon, 6=Sun)"),
    Feature("is_weekend", FeatureType.BOOL, "Transaction on Sat/Sun"),
    Feature("is_night", FeatureType.BOOL, "Transaction between 00:00-05:00 UTC"),
    Feature("velocity_5m", FeatureType.INT, "Txn count for same user in last 5 minutes"),
    Feature("velocity_1h", FeatureType.INT, "Txn count for same user in last 1 hour"),
    Feature("velocity_24h", FeatureType.INT, "Txn count for same user in last 24 hours"),
    Feature("amount_5m", FeatureType.FLOAT, "Sum of amounts for same user in last 5 minutes"),
    Feature("amount_1h", FeatureType.FLOAT, "Sum of amounts for same user in last 1 hour"),
    Feature("amount_24h", FeatureType.FLOAT, "Sum of amounts for same user in last 24 hours"),
    Feature("user_account_age_days", FeatureType.FLOAT, "Days since user account creation"),
    Feature("user_txn_count_30d", FeatureType.INT, "User total txn count in last 30 days"),
    Feature("user_failed_attempts_24h", FeatureType.INT, "Failed txn attempts for user in last 24h"),
    Feature("device_age_days", FeatureType.FLOAT, "Days since device first_seen"),
    Feature("device_is_trusted", FeatureType.BOOL, "Device marked as trusted"),
    Feature("device_is_new", FeatureType.BOOL, "Device first_seen within last 7 days"),
    Feature("device_txn_count_24h", FeatureType.INT, "Txn count from this device in last 24h"),
    Feature("merchant_risk_score", FeatureType.FLOAT, "Merchant risk tier as numeric (LOW=0.1, MED=0.5, HIGH=0.9)"),
    Feature("merchant_txn_volume_24h", FeatureType.INT, "Merchant txn volume in last 24h"),
    Feature("merchant_fraud_rate_30d", FeatureType.FLOAT, "Merchant fraud rate in last 30 days"),
    Feature("country_risk", FeatureType.FLOAT, "Country risk score (0-1)"),
    Feature("is_foreign_country", FeatureType.BOOL, "Txn country differs from user home country"),
    Feature("ip_country_mismatch", FeatureType.BOOL, "IP geolocation differs from declared country"),
    Feature("mcc_risk", FeatureType.FLOAT, "MCC category risk score (0-1)"),
    Feature("days_since_last_txn", FeatureType.FLOAT, "Days since user's previous transaction"),
    Feature("avg_amount_30d", FeatureType.FLOAT, "User average txn amount in last 30 days"),
    Feature("amount_to_avg_ratio", FeatureType.FLOAT, "Current amount / user 30d average"),
    Feature("distinct_merchants_24h", FeatureType.INT, "Distinct merchants for user in last 24h"),
    Feature("distinct_countries_24h", FeatureType.INT, "Distinct countries for user in last 24h"),
    Feature("chargeback_count_90d", FeatureType.INT, "User chargeback count in last 90 days"),
    Feature("has_chargeback_history", FeatureType.BOOL, "User has any prior chargebacks"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURE_DTYPES: dict[str, FeatureType] = {f.name: f.dtype for f in FEATURES}
NUM_FEATURES: int = len(FEATURES)


def export_contract_json(path: str | Path) -> Path:
    out = Path(path)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "num_features": NUM_FEATURES,
        "features": [
            {"name": f.name, "dtype": f.dtype.value, "description": f.description, "nullable": f.nullable}
            for f in FEATURES
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def load_contract_json(path: str | Path) -> dict:
    data: dict = json.loads(Path(path).read_text())
    return data


if __name__ == "__main__":
    out = export_contract_json("data/feature_contract.json")
    print(f"Exported {NUM_FEATURES} features to {out}")
