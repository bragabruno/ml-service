from __future__ import annotations

from ml_service.features.contract import CONTRACT_VERSION, FEATURE_DTYPES, FEATURE_NAMES, NUM_FEATURES

DBT_FEATURE_COLUMNS = (
    "amount",
    "amount_log",
    "amount_usd",
    "txn_hour",
    "txn_day_of_week",
    "is_weekend",
    "is_night",
    "velocity_5m",
    "velocity_1h",
    "velocity_24h",
    "amount_5m",
    "amount_1h",
    "amount_24h",
    "user_account_age_days",
    "user_txn_count_30d",
    "user_failed_attempts_24h",
    "device_age_days",
    "device_is_trusted",
    "device_is_new",
    "device_txn_count_24h",
    "merchant_risk_score",
    "merchant_txn_volume_24h",
    "merchant_fraud_rate_30d",
    "country_risk",
    "is_foreign_country",
    "ip_country_mismatch",
    "mcc_risk",
    "days_since_last_txn",
    "avg_amount_30d",
    "amount_to_avg_ratio",
    "distinct_merchants_24h",
    "distinct_countries_24h",
    "chargeback_count_90d",
    "has_chargeback_history",
)

BOOL_FEATURES = {name for name, dtype in FEATURE_DTYPES.items() if dtype.value == "bool"}
INT_FEATURES = {name for name, dtype in FEATURE_DTYPES.items() if dtype.value == "int"}
FLOAT_FEATURES = {name for name, dtype in FEATURE_DTYPES.items() if dtype.value == "float"}


def check_parity() -> list[str]:
    errors: list[str] = []

    contract_names = list(FEATURE_NAMES)
    dbt_names = list(DBT_FEATURE_COLUMNS)

    if len(contract_names) != len(dbt_names):
        errors.append(f"Feature count mismatch: contract has {len(contract_names)}, dbt has {len(dbt_names)}")

    contract_set = set(contract_names)
    dbt_set = set(dbt_names)

    missing_in_dbt = contract_set - dbt_set
    if missing_in_dbt:
        errors.append(f"Features in contract but missing in dbt: {sorted(missing_in_dbt)}")

    extra_in_dbt = dbt_set - contract_set
    if extra_in_dbt:
        errors.append(f"Features in dbt but not in contract: {sorted(extra_in_dbt)}")

    for i, (c_name, d_name) in enumerate(zip(contract_names, dbt_names, strict=False)):
        if c_name != d_name:
            errors.append(f"Order mismatch at position {i}: contract='{c_name}', dbt='{d_name}'")

    return errors


def assert_parity() -> None:
    errors = check_parity()
    if errors:
        msg = "TRAIN/SERVE PARITY VIOLATION:\n" + "\n".join(f"  - {e}" for e in errors)
        raise AssertionError(msg)


if __name__ == "__main__":
    try:
        assert_parity()
        print(f"Parity check passed: {NUM_FEATURES} features aligned (contract v{CONTRACT_VERSION})")
    except AssertionError as e:
        print(str(e))
        raise SystemExit(1) from e
