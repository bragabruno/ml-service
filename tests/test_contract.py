from __future__ import annotations

from ml_service.features.contract import (
    CONTRACT_VERSION,
    FEATURE_DTYPES,
    FEATURE_NAMES,
    FEATURES,
    NUM_FEATURES,
    FeatureType,
    export_contract_json,
    load_contract_json,
)
from ml_service.features.parity import assert_parity, check_parity


def test_contract_has_features() -> None:
    assert NUM_FEATURES == 34
    assert len(FEATURE_NAMES) == 34
    assert len(FEATURE_DTYPES) == 34


def test_feature_names_unique() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_feature_types_valid() -> None:
    for f in FEATURES:
        assert isinstance(f.dtype, FeatureType)


def test_contract_json_roundtrip(tmp_path) -> None:
    out = export_contract_json(tmp_path / "contract.json")
    data = load_contract_json(out)
    assert data["contract_version"] == CONTRACT_VERSION
    assert data["num_features"] == NUM_FEATURES
    assert len(data["features"]) == NUM_FEATURES


def test_parity_passes() -> None:
    errors = check_parity()
    assert errors == [], f"Parity errors: {errors}"


def test_parity_assert() -> None:
    assert_parity()
