# Model Card — Fraud Detection Classifier

## Overview

| | |
|---|---|
| **Task** | Binary classification — probability a transaction is fraudulent. |
| **Model** | XGBoost (`XGBClassifier`), benchmarked against a Logistic Regression baseline. |
| **Inputs** | The 34-feature vector defined in [`features/contract.py`](../src/ml_service/features/contract.py) (the train/serve parity contract). |
| **Output** | Calibrated fraud probability → `APPROVE` (<0.4) / `REVIEW` (0.4–0.7) / `DECLINE` (≥0.7). |
| **Serving** | In-memory, atomic hot-swap on `fraud.model.deployed`; rules-only fallback if no model is loaded (degraded mode). |

## Training data

- Offline feature + label marts built by **dbt** from the 9-table platform schema
  (`dbt/models/marts/training/training_dataset.sql`): point-in-time label join, latest-label-wins,
  and a **`sample_weight ≈ 5.0` up-weighting of missed fraud**.
- For local/demo runs, a deterministic **synthetic generator** (`data/generate_synthetic.py`)
  produces transactions/devices/merchants/labels with injected fraud patterns.

## Evaluation

Reported by `training/evaluate.py` on a held-out split, emphasizing **PR-AUC** given heavy class
imbalance: PR-AUC, ROC-AUC, recall, precision, FP-rate, and a **cost-weighted metric**
(missed-fraud ≫ false-positive). Promotion is gated by `training/gate.py`
(min PR-AUC / ROC-AUC / recall, max FP-rate, max cost-per-txn) — a candidate that fails the gate
is never deployed.

## Explainability

Per-prediction contributions via SHAP (`explain/shap_explain.py`), with a fallback to native
gain importances, narrated into plain English for analysts (`explain/narrator.py`).

## Limitations & monitoring

- **Train/serve skew risk:** the rich offline features assume upstream context (velocity,
  history) is supplied at scoring time; a thin `/predict` payload degrades to defaults. The
  parity contract guards *schema*; value provenance is the engine's responsibility.
- **Drift:** feature/prediction drift is monitored with PSI + KS (`drift/`); significant drift
  triggers `fraud.retraining.requested`.
- **Synthetic demo data** is not representative of production fraud; metrics from the demo are
  illustrative only.
