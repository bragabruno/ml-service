#!/usr/bin/env bash
set -euo pipefail

UV="uv"
PYTHON=".venv/bin/python"

echo "=== Fraud Prevention Platform — ml-service Demo ==="
echo ""

echo "1/6 Generating synthetic data..."
$PYTHON data/generate_synthetic.py --num-transactions 5000 --fraud-rate 0.05
echo ""

echo "2/6 Running dbt (seed + build + test)..."
(
  cd dbt
  $UV run dbt deps --target duckdb
  $UV run dbt seed --target duckdb
  $UV run dbt build --target duckdb
  $UV run dbt test --target duckdb
)
echo ""

echo "3/6 Checking feature parity..."
$UV run python -m ml_service.features.parity
echo ""

echo "4/6 Training model..."
$UV run python -m ml_service.training.train
echo ""

echo "5/6 Starting FastAPI server..."
$UV run uvicorn ml_service.app.main:app --host 127.0.0.1 --port 8770 &
SERVER_PID=$!
sleep 4

echo "  Testing /health..."
curl -s http://127.0.0.1:8770/health | python -m json.tool

echo "  Testing /predict..."
curl -s -X POST http://127.0.0.1:8770/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "demo-001", "amount": 1500.00, "country": "NG", "new_device": true, "failed_attempts": 5}' \
  | python -m json.tool

echo "  Testing /investigate..."
curl -s -X POST http://127.0.0.1:8770/api/v1/investigate \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "demo-001", "amount": 1500.00, "country": "NG", "new_device": true, "failed_attempts": 5}' \
  | python -m json.tool

echo "  Testing /feedback..."
curl -s -X POST http://127.0.0.1:8770/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "demo-001", "original_disposition": "REVIEW", "analyst_decision": "DECLINE", "feedback_type": "override", "notes": "Confirmed fraud"}' \
  | python -m json.tool

kill $SERVER_PID 2>/dev/null || true
echo ""

echo "6/6 Running AI evaluation suite..."
$UV run python -m eval.runner
$UV run python -m eval.gate
echo ""

echo "=== Demo complete ==="
