#!/usr/bin/env bash
set -euo pipefail

echo "=== Fraud Prevention Platform — ml-service Demo ==="
echo ""

echo "1/6 Generating synthetic data..."
python data/generate_synthetic.py --num-transactions 5000 --fraud-rate 0.05
echo ""

echo "2/6 Running dbt (seed + build + test)..."
cd dbt
dbt deps --target duckdb
dbt seed --target duckdb
dbt build --target duckdb
dbt test --target duckdb
cd ..
echo ""

echo "3/6 Checking feature parity..."
python -m ml_service.features.parity
echo ""

echo "4/6 Training model..."
python -m ml_service.training.train
echo ""

echo "5/6 Starting FastAPI server..."
uvicorn ml_service.app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
sleep 3

echo "  Testing /health..."
curl -s http://localhost:8000/health | python -m json.tool

echo "  Testing /predict..."
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "demo-001", "amount": 1500.00, "country": "NG", "new_device": true, "failed_attempts": 5}' \
  | python -m json.tool

echo "  Testing /investigate..."
curl -s -X POST http://localhost:8000/api/v1/investigate \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "demo-001"}' \
  | python -m json.tool

kill $SERVER_PID 2>/dev/null || true
echo ""

echo "6/6 Running AI evaluation suite..."
python -m eval.runner
python -m eval.gate
echo ""

echo "=== Demo complete ==="
