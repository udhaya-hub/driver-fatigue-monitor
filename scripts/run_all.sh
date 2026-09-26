#!/bin/bash
# Quick start script - runs backend, frontend, and tests
# Usage: ./scripts/run_all.sh [test|dev|prod]

set -e

MODE="${1:-dev}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Driver Fatigue Monitor - $MODE mode"
echo "=========================================="

case "$MODE" in
  dev)
    echo "Starting development servers..."
    echo ""
    echo "Terminal 1 (Backend):"
    echo "  cd $PROJECT_ROOT && python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
    echo "Terminal 2 (Frontend):"
    echo "  cd $PROJECT_ROOT && streamlit run frontend/app.py --server.headless true"
    echo ""
    echo "Then open: http://localhost:8501"
    ;;

  test)
    echo "Running test suite..."
    echo ""
    # Check if backend is running
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
      echo "✅ Backend is running"
    else
      echo "⚠️  Backend not detected. Starting temporary backend..."
      python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000 &
      BACKEND_PID=$!
      sleep 5
      trap "kill $BACKEND_PID 2>/dev/null" EXIT
    fi
    echo ""
    echo "Running API tests..."
    python tests/test_api.py --verbose
    echo ""
    echo "Running curl tests..."
    bash tests/test_api.sh
    echo ""
    echo "Running pytest..."
    pytest tests/test_pytest_api.py -v
    ;;

  prod)
    echo "Starting production stack with Docker..."
    docker-compose up -d --build
    echo ""
    echo "Services:"
    echo "  API:       http://localhost:8000"
    echo "  Frontend:  http://localhost:8501"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana:   http://localhost:3000 (admin/admin)"
    ;;

  *)
    echo "Usage: $0 [dev|test|prod]"
    echo "  dev   - Show commands for manual development"
    echo "  test  - Run full test suite"
    echo "  prod  - Start production Docker stack"
    exit 1
    ;;
esac