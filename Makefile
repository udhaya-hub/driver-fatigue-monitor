# Driver Fatigue Monitor - Development Commands
.PHONY: help install test test-api test-integration lint format clean up down logs

# Default target
help:
	@echo "Driver Fatigue Monitor - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install all dependencies"
	@echo "  make install-dev  Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests (pytest)"
	@echo "  make test-api     Run API tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-curl    Run curl-based API tests"
	@echo ""
	@echo "Development:"
	@echo "  make lint         Run linting (ruff, mypy)"
	@echo "  make format       Format code (black, isort)"
	@echo "  make typecheck    Run mypy type checking"
	@echo ""
	@echo "Docker:"
	@echo "  make up           Start all services (docker-compose)"
	@echo "  make down         Stop all services"
	@echo "  make logs         View service logs"
	@echo "  make build        Build Docker images"
	@echo ""
	@echo "Services:"
	@echo "  make run-api      Start FastAPI backend"
	@echo "  make run-frontend Start Streamlit frontend"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --tb=short

test-api:
	pytest tests/test_pytest_api.py -v -m "api"

test-integration:
	pytest tests/test_pytest_api.py -v -m "integration"

test-slow:
	pytest tests/test_pytest_api.py -v -m "slow"

test-curl:
	bash tests/test_api.sh

test-all: test test-curl

# Linting & Formatting
lint:
	ruff check .
	mypy .

format:
	black .
	isort .

typecheck:
	mypy .

# Docker
up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

build:
	docker-compose build

# Services
run-api:
	python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	streamlit run frontend/app.py --server.headless true

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info/