.PHONY: help install test lint format clean docker-build docker-up docker-down

help:
	@echo "AI Market Intelligence - Available commands:"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run tests with coverage"
	@echo "  lint          - Run linters"
	@echo "  format        - Format code with black"
	@echo "  clean         - Clean build artifacts"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-up     - Start services with docker-compose"
	@echo "  docker-down   - Stop services"
	@echo "  run-api       - Run API server locally"
	@echo "  run-mlflow    - Run MLflow server"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/ *.egg-info

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-mlflow:
	mlflow server --host 0.0.0.0 --port 5000

train:
	python scripts/train_models.py --symbol AAPL --backtest

train-ensemble:
	python scripts/train_models.py --symbol AAPL --ensemble --backtest

demo:
	python scripts/quick_start.py

dashboard:
	python scripts/terminal_dashboard.py

setup-dev:
	@echo "Setting up development environment..."
	@cp .env.example .env
	@echo "Created .env file - please fill in your API keys"
	@mkdir -p data/{raw,processed,cache,feedback} logs mlruns scripts
	@echo "Created data directories"
	@pip install -r requirements.txt
	@echo "Installed dependencies"
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "1. Edit .env and add your API keys"
	@echo "2. Run 'make train' to train models"
	@echo "3. Run 'make docker-up' to start services"
	@echo "4. Run 'make run-api' to start the API server"
