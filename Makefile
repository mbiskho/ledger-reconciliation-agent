.PHONY: help generate reconcile evaluate serve test clean install lint format

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install in dev mode
	pip install -e ".[dev]"

generate: ## Generate synthetic ledgers
	ledger-rec generate --transactions 5000 --break-rate 0.15 --output data/synthetic

generate-small: ## Generate small test ledgers
	ledger-rec generate --transactions 100 --break-rate 0.15 --output data/synthetic

reconcile: ## Run reconciliation agent
	ledger-rec reconcile --data-dir data/synthetic --output results

reconcile-dry: ## Dry run without LLM calls
	ledger-rec reconcile --data-dir data/synthetic --output results --dry-run

evaluate: ## Evaluate agent performance
	ledger-rec evaluate --results results/results.json --ground-truth data/synthetic/ground_truth.json

serve: ## Start API server
	ledger-rec serve --port 8000

query: ## Query ledgers (usage: make query REF=TXN-00000042)
	ledger-rec query --reference "$(REF)"

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t ledger-rec .

docker-up: ## Start API server in Docker
	docker compose -f docker/docker-compose.yml up -d

docker-reconcile: ## Run reconciliation in Docker
	docker compose -f docker/docker-compose.yml --profile batch run --rm reconcile

test: ## Run unit tests
	python -m pytest tests/unit -v

test-cov: ## Run tests with coverage
	python -m pytest tests/unit -v --cov=src/ledger_rec --cov-report=term-missing

clean: ## Clean generated data and results
	rm -rf data/synthetic/* results/*
	rm -f data/audit.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint: ## Lint code
	python -m ruff check src/ tests/

format: ## Format code
	python -m ruff format src/ tests/

typecheck: ## Type check
	python -m mypy src/
