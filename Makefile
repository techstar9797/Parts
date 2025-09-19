SHELL := /bin/bash

.PHONY: dev api web ingest eval db

dev: ## Run Postgres and API
	docker compose -f infra/docker/docker-compose.yml up --build

api: ## Run API locally (requires local Postgres or Docker)
	cd apps/api && uvicorn main:app --reload --host 0.0.0.0 --port 8000

web: ## (placeholder) Web dev server
	cd apps/web && echo "TODO: install Next.js & run dev"

ingest: ## Parse PDFs in data/samples and extract
	cd packages/extraction && python -m extraction.cli --path ../../data/samples --out ../../data/out/records.jsonl

eval:
	cd packages/eval && python -m eval.run
