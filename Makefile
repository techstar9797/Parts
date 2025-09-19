SHELL := /bin/bash

.PHONY: dev api web ingest eval db crawl install setup

# Environment setup
setup: ## Initial setup - install dependencies and create directories
	mkdir -p data/samples data/out
	cd apps/api && pip install -r requirements.txt
	cd apps/web && npm install

install: setup ## Alias for setup

# Development
dev: ## Run full stack (Postgres + API + Web)
	docker compose -f infra/docker/docker-compose.yml up --build

api: ## Run API locally (requires local Postgres or Docker)
	cd apps/api && uvicorn main:app --reload --host 0.0.0.0 --port 8000

web: ## Run Next.js web app
	cd apps/web && npm run dev

gradio: ## Run Gradio app with LlamaCloud integration
	python3 final_gradio_app.py

fff-gradio: ## Run FFF Scoring Gradio app
	python3 fff_gradio_app.py

dynamic-fff: ## Run Dynamic FFF with FAISS default + user override
	python3 dynamic_fff_gradio.py

enhanced-fff: ## Run Enhanced FFF with Cost & Tariff Analysis
	python3 enhanced_cost_tariff_fff.py

demo-gradio: ## Run demo Gradio app
	python3 demo_gradio_app.py

pipeline-demo: ## Run full pipeline demo
	python3 full_pipeline_demo.py

ui: enhanced-fff ## Alias for Enhanced FFF app

# Data ingestion
crawl: ## Crawl Mouser for datasheets (requires APIFY_TOKEN)
	@if [ -z "$(APIFY_TOKEN)" ]; then echo "Error: APIFY_TOKEN not set"; exit 1; fi
	curl -X POST "http://localhost:8000/ingest/mouser/run" \
		-H "Content-Type: application/json" \
		-d '{"query": "$(QUERY)", "max_pages": $(PAGES:-5), "enrich_with_mouser": true}'

crawl-lm358: ## Quick demo crawl for LM358
	$(MAKE) crawl QUERY="LM358" PAGES=2

crawl-ldo: ## Quick demo crawl for LDO regulators
	$(MAKE) crawl QUERY="QFN LDO 3.3V" PAGES=3

ingest: ## Parse PDFs in data/samples and extract
	cd packages/extraction && python -m extraction.cli --path ../../data/samples --out ../../data/out/records.jsonl

# Database operations
db-reset: ## Reset database (WARNING: deletes all data)
	docker compose -f infra/docker/docker-compose.yml down -v
	docker compose -f infra/docker/docker-compose.yml up -d postgres

db-migrate: ## Run database migrations
	cd packages/storage && python -c "from db import init_db; init_db()"

# Testing and evaluation
test-api: ## Test API endpoints
	curl -X GET "http://localhost:8000/"
	curl -X GET "http://localhost:8000/parts?query=LM358"
	curl -X POST "http://localhost:8000/replace/find-replacement" \
		-H "Content-Type: application/json" \
		-d '{"mpn": "LM358"}'

eval: ## Run evaluation suite
	cd packages/eval && python -m eval.run

# Demo commands
demo: ## Run complete demo workflow
	@echo "Starting PartSync demo..."
	@echo "1. Starting API server..."
	@$(MAKE) api &
	@sleep 5
	@echo "2. Crawling Mouser for LM358..."
	@$(MAKE) crawl-lm358
	@sleep 10
	@echo "3. Testing replacement search..."
	@$(MAKE) test-api
	@echo "4. Starting web interface..."
	@$(MAKE) web

# Utility commands
logs: ## View Docker logs
	docker compose -f infra/docker/docker-compose.yml logs -f

clean: ## Clean up containers and volumes
	docker compose -f infra/docker/docker-compose.yml down -v
	docker system prune -f

help: ## Show this help message
	@echo "PartSync - AI-Powered Part Replacement Engine"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
