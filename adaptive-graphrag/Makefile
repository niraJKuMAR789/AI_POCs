.PHONY: install lint typecheck test check ingest serve ui mcp eval demo docker

install:            ## Create venv and install with dev + UI extras
	uv venv && uv pip install -e ".[dev,ui]"

lint:
	ruff check src tests && ruff format --check src tests

typecheck:
	mypy src

test:               ## Offline test suite (no API key or network needed)
	pytest

check: lint typecheck test

ingest:             ## Ingest the demo corpus
	atlas ingest data/corpus

serve:
	atlas serve --port 8000

ui:
	streamlit run ui/app.py

mcp:                ## MCP server over stdio (for Claude Desktop / Cursor / agents)
	atlas mcp

eval:               ## Benchmark Atlas vs naive RAG; writes reports/eval_report.md
	atlas eval

demo:               ## Keyless end-to-end demo with the offline providers
	ATLAS_PROVIDER=offline ATLAS_DATA_DIR=.atlas-offline atlas ingest data/corpus && \
	ATLAS_PROVIDER=offline ATLAS_DATA_DIR=.atlas-offline atlas ask --trace \
	  "Who is the primary on-call lead for the service whose schema change caused INC-2041?"

docker:
	docker compose up --build
