.PHONY: install data check eval serve ui mcp demo

install:
	uv venv && uv pip install -e ".[dev,ui]"

data:            ## Regenerate the synthetic registry + labeled claims (deterministic)
	aegis generate

check:
	ruff check src tests ui && ruff format --check src tests ui && mypy src && pytest

eval:            ## rules-only vs rules+agents vs LLM-only → reports/eval_report.md
	aegis eval

serve:
	aegis serve

ui:
	streamlit run ui/app.py

mcp:
	aegis mcp
