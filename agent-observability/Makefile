.PHONY: install check demo serve ui eval eval-quick

install:
	uv venv && uv pip install -e ".[dev,ui]"

check:
	ruff check src tests ui examples && ruff format --check src tests ui examples && mypy src && pytest

demo:           ## simulate 12 h of three agent apps with injected incidents; prints project API keys
	argus demo

serve:
	argus serve

ui:
	streamlit run ui/app.py

eval:           ## detection + RCA benchmark: 4 intensities x 3 seeds
	argus eval

eval-quick:
	argus eval --quick --out /tmp/argus-eval.md
