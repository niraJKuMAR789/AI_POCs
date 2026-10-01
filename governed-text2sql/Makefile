.PHONY: install data check test eval serve ui mcp chat sft finetune-smoke

install:
	uv venv && uv pip install -e ".[dev,ui]"

data:                 ## Generate the synthetic warehouse + NL→SQL splits
	quill init-db

check:
	ruff check src tests finetune && ruff format --check src tests finetune && mypy src && pytest

eval:                 ## Execution accuracy + red-team report → reports/eval_report.md
	quill eval

serve:
	quill serve

ui:
	streamlit run ui/app.py

mcp:
	QUILL_MCP_ROLE=analyst quill mcp

chat:                 ## Agno agent in the terminal (add ARGS="--mcp" to use MCP tools)
	quill chat $(ARGS)

sft:                  ## Build the SFT dataset (same prompts as inference)
	python finetune/prepare.py

finetune-smoke:       ## CPU end-to-end check of the training pipeline on a tiny random model
	uv pip install -e ".[train]" && python finetune/prepare.py && python finetune/make_tiny_model.py && \
	python finetune/train_lora.py --config finetune/configs/smoke.yaml && \
	python finetune/predict_local.py --base runs/tiny-base --adapter runs/smoke/adapter --limit 4 --max-new-tokens 16
