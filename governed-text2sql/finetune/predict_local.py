"""Execution accuracy of a (base + LoRA adapter) model with plain transformers, no server needed.

For larger models, serve with vLLM instead and run `quill eval` against it (see README).

    python finetune/predict_local.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter runs/qwen-lora/adapter
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from quill.config import Settings
from quill.data.questions import HELD_OUT_TEMPLATES
from quill.db.engine import Database
from quill.evals.execution import has_top_level_order, results_match
from quill.pipeline import extract_sql


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, default=Path("data/sft/test.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("reports/finetune_eval.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    db = Database(Settings().db_path, timeout_s=5)

    records = [json.loads(x) for x in args.data.read_text().splitlines() if x.strip()][: args.limit]
    scores: dict[str, list[bool]] = defaultdict(list)
    predictions = []
    for rec in records:
        inputs = tokenizer.apply_chat_template(
            rec["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred, gold = extract_sql(text), extract_sql(rec["completion"][0]["content"])
        try:
            correct = results_match(
                db.query(pred, max_rows=100_000).rows,
                db.query(gold, max_rows=100_000).rows,
                ordered=has_top_level_order(gold),
            )
        except Exception:
            correct = False
        scores["overall"].append(correct)
        scores["held_out" if rec["template"] in HELD_OUT_TEMPLATES else "seen"].append(correct)
        predictions.append({"id": rec["id"], "pred": pred, "gold": gold, "correct": correct})

    summary = {k: round(sum(v) / len(v), 4) for k, v in scores.items() if v}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "n": len(records), "predictions": predictions}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
