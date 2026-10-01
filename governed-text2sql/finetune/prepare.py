"""Build the SFT dataset in TRL's conversational prompt/completion format.

Training prompts are produced by the *same* code path as inference (role-filtered linked
schema + retrieved few-shot examples), so the fine-tuned model sees exactly the prompt
distribution it will be served with. Few-shot examples never include the gold query itself.

    python finetune/prepare.py --out data/sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quill.config import Settings
from quill.data.questions import Example, load_split
from quill.pipeline import Text2SQL
from quill.prompts import sql_messages


def to_record(pipeline: Text2SQL, ex: Example, few_shot: int) -> dict:
    policy = pipeline.policy("analyst")
    _, ddl, metrics = pipeline.visible_schema(ex.question, policy)
    shots = [e for e in pipeline.few_shot(ex.question, policy) if e.sql != ex.sql][:few_shot]
    prompt = sql_messages(ex.question, dialect=pipeline.layer.dialect, schema_ddl=ddl, metrics=metrics, examples=shots)
    return {
        "id": ex.id,
        "template": ex.template,
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": f"```sql\n{ex.sql}\n```"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--out", type=Path, default=Path("data/sft"))
    parser.add_argument("--few-shot", type=int, default=3)
    args = parser.parse_args()

    settings = Settings(provider="offline", examples_path=args.splits / "train.jsonl")
    pipeline = Text2SQL.from_settings(settings)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        rows = [to_record(pipeline, ex, args.few_shot) for ex in load_split(args.splits / f"{split}.jsonl")]
        with (args.out / f"{split}.jsonl").open("w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"{split}: {len(rows)} examples")


if __name__ == "__main__":
    main()
