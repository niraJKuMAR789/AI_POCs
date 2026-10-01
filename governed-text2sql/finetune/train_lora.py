"""LoRA / QLoRA supervised fine-tuning for text-to-SQL with TRL.

Loss is computed on the completion (the SQL) only, not on the long schema prompt.

    python finetune/train_lora.py --config finetune/configs/qwen2.5-coder-1.5b.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    cfg = load_config(parser.parse_args().config)

    data = load_dataset("json", data_files={"train": cfg["train_file"], "eval": cfg["eval_file"]})
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"dtype": cfg.get("dtype", "auto")}
    quantization_config = None
    if cfg.get("qlora"):
        import torch
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **model_kwargs)

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )
    train = cfg["training"]
    args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=train.get("epochs", 2),
        max_steps=train.get("max_steps", -1),
        per_device_train_batch_size=train["batch_size"],
        per_device_eval_batch_size=train["batch_size"],
        gradient_accumulation_steps=train.get("grad_accum", 1),
        learning_rate=train["learning_rate"],
        lr_scheduler_type=train.get("scheduler", "cosine"),
        warmup_steps=train.get("warmup", 0.05),  # transformers>=5: a float in [0, 1) is a ratio
        max_length=train["max_length"],
        completion_only_loss=True,
        gradient_checkpointing=train.get("gradient_checkpointing", False),
        bf16=train.get("bf16", False),
        logging_steps=train.get("logging_steps", 10),
        eval_strategy="steps" if train.get("eval_steps") else "no",
        eval_steps=train.get("eval_steps"),
        save_strategy="no",
        report_to=train.get("report_to", "none"),
        use_cpu=train.get("use_cpu", False),
        seed=train.get("seed", 42),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    started = time.time()
    result = trainer.train()
    adapter_dir = Path(cfg["output_dir"]) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    meta = {
        "base_model": cfg["base_model"],
        "train_examples": len(data["train"]),
        "train_loss": result.training_loss,
        "runtime_s": round(time.time() - started, 1),
        "trainable_params": trainable,
        "total_params": total,
        "lora": lora,
    }
    if train.get("eval_steps"):
        meta["eval"] = trainer.evaluate()
    (Path(cfg["output_dir"]) / "training_summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
