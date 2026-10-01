"""Create a tiny, randomly initialized Llama + BPE tokenizer with a chat template.

Used only to smoke-test the fine-tuning pipeline end to end on CPU, without downloading
a real base model. The resulting model is untrained noise; its accuracy is meaningless.

    python finetune/make_tiny_model.py --out runs/tiny-base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)
SPECIAL = ["<|pad|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/tiny-base"))
    parser.add_argument("--corpus", type=Path, default=Path("data/sft/train.jsonl"))
    args = parser.parse_args()

    texts = []
    for line in args.corpus.read_text().splitlines():
        rec = json.loads(line)
        texts += [m["content"] for m in rec["prompt"] + rec["completion"]]
    bpe = Tokenizer(models.BPE(unk_token="<|pad|>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=SPECIAL))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe, pad_token="<|pad|>", eos_token="<|end|>")
    tokenizer.chat_template = CHAT_TEMPLATE

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size + len(SPECIAL),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"tiny model written to {args.out}")


if __name__ == "__main__":
    main()
