"""Turkce prompt'larla taban modelden ornek uretim.

Kullanim:
    python -m scripts.sample_tr --model-tag d20 --max-tokens 60
"""
import argparse

import torch

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

PROMPTS = [
    "Türkiye'nin başkenti",
    "İstanbul Boğazı'nın iki yakasını birbirine bağlayan",
    "Türk mutfağının en sevilen yemeklerinden biri olan",
    "Cumhuriyet, 29 Ekim 1923 tarihinde",
    "Yapay zekâ teknolojileri sayesinde",
    "Karadeniz Bölgesi'nde en çok yetiştirilen ürün",
    "Bir zamanlar küçük bir kasabada yaşayan yaşlı bir adam",
    "Eğitim sisteminin en önemli amacı",
    "Diplomasi, ülkeler arasındaki ilişkilerin",
    "Dün akşam hava çok soğuktu, bu yüzden",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", type=str, default="d20")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    ddp, rank, local_rank, world_size, device = compute_init()
    model, tokenizer, meta = load_model("base", device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    bos = tokenizer.get_bos_token_id()

    for prompt in PROMPTS:
        tokens = [bos] + tokenizer.encode(prompt)
        sample, _ = engine.generate_batch(
            tokens, num_samples=1, max_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        text = tokenizer.decode(sample[0])
        print("=" * 70)
        print(text)
    print("SAMPLING_DONE")


if __name__ == "__main__":
    main()
