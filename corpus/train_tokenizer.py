"""Train Turkish byte-level BPE tokenizers at several vocab sizes.

Trains on data/tok_train.txt and writes tokenizers/nanosohbet-<size>.json.
Special tokens follow the nanochat convention so the tokenizer can be dropped
into the training pipeline later.
"""
import os
import time

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

DATA = os.path.expanduser("~/nanosohbet/data/tok_train.txt")
OUT_DIR = os.path.expanduser("~/nanosohbet/tokenizers")
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB_SIZES = [32768, 49152, 65536]

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]


def train_one(vocab_size: int):
    t0 = time.time()
    tok = Tokenizer(models.BPE(byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tok.train([DATA], trainer)
    out = f"{OUT_DIR}/nanosohbet-{vocab_size}.json"
    tok.save(out)
    print(f"saved {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for vs in VOCAB_SIZES:
        print(f"=== training vocab_size={vs} ===", flush=True)
        train_one(vs)
    print("TOKENIZER_TRAINING_COMPLETE", flush=True)
