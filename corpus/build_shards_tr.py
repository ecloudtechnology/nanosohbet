"""Build Turkish pretraining shards in nanochat's exact parquet format.

Streams FineWeb2 (tur_Latn) with buffer shuffling and writes
$NANOCHAT_BASE_DIR/base_data_tr/shard_NNNNN.parquet files that
nanochat/dataset.py can read directly:

  - single 'text' column
  - row_group_size = 1024 documents (reader iterates row groups)
  - ~250M characters per shard, zstd level 3

Usage:
  python corpus/build_shards_tr.py -n 170        # ~42 GB, d20 icin yeterli
  python corpus/build_shards_tr.py -n 170 --start 120   # kaldigi yerden devam
"""
import argparse
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

CHARS_PER_SHARD = 250_000_000
ROW_GROUP_SIZE = 1024
MIN_DOC_CHARS = 200
SHUFFLE_BUFFER = 100_000
SEED = 1337


def out_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
    d = os.path.join(base, "base_data_tr")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-shards", type=int, default=170)
    ap.add_argument("--start", type=int, default=-1,
                    help="ilk yazilacak shard indeksi (-1 = diskteki shardlardan otomatik devam)")
    args = ap.parse_args()

    d = out_dir()
    if args.start < 0:
        args.start = len([f for f in os.listdir(d) if f.endswith(".parquet")])
        print(f"otomatik devam: {args.start}. shard'dan basliyor", flush=True)
    if args.start >= args.num_shards:
        print("SHARDS_COMPLETE", flush=True)
        return
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="tur_Latn", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

    shard_idx = args.start
    docs, chars = [], 0
    t0 = time.time()
    skipped = 0

    for row in ds:
        text = row.get("text", "")
        if not text or len(text) < MIN_DOC_CHARS:
            continue

        # devam modunda: onceki shard'lara ait dokumanlari hizla atla
        target_skip = args.start * CHARS_PER_SHARD
        if skipped < target_skip:
            skipped += len(text)
            continue

        docs.append(text)
        chars += len(text)

        full = chars >= CHARS_PER_SHARD and len(docs) % ROW_GROUP_SIZE == 0
        if full:
            path = os.path.join(d, f"shard_{shard_idx:05d}.parquet")
            tmp = path + ".tmp"
            pq.write_table(
                pa.table({"text": docs}),
                tmp,
                row_group_size=ROW_GROUP_SIZE,
                compression="zstd",
                compression_level=3,
            )
            os.rename(tmp, path)
            el = time.time() - t0
            print(f"shard {shard_idx:05d}: {len(docs)} docs, {chars/1e6:.0f}M chars  ({el:.0f}s)", flush=True)
            shard_idx += 1
            docs, chars = [], 0
            if shard_idx >= args.num_shards:
                break

    print("SHARDS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
