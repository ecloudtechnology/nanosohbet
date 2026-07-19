"""Download Turkish (and reference English) text corpora for tokenizer training and evaluation.

Streams FineWeb2 (tur_Latn) and writes plain-text shards:
  data/tok_train.txt      ~4 GB   tokenizer training corpus (Turkish web)
  data/eval_tr_web.txt    ~150 MB held-out Turkish web text
  data/eval_tr_wiki.txt   ~100 MB Turkish Wikipedia (different register)
  data/eval_en_web.txt    ~100 MB English web text (cross-language reference)

Run on a login node (compute nodes may not have internet access).
"""
import os
import sys

from datasets import load_dataset

DATA_DIR = os.path.expanduser("~/nanosohbet/data")
os.makedirs(DATA_DIR, exist_ok=True)

GB = 1024**3
MB = 1024**2

TARGETS = {
    "tok_train.txt": 4 * GB,
    "eval_tr_web.txt": 150 * MB,
    "eval_tr_wiki.txt": 100 * MB,
    "eval_en_web.txt": 100 * MB,
}


def stream_to_file(ds_iter, path, target_bytes, text_key="text", skip_docs=0):
    written = 0
    docs = 0
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds_iter):
            if i < skip_docs:
                continue
            text = row.get(text_key, "")
            if not text or len(text) < 200:
                continue
            f.write(text.replace("\r\n", "\n").strip() + "\n\n")
            written += len(text.encode("utf-8"))
            docs += 1
            if docs % 20000 == 0:
                print(f"  {os.path.basename(path)}: {docs} docs, {written/MB:.0f} MB", flush=True)
            if written >= target_bytes:
                break
    print(f"DONE {os.path.basename(path)}: {docs} docs, {written/MB:.0f} MB", flush=True)
    return docs


def main():
    print("=== FineWeb2 Turkish (tur_Latn) ===", flush=True)
    fw_tr = load_dataset("HuggingFaceFW/fineweb-2", name="tur_Latn", split="train", streaming=True)
    it = iter(fw_tr)
    stream_to_file(it, f"{DATA_DIR}/tok_train.txt", TARGETS["tok_train.txt"])
    # continue the same iterator so eval never overlaps training data
    stream_to_file(it, f"{DATA_DIR}/eval_tr_web.txt", TARGETS["eval_tr_web.txt"])

    print("=== Turkish Wikipedia ===", flush=True)
    wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    stream_to_file(iter(wiki), f"{DATA_DIR}/eval_tr_wiki.txt", TARGETS["eval_tr_wiki.txt"])

    print("=== FineWeb English (reference) ===", flush=True)
    fw_en = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    stream_to_file(iter(fw_en), f"{DATA_DIR}/eval_en_web.txt", TARGETS["eval_en_web.txt"])

    print("ALL_DOWNLOADS_COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
