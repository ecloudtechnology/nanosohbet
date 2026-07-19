"""Comprehensive tokenizer efficiency benchmark on Turkish (and English) text.

Compares nanosohbet tokenizers against widely used open/closed-model tokenizers
on held-out corpora. Metrics per (tokenizer, corpus):

  fertility     tokens per whitespace-delimited word (lower is better)
  bytes/token   UTF-8 bytes encoded per token (higher is better)
  cost x        token count relative to nanosohbet-65536 on the same corpus

Writes results/tokenizer_benchmark.json and a markdown table to
results/tokenizer_benchmark.md.
"""
import json
import os
import unicodedata

EVAL_FILES = {
    "tr_web": os.path.expanduser("~/nanosohbet/data/eval_tr_web.txt"),
    "tr_wiki": os.path.expanduser("~/nanosohbet/data/eval_tr_wiki.txt"),
    "en_web": os.path.expanduser("~/nanosohbet/data/eval_en_web.txt"),
}
MAX_BYTES = 50 * 1024 * 1024  # cap per corpus for runtime
OUT_DIR = os.path.expanduser("~/nanosohbet/results")
TOK_DIR = os.path.expanduser("~/nanosohbet/tokenizers")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- tokenizers

def load_candidates():
    """Return list of (name, encode_fn, vocab_size). Skips anything unavailable."""
    cands = []

    # ours
    from tokenizers import Tokenizer
    for vs in (32768, 49152, 65536):
        path = f"{TOK_DIR}/nanosohbet-{vs}.json"
        if os.path.exists(path):
            t = Tokenizer.from_file(path)
            cands.append((f"erk-{vs//1024}k", lambda s, t=t: t.encode(s).ids, t.get_vocab_size()))

    # OpenAI (tiktoken)
    try:
        import tiktoken
        for enc_name, label in [("cl100k_base", "GPT-4 (cl100k)"), ("o200k_base", "GPT-4o (o200k)")]:
            enc = tiktoken.get_encoding(enc_name)
            cands.append((label, lambda s, e=enc: e.encode(s, disallowed_special=()), e_n(enc)))
    except Exception as e:
        print(f"skip tiktoken: {e}")

    # HuggingFace-hosted tokenizers
    hf_models = [
        ("Llama-3.1", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
        ("Qwen2.5", "Qwen/Qwen2.5-7B-Instruct"),
        ("Mistral-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("Gemma-2", "unsloth/gemma-2-9b-it"),
        ("Kumru-2B (TR)", "vngrs-ai/Kumru-2B"),
        ("cosmosGPT (TR)", "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"),
        ("Turkcell-7b (TR)", "TURKCELL/Turkcell-LLM-7b-v1"),
        ("Trendyol-7b (TR)", "Trendyol/Trendyol-LLM-7b-chat-v1.8"),
        ("aya-expanse", "CohereForAI/aya-expanse-8b"),
    ]
    from transformers import AutoTokenizer
    for label, repo in hf_models:
        try:
            t = AutoTokenizer.from_pretrained(repo, use_fast=True, trust_remote_code=False)
            cands.append((label, lambda s, t=t: t.encode(s, add_special_tokens=False), len(t)))
        except Exception as e:
            print(f"skip {label} ({repo}): {type(e).__name__}")
    return cands


def e_n(enc):
    try:
        return enc.n_vocab
    except Exception:
        return -1

# ---------------------------------------------------------------- metrics

def read_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(MAX_BYTES)
    # cut at last full line to avoid a torn multibyte char artifact
    text = text[: text.rfind("\n")]
    return text


def corpus_stats(text):
    return {
        "bytes": len(text.encode("utf-8")),
        "chars": len(text),
        "words": len(text.split()),
    }


def main():
    corpora = {}
    for name, path in EVAL_FILES.items():
        if not os.path.exists(path):
            print(f"missing corpus {name}: {path}")
            continue
        text = read_corpus(path)
        corpora[name] = (text, corpus_stats(text))
        print(f"corpus {name}: {corpora[name][1]}")

    results = {}
    for label, encode, vocab in load_candidates():
        results[label] = {"vocab_size": vocab, "corpora": {}}
        for cname, (text, stats) in corpora.items():
            # chunk to keep memory bounded
            n_tokens = 0
            CH = 2 * 1024 * 1024
            for i in range(0, len(text), CH):
                n_tokens += len(encode(text[i : i + CH]))
            r = {
                "tokens": n_tokens,
                "fertility": round(n_tokens / stats["words"], 4),
                "bytes_per_token": round(stats["bytes"] / n_tokens, 4),
                "chars_per_token": round(stats["chars"] / n_tokens, 4),
            }
            results[label]["corpora"][cname] = r
            print(f"{label:22s} {cname:8s} fertility={r['fertility']:.3f} bytes/tok={r['bytes_per_token']:.2f}")

    # relative cost vs our 65k tokenizer
    ref = "erk-64k"
    if ref in results:
        for label in results:
            for cname in results[label]["corpora"]:
                ref_tokens = results[ref]["corpora"][cname]["tokens"]
                results[label]["corpora"][cname]["cost_x"] = round(
                    results[label]["corpora"][cname]["tokens"] / ref_tokens, 3
                )

    with open(f"{OUT_DIR}/tokenizer_benchmark.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # markdown table (tr_web ordered by fertility)
    lines = [
        "| Tokenizer | Vocab | TR-web fertility | TR-wiki fertility | EN-web fertility | TR-web bytes/tok | TR-web cost× |",
        "|---|---|---|---|---|---|---|",
    ]
    order = sorted(results, key=lambda l: results[l]["corpora"].get("tr_web", {}).get("fertility", 9e9))
    for label in order:
        c = results[label]["corpora"]
        g = lambda k, f: c.get(k, {}).get(f, "—")
        lines.append(
            f"| {label} | {results[label]['vocab_size']:,} | {g('tr_web','fertility')} | "
            f"{g('tr_wiki','fertility')} | {g('en_web','fertility')} | "
            f"{g('tr_web','bytes_per_token')} | {g('tr_web','cost_x')} |"
        )
    with open(f"{OUT_DIR}/tokenizer_benchmark.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("BENCHMARK_COMPLETE")


if __name__ == "__main__":
    main()
