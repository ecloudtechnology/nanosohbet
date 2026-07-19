"""Prefetch third-party tokenizers into the local HF cache (login node has
internet; compute nodes may not). Benchmark then runs with HF_HUB_OFFLINE=1."""
from transformers import AutoTokenizer

REPOS = [
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "unsloth/gemma-2-9b-it",
    "vngrs-ai/Kumru-2B",
    "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1",
    "TURKCELL/Turkcell-LLM-7b-v1",
    "Trendyol/Trendyol-LLM-7b-chat-v1.8",
    "CohereForAI/aya-expanse-8b",
]

for repo in REPOS:
    try:
        AutoTokenizer.from_pretrained(repo, use_fast=True)
        print(f"ok   {repo}")
    except Exception as e:
        print(f"FAIL {repo}: {type(e).__name__}: {e}")

import tiktoken
for enc in ("cl100k_base", "o200k_base"):
    tiktoken.get_encoding(enc)
    print(f"ok   tiktoken/{enc}")
print("PREFETCH_DONE")
