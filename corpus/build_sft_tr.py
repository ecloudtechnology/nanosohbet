"""Erk Nano icin Turkce SFT veri karisimi olusturur.

Karisim, yapilandirilabilir Turkce talimat/konusma kaynaklarindan +
kendi kimlik verimizden olusur. Kaynaklar `--sources` ile verilir
(HuggingFace dataset repo id'leri) ya da SOURCES ortam degiskeninden
okunur; boylece reçete herhangi bir Turkce SFT kaynagiyla calisir.

Cikti: her satir bir konusma —
  {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}

Kimlik verisi orantisiz agirlikli tekrarlanir (identity-repeat) ki model
kendini tutarli tanitsin.

Kullanim:
  python corpus/build_sft_tr.py --model-name "Erk Nano" \
      --sources <hf_repo_1> <hf_repo_2> ... --cap 200000 \
      --identity-repeat 15 --out data/sft_tr.jsonl
"""
import argparse
import json
import os
import random
import subprocess
import sys

MIN_OUT_CHARS = 20
MAX_OUT_CHARS = 8000


def clean(text):
    return (text or "").strip()


def to_conv(row):
    """Kaynak satirini nanochat konusma formatina cevirir.

    Desteklenen semalar otomatik algilanir:
      - {"messages": [...]}                 (rol tabanli konusma)
      - {"system","user","assistant"}       (uc kolon)
      - {"Input","Output"} / {"inputs","targets"} / {"prompt","response"}
    """
    if isinstance(row.get("messages"), list):
        turns, sys_txt = [], ""
        for m in row["messages"]:
            role, content = m.get("role"), clean(m.get("content"))
            if role == "system":
                sys_txt = content
            elif role in ("user", "assistant"):
                turns.append({"role": role, "content": content})
        if len(turns) < 2 or turns[0]["role"] != "user":
            return None
        if sys_txt:
            turns[0]["content"] = f"{sys_txt}\n\n{turns[0]['content']}"
        if len(clean(turns[1]["content"])) < MIN_OUT_CHARS:
            return None
        return {"messages": turns}

    for uk, ak in (("system", "assistant"),):
        if row.get(uk) is not None and row.get("assistant") is not None:
            msgs = []
            if clean(row.get("system")):
                msgs.append({"role": "system", "content": row["system"]})
            msgs.append({"role": "user", "content": row.get("user", "")})
            msgs.append({"role": "assistant", "content": row.get("assistant", "")})
            return to_conv({"messages": msgs})

    for ik, ok in (("Input", "Output"), ("inputs", "targets"),
                   ("prompt", "response"), ("instruction", "output")):
        if row.get(ik) is not None and row.get(ok) is not None:
            inp, out = clean(row[ik]), clean(row[ok])
            if len(inp) < 3 or not (MIN_OUT_CHARS <= len(out) <= MAX_OUT_CHARS):
                return None
            return {"messages": [{"role": "user", "content": inp},
                                 {"role": "assistant", "content": out}]}
    return None


def load_source(repo, cap):
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=20_000)
    rows = []
    for r in ds:
        c = to_conv(r)
        if c:
            rows.append(c)
        if len(rows) >= cap:
            break
    return rows


def load_identity(model_name, repeat):
    tmp = "/tmp/erk_identity.jsonl"
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(here, "identity_tr.py"),
                    "--model-name", model_name, "--out", tmp], check=True)
    base = [json.loads(l) for l in open(tmp, encoding="utf-8")]
    return base * repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Erk Nano")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="HuggingFace Turkce SFT dataset repo id'leri")
    ap.add_argument("--cap", type=int, default=200_000, help="kaynak basina ust sinir")
    ap.add_argument("--identity-repeat", type=int, default=15)
    ap.add_argument("--out", default="sft_tr.jsonl")
    args = ap.parse_args()

    sources = args.sources or [s for s in os.environ.get("SOURCES", "").split(",") if s]
    rng = random.Random(42)
    rows = []
    for repo in sources:
        print(f"kaynak yukleniyor: {repo}", flush=True)
        got = load_source(repo, args.cap)
        rows += got
        print(f"  {len(got)} konusma", flush=True)

    ident = load_identity(args.model_name, args.identity_repeat)
    rows += ident
    print(f"kimlik: {len(ident)} konusma ({args.identity_repeat}x)", flush=True)

    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nTOPLAM {len(rows)} konusma yazildi: {args.out}", flush=True)
    print("SFT_HAZIR", flush=True)


if __name__ == "__main__":
    main()
