"""Yerel Turkce SFT konusma dosyasini (JSONL) nanochat Task olarak okur.

Her satir: {"messages": [{"role":"user","content":...},
                          {"role":"assistant","content":...}, ...]}
Rol sirasi: (opsiyonel system) + user/assistant/user/assistant...

Dosya yolu TURKCE_SFT_PATH ortam degiskeninden okunur.
"""
import json
import os

from tasks.common import Task

DEFAULT_PATH = os.environ.get("TURKCE_SFT_PATH", "/ari/users/ytanis/erk/sft_final.jsonl")


class TurkceSFT(Task):
    def __init__(self, split="train", path=None, val_frac=0.01, **kwargs):
        super().__init__(**kwargs)
        path = path or DEFAULT_PATH
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = obj.get("messages")
                if self._valid(msgs):
                    rows.append(msgs)
        # deterministik train/val ayrimi
        n_val = max(1, int(len(rows) * val_frac))
        if split == "train":
            self.rows = rows[:-n_val]
        elif split == "test":
            self.rows = rows[-n_val:]
        else:
            raise ValueError("split train|test olmali")
        self.length = len(self.rows)

    @staticmethod
    def _valid(msgs):
        if not isinstance(msgs, list) or len(msgs) < 2:
            return False
        rest = msgs[1:] if msgs[0].get("role") == "system" else msgs
        if len(rest) < 2:
            return False
        for i, m in enumerate(rest):
            exp = "user" if i % 2 == 0 else "assistant"
            if m.get("role") != exp or not isinstance(m.get("content"), str):
                return False
            if not m["content"].strip():
                return False
        return True

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.rows[index]}
