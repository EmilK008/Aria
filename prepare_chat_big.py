"""
prepare_chat_big.py -- build a LARGER, richer chat corpus to make the bot smarter.

Combines three conversation datasets into our single chat format:
  * DailyDialog          -- everyday small talk (dd_train.parquet)
  * SODA (subset)        -- hundreds of thousands of social dialogues
  * Synthetic-Persona-Chat -- "getting to know you" persona conversations

Everything is normalized to a list of utterances per dialogue, then rendered as
  <|user|>...<|eot|><|bot|>...<|eot|>...
We train a bigger BPE (vocab 8000) and save the canonical chat data files.

Run:  python prepare_chat_big.py --max_soda 100000
"""

import argparse
import json
import os
import re
import urllib.request

import pyarrow.parquet as pq
import torch

from bpe import BPETokenizer

USER, BOT, EOT = "<|user|>", "<|bot|>", "<|eot|>"
SPECIALS = [USER, BOT, EOT]
UA = {"User-Agent": "Mozilla/5.0"}


def _download(url, path):
    if os.path.exists(path):
        print(f"  using cached {path}")
        return path
    print(f"  downloading {os.path.basename(path)} ...")
    req = urllib.request.Request(url, headers=UA)
    data = urllib.request.urlopen(req, timeout=300).read()
    with open(path, "wb") as f:
        f.write(data)
    print(f"    {len(data)/1e6:.1f} MB")
    return path


def _parquet_urls(ds, cfg, split):
    url = f"https://huggingface.co/api/datasets/{ds}/parquet/{cfg}/{split}"
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())


def load_dailydialog(path="dd_train.parquet"):
    if not os.path.exists(path):
        _download(_parquet_urls("roskoN/dailydialog", "full", "train")[0], path)
    rows = pq.read_table(path, columns=["utterances"]).column("utterances").to_pylist()
    out = []
    for utts in rows:
        d = [u.strip() for u in utts if u and u.strip()]
        if len(d) >= 2:
            out.append(d)
    return out


def load_soda(max_rows, path="soda_train.parquet"):
    if not os.path.exists(path):
        _download(_parquet_urls("allenai/soda", "default", "train")[0], path)
    rows = pq.read_table(path, columns=["dialogue"]).column("dialogue").to_pylist()
    out = []
    for utts in rows[:max_rows]:
        d = [u.strip() for u in utts if u and u.strip()]
        if len(d) >= 2:
            out.append(d)
    return out


_BRACKET = re.compile(r"\[[^\]]*\]")  # strip "[user 1's name]" placeholders


def load_persona(path="persona_train.parquet"):
    if not os.path.exists(path):
        _download(_parquet_urls("google/Synthetic-Persona-Chat", "default", "train")[0], path)
    rows = pq.read_table(path, columns=["Best Generated Conversation"]) \
        .column("Best Generated Conversation").to_pylist()
    out = []
    for convo in rows:
        if not convo:
            continue
        utts = []
        for line in convo.split("\n"):
            line = re.sub(r"^User\s*\d+\s*:\s*", "", line.strip())
            line = _BRACKET.sub("", line).strip()
            if line:
                utts.append(line)
        if len(utts) >= 2:
            out.append(utts)
    return out


def render(dialogue):
    out = []
    for i, utt in enumerate(dialogue):
        tag = USER if i % 2 == 0 else BOT
        out.append(f"{tag}{utt}{EOT}")
    return "".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_soda", type=int, default=100000)
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--tokenizer", default="bpe_big.json")
    p.add_argument("--out_train", default="chat_big_train.pt")
    p.add_argument("--out_val", default="chat_big_val.pt")
    p.add_argument("--val_frac", type=float, default=0.02)
    args = p.parse_args()

    print("Loading datasets ...")
    dialogues = []
    dd = load_dailydialog();      print(f"  DailyDialog: {len(dd):,}");      dialogues += dd
    pc = load_persona();          print(f"  Persona-Chat: {len(pc):,}");     dialogues += pc
    soda = load_soda(args.max_soda); print(f"  SODA: {len(soda):,}");        dialogues += soda
    print(f"  TOTAL dialogues: {len(dialogues):,}")

    # 1. train BPE on the plain conversation text
    plain = "\n".join(u for d in dialogues for u in d)
    print(f"  training BPE (vocab {args.vocab_size}) on {len(plain)/1e6:.1f}M chars ...")
    tok = BPETokenizer()
    tok.train(plain, vocab_size=args.vocab_size, special_tokens=SPECIALS)
    tok.save(args.tokenizer)
    print(f"  tokenizer -> {args.tokenizer} (vocab {tok.vocab_size})")

    # 2. render + encode everything (cache makes this fast)
    print("  encoding all dialogues ...")
    all_ids = []
    for i, d in enumerate(dialogues):
        all_ids.extend(tok.encode(render(d)))
        if (i + 1) % 50000 == 0:
            print(f"    {i+1:,}/{len(dialogues):,} ({len(all_ids)/1e6:.1f}M tokens so far)")
    data = torch.tensor(all_ids, dtype=torch.long)
    print(f"  total tokens: {len(data):,}")

    # 3. split + save
    # .clone() is important: saving a *slice* of a tensor otherwise writes the
    # entire underlying storage to disk, not just the slice we sliced out.
    n_val = int(len(data) * args.val_frac)
    torch.save(data[:-n_val].clone(), args.out_train)
    torch.save(data[-n_val:].clone(), args.out_val)
    print(f"  saved {len(data)-n_val:,} train / {n_val:,} val tokens "
          f"-> {args.out_train}, {args.out_val}")


if __name__ == "__main__":
    main()
