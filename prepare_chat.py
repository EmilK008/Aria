"""
prepare_chat.py -- turn raw DailyDialog conversations into training-ready tokens.

Steps:
  1. read the dialogues (list of utterances, speakers alternate each turn)
  2. render each into our chat format with special markers:
       <|user|>hi how are you<|eot|><|bot|>great, you?<|eot|> ...
  3. train our BPE tokenizer on the natural-language text
  4. encode the whole formatted corpus into one long stream of token ids
  5. save train/val token tensors + the tokenizer

The <|eot|> ("end of turn") marker is what teaches the model to STOP after a
reply instead of rambling -- at chat time we generate until it emits <|eot|>.
"""

import argparse

import pyarrow.parquet as pq
import torch

from bpe import BPETokenizer

USER, BOT, EOT = "<|user|>", "<|bot|>", "<|eot|>"
SPECIALS = [USER, BOT, EOT]


def load_dialogues(path):
    table = pq.read_table(path)
    rows = table.column("utterances").to_pylist()
    dialogues = []
    for utts in rows:
        cleaned = [u.strip() for u in utts if u and u.strip()]
        if len(cleaned) >= 2:
            dialogues.append(cleaned)
    return dialogues


def render(dialogue):
    """Speakers alternate: even turns = user, odd turns = bot."""
    out = []
    for i, utt in enumerate(dialogue):
        tag = USER if i % 2 == 0 else BOT
        out.append(f"{tag}{utt}{EOT}")
    return "".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="dd_train.parquet")
    p.add_argument("--vocab_size", type=int, default=4000)
    p.add_argument("--tokenizer", default="bpe_tokenizer.json")
    p.add_argument("--out_train", default="chat_train.pt")
    p.add_argument("--out_val", default="chat_val.pt")
    p.add_argument("--val_frac", type=float, default=0.05)
    args = p.parse_args()

    print(f"Loading dialogues from {args.parquet} ...")
    dialogues = load_dialogues(args.parquet)
    print(f"  {len(dialogues):,} dialogues")

    # 1. train BPE on the plain conversation text (no special markers in training text)
    plain_text = "\n".join(u for d in dialogues for u in d)
    print(f"  training BPE (vocab {args.vocab_size}) on {len(plain_text):,} chars ...")
    tok = BPETokenizer()
    tok.train(plain_text, vocab_size=args.vocab_size, special_tokens=SPECIALS)
    tok.save(args.tokenizer)
    print(f"  tokenizer -> {args.tokenizer} (vocab {tok.vocab_size})")

    # 2. render + encode every dialogue into one long token stream
    print("  encoding formatted dialogues ...")
    all_ids = []
    for d in dialogues:
        all_ids.extend(tok.encode(render(d)))
    data = torch.tensor(all_ids, dtype=torch.long)
    print(f"  total tokens: {len(data):,}")

    # show how much BPE compresses vs char-level
    n_chars = sum(len(render(d)) for d in dialogues)
    print(f"  compression: {n_chars/len(data):.2f} chars/token")

    # 3. train/val split
    n_val = int(len(data) * args.val_frac)
    train_data, val_data = data[:-n_val], data[-n_val:]
    torch.save(train_data, args.out_train)
    torch.save(val_data, args.out_val)
    print(f"  saved {len(train_data):,} train / {len(val_data):,} val tokens")

    # sanity check: encode->decode round trip
    sample = render(dialogues[0])[:200]
    rt = tok.decode(tok.encode(sample))
    print("\n  round-trip OK:", rt[:120].replace("\n", " "))


if __name__ == "__main__":
    main()
