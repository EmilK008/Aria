"""
data.py -- tokenizer + data loading.

For our first model we use a *character-level* tokenizer: every distinct
character in the training text is one token. It's the simplest possible
tokenizer and great for learning -- the model literally learns language one
letter at a time. (We upgrade to subword/BPE tokens in a later milestone.)
"""

import json
import os

import torch


class CharTokenizer:
    """Maps each unique character <-> an integer id."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        # unknown chars are skipped so the chatbot never crashes on weird input
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["chars"])


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_splits(text, tokenizer, val_frac=0.1):
    """Encode the whole corpus and split into train/val tensors."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(len(data) * val_frac)
    train_data = data[:-n_val] if n_val > 0 else data
    val_data = data[-n_val:] if n_val > 0 else data[:0]
    return train_data, val_data


def get_batch(data, block_size, batch_size, device):
    """Grab a random batch of (context, next-token) pairs.

    For each example we pick a random window of `block_size` tokens as the input
    x, and the same window shifted by one as the target y -- so at every position
    the model is asked to predict the very next character.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    # pin + non_blocking gives a small speedup moving data to the GPU
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
