"""
bpe.py -- a Byte-Pair Encoding tokenizer, written from scratch.

Why BPE? A char-level tokenizer makes every letter a token, so sentences become
very long sequences. BPE instead learns common chunks ("the", "ing", " you")
and gives each its own token. Fewer tokens per sentence => the model sees more
context and learns faster. This is (a simplified version of) what GPT-2 uses.

How training works:
  1. start with raw bytes (256 possible values) as the base vocabulary
  2. count which adjacent pair of tokens is most common in the corpus
  3. merge that pair into a single new token
  4. repeat until we reach the target vocab size
Each merge is recorded so we can apply the exact same merges at encode time.

We also support *special tokens* (<|user|>, <|bot|>, <|eot|>) which are never
split -- they're structural markers for the chat format.
"""

import json
import re
from collections import Counter

# GPT-2-style pre-tokenization: split text into words/punctuation/whitespace runs
# *before* BPE, so merges never cross these natural boundaries (keeps tokens sane).
# ASCII-oriented, which is fine for our English chat data and avoids extra deps.
_SPLIT_PAT = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (id_a, id_b) -> new_id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.special_tokens = {}  # str -> id
        self._special_inv = {}    # id -> str
        self._special_pat = None
        self._cache = {}          # memoize chunk -> ids (words repeat a LOT)

    # ---------- training ----------
    def train(self, text, vocab_size, special_tokens=None):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # pre-tokenize into chunks, then work on *unique* chunks weighted by count
        # (massively faster than scanning the whole corpus every merge)
        chunk_counts = Counter(_SPLIT_PAT.findall(text))
        words = [
            (list(chunk.encode("utf-8")), count)
            for chunk, count in chunk_counts.items()
        ]

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for m in range(num_merges):
            # count every adjacent pair across all words, weighted by word frequency
            pair_counts = Counter()
            for symbols, count in words:
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += count
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + m
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            # apply this merge inside every word
            words = [(_merge_symbols(sym, best, new_id), c) for sym, c in words]

        self._register_specials(special_tokens or [])

    def _register_specials(self, names):
        base = 256 + len(self.merges)
        self.special_tokens = {name: base + i for i, name in enumerate(names)}
        self._special_inv = {i: name for name, i in self.special_tokens.items()}
        if names:
            self._special_pat = re.compile(
                "(" + "|".join(re.escape(n) for n in names) + ")"
            )

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    # ---------- encoding ----------
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        # greedily apply merges in the order they were learned (lowest new_id first)
        while len(ids) >= 2:
            # find the adjacent pair with the earliest-learned merge
            pair = min(
                zip(ids, ids[1:]),
                key=lambda p: self.merges.get(p, float("inf")),
            )
            if pair not in self.merges:
                break
            ids = _merge_symbols(ids, pair, self.merges[pair])
        return ids

    def _encode_ordinary(self, text):
        ids = []
        for chunk in _SPLIT_PAT.findall(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk.encode("utf-8"))
                self._cache[chunk] = cached
            ids.extend(cached)
        return ids

    def encode(self, text):
        """Encode text, treating any special-token strings as atomic tokens."""
        if not self._special_pat:
            return self._encode_ordinary(text)
        ids = []
        for piece in self._special_pat.split(text):
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])
            elif piece:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def decode(self, ids):
        parts = []
        for i in ids:
            i = int(i)
            if i in self._special_inv:
                parts.append(self._special_inv[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---------- persistence ----------
    def save(self, path):
        data = {
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.merges = {(a, b): nid for a, b, nid in data["merges"]}
        # rebuild vocab bytes in merge order
        for (a, b), nid in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            tok.vocab[nid] = tok.vocab[a] + tok.vocab[b]
        names = sorted(data["special_tokens"], key=lambda n: data["special_tokens"][n])
        tok._register_specials(names)
        return tok


def _merge_symbols(symbols, pair, new_id):
    """Replace every occurrence of `pair` in the list with `new_id`."""
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out
