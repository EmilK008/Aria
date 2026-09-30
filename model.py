"""
model.py -- A GPT (decoder-only transformer) written from scratch.

Nothing here uses nn.Transformer or any prebuilt attention. We implement
the actual math ourselves so it's clear how a language model works:

  tokens -> embeddings -> [ self-attention + MLP ] x N -> predict next token

The only "magic" we lean on is PyTorch's autograd (so we don't hand-derive
gradients) and its tensor ops on the GPU.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def _block_repeat_ngrams(logits, idx, n, gen_start=0):
    """Stop the model repeating ITSELF: ban any token that would complete an n-gram
    already produced in the generated span (batch 1). We only look at tokens from
    `gen_start` on, so copying phrases from a retrieved passage is still allowed --
    only re-repeating what it already said is blocked."""
    if n <= 0:
        return
    seq = idx[0].tolist()
    if len(seq) < n:
        return
    prefix = tuple(seq[-(n - 1):])
    for i in range(max(gen_start, 0), len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            logits[0, seq[i + n - 1]] = float("-inf")


def _apply_repetition_penalty(logits, idx, penalty, gen_start=0):
    """Lower the score of tokens the model already GENERATED, so it stops looping.

    Only tokens from `gen_start` on are penalized -- for grounded answers this means
    we don't punish reusing words from the retrieved context (which we WANT copied),
    just the model repeating its own output.
    """
    if penalty == 1.0:
        return
    gen = idx[:, gen_start:]
    if gen.numel() == 0:
        return
    score = torch.gather(logits, 1, gen)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, gen, score)


@dataclass
class GPTConfig:
    vocab_size: int = 256      # number of distinct tokens (set from the data)
    block_size: int = 256      # max context length (how many tokens it can look back on)
    n_layer: int = 6           # number of transformer blocks stacked
    n_head: int = 6            # number of attention heads
    n_embd: int = 384          # embedding / hidden dimension
    dropout: float = 0.1       # regularization
    use_flash: bool = True     # use PyTorch's fused attention kernel (fast path)
    use_rope: bool = False     # rotary position embeddings instead of a learned table
    use_checkpoint: bool = False  # gradient checkpointing: less memory, ~25% slower


def _rotate_half(x):
    """Split the last dim in two and rotate: [a, b] -> [-b, a]."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _apply_rope(q, k, cos, sin):
    """Rotate q and k by their position (RoPE). cos/sin are (1,1,T,head_dim)."""
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    "Causal" = each position can only attend to itself and earlier positions,
    never the future. That's what makes it able to predict the *next* token.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        # One big linear layer produces Query, Key, and Value for every head at once.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        # Projection applied after we recombine the heads.
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout_p = cfg.dropout
        # the fused kernel ("flash attention") computes the same math far more
        # cheaply at long context -- it never materializes the full TxT matrix
        self.flash = cfg.use_flash and hasattr(F, "scaled_dot_product_attention")
        # A lower-triangular matrix of 1s -> our causal mask. Stored as a buffer
        # (moves with the model to GPU, but isn't a trainable parameter).
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )
        # RoPE: precompute rotation cos/sin for every position up to block_size.
        self.rope = cfg.use_rope
        if self.rope:
            head_dim = cfg.n_embd // cfg.n_head
            assert head_dim % 2 == 0, "RoPE needs an even head dimension"
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            pos = torch.arange(cfg.block_size).float()
            freqs = torch.outer(pos, inv_freq)              # (block_size, head_dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)         # (block_size, head_dim)
            self.register_buffer("rope_cos", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("rope_sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Project to q, k, v and split into heads.
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        # reshape (B, T, C) -> (B, n_head, T, head_dim) so each head works independently
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.rope:  # inject position by rotating q and k (no position table needed)
            q, k = _apply_rope(q, k, self.rope_cos[:, :, :T], self.rope_sin[:, :, :T])

        if self.flash:
            # fast path: same computation as below, fused into one efficient kernel
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            # --- attention, computed by hand (what the fast path does under the hood) ---
            # how much each token "attends" to every other: scaled dot product of Q and K
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)   # (B, n_head, T, T)
            # mask out the future: set those scores to -inf so softmax makes them 0
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)        # turn scores into probabilities
            att = self.attn_dropout(att)
            y = att @ v                          # weighted sum of values -> (B, n_head, T, head_dim)

        # recombine heads back into (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """A simple position-wise feed-forward network (the 'thinking' between attentions)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        return self.dropout(self.proj(x))


class Block(nn.Module):
    """One transformer block: attention then MLP, each with a residual connection."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # "x + ..." are residual connections: we add the layer's output back to its
        # input. This keeps gradients flowing and lets the network refine, not replace.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # what each token means
        # RoPE encodes position inside attention, so no learned position table is needed
        self.pos_emb = None if cfg.use_rope else nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # predict next-token logits

        # weight tying: share the embedding and output matrix (a common, helpful trick)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than the model's context window"

        x = self.token_emb(idx)                       # what each token means
        if self.pos_emb is not None:                  # add learned position (unless RoPE)
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            if self.cfg.use_checkpoint and self.training:
                # recompute this block's activations in the backward pass -> saves
                # memory (lets a bigger model fit in VRAM) at ~25% extra compute
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)   # (B, T, vocab_size): a score for every possible next token

        loss = None
        if targets is not None:
            # cross-entropy between predicted next-token and the actual next token
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None,
                 repetition_penalty=1.0, min_new_tokens=0, no_repeat_ngram_size=0):
        """Autoregressively generate tokens, one at a time, feeding each back in.

        If `stop_token` is given (and we're generating a single sequence), we stop
        early as soon as the model emits it -- that's how the chatbot ends its turn.
        `repetition_penalty` > 1 discourages repeating tokens (kills "yes yes yes" loops).
        `min_new_tokens` forbids stopping before that many tokens (avoids 2-word replies).
        """
        self.eval()
        gen_start = idx.size(1)   # tokens before this are the prompt/context, not output
        for step in range(max_new_tokens):
            # crop context to the last block_size tokens (the model can't see further)
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # focus on the last position
            _apply_repetition_penalty(logits, idx, repetition_penalty, gen_start)
            _block_repeat_ngrams(logits, idx, no_repeat_ngram_size, gen_start)
            if stop_token is not None and step < min_new_tokens:
                logits[:, stop_token] = float("-inf")   # not allowed to stop yet
            if top_k is not None:
                # keep only the k most likely tokens (reduces incoherent rambling)
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample from the distribution
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and next_id.item() == stop_token:
                break
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None,
                        repetition_penalty=1.0, min_new_tokens=0, no_repeat_ngram_size=0):
        """Same as generate(), but *yields* each new token id as it's produced.

        This is what lets the API send words to the user one at a time instead of
        making them wait for the whole reply.
        """
        self.eval()
        gen_start = idx.size(1)   # tokens before this are the prompt/context, not output
        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            _apply_repetition_penalty(logits, idx, repetition_penalty, gen_start)
            _block_repeat_ngrams(logits, idx, no_repeat_ngram_size, gen_start)
            if stop_token is not None and step < min_new_tokens:
                logits[:, stop_token] = float("-inf")   # not allowed to stop yet
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            tok_id = int(next_id.item())
            if stop_token is not None and tok_id == stop_token:
                return
            idx = torch.cat([idx, next_id], dim=1)
            yield tok_id
