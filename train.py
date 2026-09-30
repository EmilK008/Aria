"""
train.py -- train the GPT on a text corpus.

Run:  python train.py --data data_shakespeare.txt --steps 3000

What happens each step:
  1. grab a random batch of (context, next-char) pairs
  2. ask the model to predict the next char at every position
  3. measure how wrong it was (cross-entropy loss)
  4. backprop + AdamW nudges every weight to be a little less wrong
Repeat a few thousand times and it learns the structure of the text.
"""

import argparse
import os
import time

import torch

from data import CharTokenizer, load_corpus, make_splits, get_batch
from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_shakespeare.txt")
    p.add_argument("--out", default="ckpt.pt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device, iters=50):
    """Average loss over a few batches of train and val (a cleaner signal than one batch)."""
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, 32, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    torch.manual_seed(1337)

    print(f"Loading corpus from {args.data} ...")
    text = load_corpus(args.data)
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(args.tokenizer)
    train_data, val_data = make_splits(text, tokenizer)
    print(f"  {len(text):,} chars | vocab {tokenizer.vocab_size} | "
          f"{len(train_data):,} train / {len(val_data):,} val tokens")

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(args.device)
    print(f"  model: {model.num_params()/1e6:.2f}M parameters on {args.device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    # mixed precision: use fast 16-bit math on the GPU where safe (big speedup)
    use_amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Training for {args.steps} steps ...")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, cfg.block_size, args.batch_size, args.device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        scaler.step(optimizer)
        scaler.update()

        if step % args.eval_every == 0 or step == 1:
            stats = estimate_loss(model, train_data, val_data, cfg, args.device)
            dt = time.time() - t0
            print(f"  step {step:>5} | train {stats['train']:.3f} | "
                  f"val {stats['val']:.3f} | {dt:.0f}s elapsed")

    # save everything needed to reload and chat later
    torch.save(
        {"model": model.state_dict(), "config": cfg.__dict__, "tokenizer": args.tokenizer},
        args.out,
    )
    print(f"Saved checkpoint -> {args.out}")

    # quick sample so we can see what it learned
    print("\n--- sample ---")
    start = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    out = model.generate(start, max_new_tokens=400, temperature=0.8, top_k=40)[0]
    print(tokenizer.decode(out.tolist()))


if __name__ == "__main__":
    main()
