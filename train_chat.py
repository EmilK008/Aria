"""
train_chat.py -- train the GPT on the chat-formatted DailyDialog tokens.

Same network and learning loop as train.py, but:
  - it reads the pre-tokenized BPE stream (chat_train.pt / chat_val.pt)
  - it records the BPE tokenizer in the checkpoint so chat.py knows how to talk

Run:  python train_chat.py --steps 5000
"""

import argparse
import math
import os
import sys
import time

import torch

try:  # sample text may contain accents/IPA; don't let the Windows console crash on it
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bpe import BPETokenizer
from data import get_batch
from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="chat_train.pt")
    p.add_argument("--val", default="chat_val.pt")
    p.add_argument("--tokenizer", default="bpe_tokenizer.json")
    p.add_argument("--out", default="ckpt_chat.pt")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--use_rope", action="store_true",
                   help="use rotary position embeddings (better for long context)")
    p.add_argument("--checkpoint", action="store_true",
                   help="gradient checkpointing: fit a bigger model in VRAM (~25% slower)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="accumulate this many micro-batches per optimizer step "
                        "(effective batch = batch_size * grad_accum)")
    p.add_argument("--warmup", type=int, default=200, help="LR warmup steps")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000,
                   help="save a resumable checkpoint every N steps")
    p.add_argument("--resume", action="store_true",
                   help="resume from --out if it contains training state")
    p.add_argument("--init_from", default=None,
                   help="initialize weights from this checkpoint (fine-tune / extend context). "
                        "RoPE weights are position-agnostic, so a model trained at one block "
                        "size can continue at a larger one.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def lr_at(step, total, base_lr, warmup):
    """Linear warmup then cosine decay down to 10% of base_lr."""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 * base_lr + 0.5 * (base_lr - 0.1 * base_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device, iters=50):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, 8, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    torch.manual_seed(1337)

    tok = BPETokenizer.load(args.tokenizer)
    train_data = torch.load(args.train)
    val_data = torch.load(args.val)
    print(f"Loaded {len(train_data):,} train / {len(val_data):,} val tokens | "
          f"vocab {tok.vocab_size}")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        use_rope=args.use_rope,
        use_checkpoint=args.checkpoint,
    )
    model = GPT(cfg).to(args.device)
    eff_batch = args.batch_size * args.grad_accum
    print(f"  model: {model.num_params()/1e6:.2f}M parameters on {args.device} | "
          f"micro-batch {args.batch_size} x grad_accum {args.grad_accum} = {eff_batch} effective")

    # optionally warm-start weights from an existing model (for fine-tuning). We drop
    # the causal-mask buffers so a bigger block_size builds its own correctly-sized ones.
    if args.init_from and not (args.resume and os.path.exists(args.out)):
        ck = torch.load(args.init_from, map_location=args.device)
        sd = {k: v for k, v in ck["model"].items() if not k.endswith(".mask")}
        model.load_state_dict(sd, strict=False)
        del ck, sd                      # free the transient checkpoint copy from VRAM
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"  initialized weights from {args.init_from} (block_size {args.block_size})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                                  betas=(0.9, 0.95))
    use_amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    torch.set_float32_matmul_precision("high")  # allow TF32 matmuls (faster)

    def save_ckpt(step, with_optim=True):
        """Save a checkpoint. With optimizer state + step it can be resumed; either
        way chat.py/api.py can load it (they only read model/config/tokenizer)."""
        blob = {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "tokenizer": args.tokenizer,
            "tokenizer_type": "bpe",
            "step": step,
        }
        if with_optim:
            blob["optimizer"] = optimizer.state_dict()
            blob["scaler"] = scaler.state_dict()
        tmp = args.out + ".tmp"
        torch.save(blob, tmp)           # write to temp then replace -> never a half-written file
        os.replace(tmp, args.out)

    start_step = 0
    if args.resume and os.path.exists(args.out):
        ck = torch.load(args.out, map_location=args.device)
        if "optimizer" in ck and "step" in ck:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            if "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            start_step = ck["step"]
            print(f"  RESUMED from step {start_step}")

    print(f"Training for {args.steps} steps (from {start_step + 1}) ...")
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        # accumulate gradients over several micro-batches to get a big effective batch
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, cfg.block_size, args.batch_size, args.device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.eval_every == 0 or step == 1:
            stats = estimate_loss(model, train_data, val_data, cfg, args.device)
            dt = time.time() - t0
            print(f"  step {step:>5} | train {stats['train']:.3f} | "
                  f"val {stats['val']:.3f} | lr {lr:.1e} | {dt:.0f}s elapsed", flush=True)

        # periodic resumable checkpoint so a kill/crash never loses much progress
        if step % args.save_every == 0:
            save_ckpt(step, with_optim=True)
            print(f"    (checkpoint saved at step {step})", flush=True)

    save_ckpt(args.steps, with_optim=True)
    print(f"Saved checkpoint -> {args.out}")

    # sample a fake conversation to see what it learned
    print("\n--- sample conversation ---")
    prompt = "<|user|>Hi there! How are you doing today?<|eot|><|bot|>"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=args.device)
    eot = tok.special_tokens["<|eot|>"]
    out = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=40, stop_token=eot)[0]
    print(tok.decode(out.tolist()))


if __name__ == "__main__":
    main()
