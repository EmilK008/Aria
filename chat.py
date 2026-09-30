"""
chat.py -- talk to a trained model in the terminal.

Handles both kinds of model we trained:
  * a BPE chat model (ckpt_chat.pt): real back-and-forth conversation. We keep
    the dialogue history, wrap your message in <|user|>...<|eot|><|bot|>, and let
    the model generate its reply until it emits <|eot|>.
  * a char-level model (ckpt.pt): "continue this text" autocomplete.

Run:  python chat.py                 (chat model, default)
      python chat.py --ckpt ckpt.pt  (the Shakespeare char model)

Commands: /temp <f>  /topk <n>  /reset (clear history)  /quit
"""

import argparse
import sys

import torch

# Wikipedia text is full of accents/IPA; make sure the terminal never crashes on them
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from data import CharTokenizer
from bpe import BPETokenizer
from model import GPT, GPTConfig
from retrieval import needs_search, grounded_user_text, retrieve
from tools import solve_math

USER, BOT, EOT = "<|user|>", "<|bot|>", "<|eot|>"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ttype = ckpt.get("tokenizer_type", "char")
    if ttype == "bpe":
        tok = BPETokenizer.load(ckpt["tokenizer"])
    else:
        tok = CharTokenizer.load(ckpt["tokenizer"])
    return model, tok, cfg, ttype


def chat_loop(model, tok, cfg, device, temp, topk):
    """Conversational loop for the BPE chat model."""
    eot = tok.special_tokens[EOT]
    history = ""   # running transcript in chat format
    print("Chatbot ready. Say hi! (/reset to clear, /quit to exit)\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not msg:
            continue
        if msg.startswith("/"):
            parts = msg.split()
            if parts[0] == "/quit":
                break
            elif parts[0] == "/reset":
                history = ""; print("  (history cleared)")
            elif parts[0] == "/temp" and len(parts) > 1:
                temp = float(parts[1]); print(f"  temperature = {temp}")
            elif parts[0] == "/topk" and len(parts) > 1:
                topk = int(parts[1]); print(f"  top_k = {topk}")
            else:
                print("  commands: /temp /topk /reset /quit")
            continue

        # exact math beats guessing -> answer directly
        calc = solve_math(msg)
        if calc is not None:
            print(f"bot> {calc}\n")
            history = history + f"{USER}{msg}{EOT}{BOT}{calc}{EOT}"
            continue

        # factual question? look it up and ground the answer in real text.
        source = None
        if needs_search(msg):
            print("  (searching the web...)")
            ctx_text, source, _ = retrieve(msg)   # Wikipedia or general web
        else:
            ctx_text = None

        if ctx_text:
            # grounded answer: self-contained prompt, no chat history
            prompt = f"{USER}{grounded_user_text(ctx_text, msg)}{EOT}{BOT}"
            grounded = True
        else:
            # normal chat: prior history + this user turn
            prompt = f"{history}{USER}{msg}{EOT}{BOT}"
            grounded = False

        ids = tok.encode(prompt)[-(cfg.block_size - 130):]
        context = torch.tensor([ids], dtype=torch.long, device=device)
        # grounded answers: lower temperature (precise), longer + min length, no repeats
        gen_temp = 0.3 if grounded else temp
        out = model.generate(context, max_new_tokens=(180 if grounded else 120),
                             temperature=gen_temp, top_k=topk, stop_token=eot,
                             repetition_penalty=(1.2 if grounded else 1.15),
                             min_new_tokens=(24 if grounded else 0),
                             no_repeat_ngram_size=(3 if grounded else 0))[0]
        reply = tok.decode(out.tolist()[len(ids):]).replace(EOT, "").strip()
        if source:
            print(f"bot> {reply}\n     (source: {source})\n")
        else:
            print(f"bot> {reply}\n")

        # update history (for grounded turns, store just the short Q/A, not the big context)
        history = (history + f"{USER}{msg}{EOT}{BOT}{reply}{EOT}") if grounded else (prompt + reply + EOT)
        if len(tok.encode(history)) > cfg.block_size * 2:
            history = history[len(history) // 2:]


def complete_loop(model, tok, device, temp, topk, n_tokens=300):
    """Autocomplete loop for the char-level model."""
    print("Type a prompt; the model continues it. (/quit to exit)\n")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not prompt.strip():
            continue
        if prompt.strip() == "/quit":
            break
        ids = tok.encode(prompt) or [0]
        context = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=n_tokens, temperature=temp, top_k=topk)[0]
        generated = tok.decode(out.tolist()[len(ids):])
        print(f"bot> {prompt}{generated}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_chat.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=40)
    args = p.parse_args()

    model, tok, cfg, ttype = load_model(args.ckpt, args.device)
    print(f"Loaded {model.num_params()/1e6:.1f}M-param {ttype} model on {args.device}.\n")

    if ttype == "bpe":
        chat_loop(model, tok, cfg, args.device, args.temp, args.topk)
    else:
        complete_loop(model, tok, args.device, args.temp, args.topk)


if __name__ == "__main__":
    main()
