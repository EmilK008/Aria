"""
prepare_aria.py -- build the full "Aria" corpus: chat + persona + Q&A.

Blends four ingredients into our one chat format:
  * everyday chat       -- DailyDialog + SODA + Persona-Chat (keeps normal conversation)
  * persona (oversampled) -- so Aria reliably knows her name/identity
  * Alpaca Q&A          -- so she learns to actually answer questions

Run:  python prepare_aria.py --max_soda 100000 --max_alpaca 30000 --persona_repeat 40
"""

import argparse
import os
import re
import urllib.request

import pyarrow.parquet as pq
import torch

from bpe import BPETokenizer
from persona import build_persona_dialogues
from retrieval import grounded_user_text
# reuse the loaders we already wrote
from prepare_chat_big import load_dailydialog, load_soda, load_persona, _parquet_urls, _download

USER, BOT, EOT = "<|user|>", "<|bot|>", "<|eot|>"
SPECIALS = [USER, BOT, EOT]


def load_alpaca(max_rows, max_out_chars=600, path="alpaca.parquet"):
    """Q&A pairs from Alpaca-cleaned, kept short enough for our context window."""
    if not os.path.exists(path):
        _download(_parquet_urls("yahma/alpaca-cleaned", "default", "train")[0], path)
    tbl = pq.read_table(path, columns=["instruction", "input", "output"]).to_pylist()
    out = []
    for r in tbl:
        instr = (r["instruction"] or "").strip()
        inp = (r["input"] or "").strip()
        ans = (r["output"] or "").strip()
        if not instr or not ans:
            continue
        if len(ans) > max_out_chars:        # skip very long answers (won't fit / too hard)
            continue
        question = f"{instr}\n{inp}" if inp else instr
        if len(question) > max_out_chars:
            continue
        out.append([question, ans])         # a single user->bot turn
        if len(out) >= max_rows:
            break
    return out


def _answer_sentence(context, answer, start, n_sentences=2):
    """Return the sentence containing the answer span, plus the next sentence.

    SQuAD answers are tiny spans ("Paris"), which trains terse 2-word replies.
    Using the answer's sentence (and the one after) as the target teaches Aria to
    answer with real, informative sentences while staying grounded in the passage.
    """
    s = context.rfind(". ", 0, start)
    s = 0 if s == -1 else s + 2
    e = start
    for _ in range(n_sentences):           # walk forward over a couple of sentences
        nxt = context.find(". ", e)
        if nxt == -1:
            e = len(context)
            break
        e = nxt + 1
    return context[s:e].strip()


def _first_sentences(text, n=3, max_chars=400):
    """First up-to-n sentences of a passage (for 'describe X' style answers)."""
    out, e = "", 0
    for _ in range(n):
        nxt = text.find(". ", e)
        if nxt == -1:
            out = text
            break
        e = nxt + 1
        out = text[:e]
        if len(out) >= max_chars:
            break
    return out.strip()


def load_squad(max_rows, max_ctx_chars=800, path="squad.parquet"):
    """Grounded QA: (context paragraph, question) -> the answer's full sentence.

    This is the key ingredient that teaches Aria to READ a passage and answer
    from it -- exactly the skill web search needs. We format each example with
    the same grounded template we'll use at inference time.
    """
    if not os.path.exists(path):
        _download(_parquet_urls("rajpurkar/squad", "plain_text", "train")[0], path)
    tbl = pq.read_table(path, columns=["context", "question", "answers"]).to_pylist()
    out, seen = [], set()
    for r in tbl:
        ctx = (r["context"] or "").strip()
        q = (r["question"] or "").strip()
        ans = r["answers"] or {}
        texts, starts = ans.get("text", []), ans.get("answer_start", [])
        if not ctx or not q or not texts or len(ctx) > max_ctx_chars:
            continue
        span = texts[0].strip()
        if not span:
            continue
        # sentence(s) containing the answer -> a more informative reply
        sentence = _answer_sentence(ctx, span, starts[0] if starts else ctx.find(span))
        target = sentence if (span.lower() in sentence.lower() and len(sentence) <= 420) else f"{span}."
        key = (q, target)
        if key in seen:           # SQuAD repeats contexts across questions; dedupe Q/A
            continue
        seen.add(key)
        out.append([grounded_user_text(ctx, q), target])
        if len(out) >= max_rows:
            break
    return out


def load_soda_multi(max_rows):
    """SODA across all its parquet shards (way more than the single-file loader)."""
    urls = _parquet_urls("allenai/soda", "default", "train")
    out = []
    for n, url in enumerate(urls):
        path = f"soda_train_{n}.parquet"
        if not os.path.exists(path):
            if n == 0 and os.path.exists("soda_train.parquet"):
                path = "soda_train.parquet"      # reuse the one we already downloaded
            else:
                _download(url, path)
        rows = pq.read_table(path, columns=["dialogue"]).column("dialogue").to_pylist()
        for utts in rows:
            d = [u.strip() for u in utts if u and u.strip()]
            if len(d) >= 2:
                out.append(d)
            if len(out) >= max_rows:
                return out
    return out


def load_dolly(path="dolly.parquet"):
    """Databricks Dolly: instruction-following + grounded QA with fuller answers.

    Rows WITH context become grounded examples (teach answering from a passage);
    rows without become plain instruction -> response pairs.
    """
    if not os.path.exists(path):
        _download(_parquet_urls("databricks/databricks-dolly-15k", "default", "train")[0], path)
    rows = pq.read_table(path, columns=["instruction", "context", "response"]).to_pylist()
    out = []
    for r in rows:
        instr = (r["instruction"] or "").strip()
        ctx = (r["context"] or "").strip()
        resp = (r["response"] or "").strip()
        if not instr or not resp or len(resp) > 700:
            continue
        if ctx and len(ctx) <= 1200:
            out.append([grounded_user_text(ctx, instr), resp])   # grounded
        elif not ctx:
            out.append([instr, resp])                            # plain instruction
    return out


def load_oasst(path="oasst.parquet", max_chars=1400):
    """OpenAssistant top-1 English: high-quality human assistant conversations.

    Much better than casual dialogue for teaching Aria to follow instructions and
    give well-structured answers. Each row is a list of {role, content} turns.
    """
    if not os.path.exists(path):
        _download(_parquet_urls("g-ronimo/oasst2_top1_en", "default", "train")[0], path)
    rows = pq.read_table(path, columns=["conversation"]).column("conversation").to_pylist()
    out = []
    for conv in rows:
        utts = []
        for m in conv:
            c = (m.get("content") or "").strip()
            if not c:
                break
            if len(c) > max_chars:                       # trim only very long turns
                c = c[:max_chars].rsplit(". ", 1)[0] + "."
            utts.append(c)
        if len(utts) >= 2:
            out.append(utts)
    return out


def load_descriptions(path="squad.parquet"):
    """'Describe / tell me about X' -> the intro of X's article (multi-sentence).

    Teaches Aria to give a longer, descriptive answer for open 'what is X' questions,
    not just a single fact. Built from the first paragraph of each unique SQuAD topic.
    """
    tbl = pq.read_table(path, columns=["title", "context"]).to_pylist()
    seen, out = set(), []
    templates = ["What is {t}?", "Tell me about {t}.", "Can you tell me about {t}?"]
    for i, r in enumerate(tbl):
        title = (r["title"] or "").replace("_", " ").strip()
        ctx = (r["context"] or "").strip()
        if not title or title in seen or len(ctx) < 80:
            continue
        seen.add(title)
        answer = _first_sentences(ctx, n=3)
        if len(answer) < 60:
            continue
        q = templates[i % len(templates)].format(t=title)
        out.append([grounded_user_text(ctx, q), answer])
    return out


# Only explicit self-introductions -- NOT "I'm American/John/etc." which is too broad
# and would delete most of the conversational data.
_NAME_INTRO = re.compile(r"\b(my name is|call me|i'm called|i am called|name's)\b", re.I)


def filter_name_intros(dialogues):
    """Drop dialogues that introduce a speaker by name -- they compete with Aria's
    persona and are the main reason a small model answers 'what's your name' wrong."""
    return [d for d in dialogues if not any(_NAME_INTRO.search(u) for u in d)]


# template placeholders the model was literally parroting ("[insert name here]", "[Name]",
# "<your name>", "{topic}", "(insert ...)"). Drop any example that contains one.
_PLACEHOLDER = re.compile(r"\[[^\]]{0,40}\]|\{[^\}]{0,40}\}|<[^>]{0,40}>|insert[^.]{0,30}here", re.I)


def filter_placeholders(dialogues):
    return [d for d in dialogues if not any(_PLACEHOLDER.search(u) for u in d)]


def render(dialogue):
    parts = []
    for i, utt in enumerate(dialogue):
        tag = USER if i % 2 == 0 else BOT
        parts.append(f"{tag}{utt}{EOT}")
    return "".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_soda", type=int, default=100000)
    p.add_argument("--max_alpaca", type=int, default=30000)
    p.add_argument("--max_squad", type=int, default=40000)
    p.add_argument("--persona_repeat", type=int, default=500)
    p.add_argument("--persona", default="aria",
                   help="which character to give this model (see persona.py: aria, sage, ...)")
    p.add_argument("--use_dolly", action="store_true", help="add the Dolly instruction set")
    p.add_argument("--use_oasst", action="store_true", help="add OpenAssistant (high quality)")
    p.add_argument("--oasst_repeat", type=int, default=5, help="oversample OASST (it's small but great)")
    p.add_argument("--bpe_sample_chars", type=int, default=80_000_000,
                   help="cap chars used to TRAIN the BPE (encoding still uses everything)")
    # Persona-Chat is OFF by default: its names are "[user N's name]" placeholders
    # that we strip to empty, which trains the bot to say "My name is ." -- it also
    # fights our own persona. Pass --include_persona_chat to add it back.
    p.add_argument("--include_persona_chat", action="store_true")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--tokenizer", default="bpe_aria.json")
    p.add_argument("--tokenizer_from", default=None,
                   help="reuse an existing tokenizer (needed to warm-start/continue an "
                        "existing model -- the token ids must match)")
    p.add_argument("--out_train", default="aria_train.pt")
    p.add_argument("--out_val", default="aria_val.pt")
    p.add_argument("--val_frac", type=float, default=0.02)
    args = p.parse_args()

    print("Loading datasets ...")
    chat = load_dailydialog()
    soda = load_soda_multi(args.max_soda) if args.max_soda > 100000 else load_soda(args.max_soda)
    alpaca = load_alpaca(args.max_alpaca)
    squad = load_squad(args.max_squad)
    descriptions = load_descriptions()
    dolly = load_dolly() if args.use_dolly else []
    oasst = filter_placeholders(load_oasst()) if args.use_oasst else []
    persona = build_persona_dialogues(args.persona)
    persona_chat = load_persona() if args.include_persona_chat else []

    # clean the non-persona data: drop random-name intros (compete with Aria's name)
    # and template placeholders like "[insert name here]" (the model was parroting them)
    chat = filter_placeholders(filter_name_intros(chat))
    soda = filter_placeholders(filter_name_intros(soda))
    alpaca = filter_placeholders(alpaca)
    dolly = filter_placeholders(dolly)
    print(f"  DailyDialog {len(chat):,} | SODA {len(soda):,} | Alpaca {len(alpaca):,} | "
          f"SQuAD(grounded) {len(squad):,} | descriptions {len(descriptions):,} | "
          f"Dolly {len(dolly):,} | OASST {len(oasst):,} x{args.oasst_repeat} | "
          f"persona {len(persona)} x{args.persona_repeat}")

    dialogues = (chat + persona_chat + soda + alpaca + squad + descriptions + dolly
                 + oasst * args.oasst_repeat + persona * args.persona_repeat)
    print(f"  TOTAL dialogues: {len(dialogues):,}")

    # 1. train BPE on a sample of the text (a representative sample is enough, and
    #    it keeps BPE training fast even when the full corpus is huge)
    sample_parts, n_chars = [], 0
    for d in dialogues:
        for u in d:
            sample_parts.append(u)
            n_chars += len(u)
        if n_chars >= args.bpe_sample_chars:
            break
    if args.tokenizer_from:                 # reuse existing tokenizer (for warm-start)
        tok = BPETokenizer.load(args.tokenizer_from)
        print(f"  reusing tokenizer {args.tokenizer_from} (vocab {tok.vocab_size})")
    else:
        plain = "\n".join(sample_parts)
        print(f"  training BPE (vocab {args.vocab_size}) on {len(plain)/1e6:.1f}M chars (sampled) ...")
        tok = BPETokenizer()
        tok.train(plain, vocab_size=args.vocab_size, special_tokens=SPECIALS)
        tok.save(args.tokenizer)
        print(f"  tokenizer -> {args.tokenizer} (vocab {tok.vocab_size})")

    # 2. encode everything
    print("  encoding ...")
    all_ids = []
    for i, d in enumerate(dialogues):
        all_ids.extend(tok.encode(render(d)))
        if (i + 1) % 100000 == 0:
            print(f"    {i+1:,}/{len(dialogues):,} ({len(all_ids)/1e6:.1f}M tokens)")
    data = torch.tensor(all_ids, dtype=torch.long)
    print(f"  total tokens: {len(data):,}")

    # 3. split + save (clone so slices don't drag the whole storage)
    n_val = int(len(data) * args.val_frac)
    torch.save(data[:-n_val].clone(), args.out_train)
    torch.save(data[-n_val:].clone(), args.out_val)
    print(f"  saved {len(data)-n_val:,} train / {n_val:,} val tokens")


if __name__ == "__main__":
    main()
