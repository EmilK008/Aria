# From-Scratch AI Chatbot

A GPT-style language model (a decoder-only transformer) built **from scratch** in
PyTorch — every layer written by hand (attention, transformer blocks, tokenizers,
training loop). No `nn.Transformer`, no external AI APIs. The model genuinely
learns language from raw text and generates its own replies.

Trained on an RTX 4070 (CUDA). Two models live here:

1. **Shakespeare model** (`ckpt.pt`) — char-level, learns to write Shakespeare.
2. **Chat model** (`ckpt_chat.pt`) — BPE tokenizer, trained on the DailyDialog
   conversation dataset to actually talk back to you.

## The pieces

| File | What it is |
|------|------------|
| [model.py](model.py) | The GPT itself — multi-head causal self-attention (computed by hand), transformer blocks, `generate()`. |
| [data.py](data.py) | Char-level tokenizer + batching for training. |
| [bpe.py](bpe.py) | Our from-scratch Byte-Pair Encoding tokenizer (GPT-2 style), with special tokens. |
| [train.py](train.py) | Trains the char model on a text corpus. |
| [prepare_chat.py](prepare_chat.py) | Downloads/formats DailyDialog, trains the BPE, encodes tokens. |
| [train_chat.py](train_chat.py) | Trains the chat model on the conversation tokens. |
| [chat.py](chat.py) | Talk to either model in the terminal. |
| [api.py](api.py) | A zero-dependency local HTTP API so a real app can use the bot. |

## How it works (the 60-second version)

Text is split into **tokens**. Each token becomes a vector (an *embedding*).
The transformer's **self-attention** lets every token look back at earlier tokens
to gather context; stacked **blocks** refine that representation; finally the model
predicts a probability for every possible *next* token. Training just means: show
it real text, measure how wrong its next-token guesses are (cross-entropy loss),
and nudge the weights to be less wrong — a few thousand times. To chat, we feed it
`<|user|>your message<|eot|><|bot|>` and let it generate until `<|eot|>`.

## Usage

```bash
# --- chat model ---
python prepare_chat.py            # build BPE + tokenized data (needs dd_train.parquet)
python train_chat.py --steps 5000 # train on the GPU (~7 min on a 4070)
python chat.py                    # talk to it in the terminal

# --- run it as a service ---
python api.py                     # serves http://127.0.0.1:8000

# --- the Shakespeare char model ---
python train.py --steps 3000
python chat.py --ckpt ckpt.pt
```

### Calling the API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hi, how are you?"}'
# -> {"reply": "..."}
```

## Reality check

These are small models (~12M parameters) trained on small datasets on one GPU.
They learn the *style and patterns* of their training data and produce plausible,
often coherent, sometimes delightfully odd replies — they are not ChatGPT, and
that's the whole point: it's a real neural net we built and trained ourselves.

To make it better: more/cleaner data, a bigger model (`--n_layer --n_embd`),
and more training steps.
