# Training Aria in the cloud

Your training code is plain PyTorch + CUDA with resumable checkpoints, so it runs
on any rented GPU with **no code changes** — you just point it at a bigger model.
This is the way to go once you want a model too big for the 12 GB 4070 (a bigger
GPU has 40–80 GB of *fast* VRAM, so nothing spills into slow system RAM).

---

## 1. Put the code somewhere the instance can grab it

Push this folder to a **private GitHub repo** (the `.py` files are tiny). You do
**not** need to upload the datasets — `prepare_aria.py` re-downloads them from
HuggingFace on the instance.

```bash
cd D:\AI_Chatbot
git init && git add *.py *.txt *.md *.html *.json *.js *.svg
git commit -m "Aria"
# create a private repo on github.com, then:
git remote add origin https://github.com/<you>/aria.git
git push -u origin main
```

Don't commit `search_keys.json` (it holds API keys) or the big `*.pt` files.

---

## 2. Rent a GPU

**Runpod** (easiest) or **Vast.ai** (cheapest). Pick a GPU and a **PyTorch**
template/image (torch pre-installed).

| Want | Pick | Rough cost |
|------|------|-----------|
| Solid, big model | **A100 80 GB** | ~$1–2/hr |
| Fastest | **H100 80 GB** | ~$2–4/hr |
| Budget | **A100 40 GB / A6000 48 GB** | ~$0.5–1/hr |

A real run is usually a few hours → **$10–50 total**.

---

## 3. On the instance

Open the web terminal (or SSH), then:

```bash
git clone https://github.com/<you>/aria.git && cd aria
pip install pyarrow                 # torch is already on PyTorch images
python -c "import torch; print(torch.cuda.get_device_name(0))"   # sanity check

# build the data (downloads datasets from HuggingFace, ~10-20 min)
# --persona picks the character (see persona.py): the cloud flagship is "sage".
python prepare_aria.py --max_soda 500000 --max_alpaca 52000 --max_squad 87000 \
  --use_dolly --use_oasst --persona_repeat 600 --vocab_size 32000 --persona sage

# train a BIGGER model (this is what the cloud GPU is for).
# On 80 GB you can go much larger; --checkpoint lets you push size even further.
python -u train_chat.py --train aria_train.pt --val aria_val.pt \
  --tokenizer bpe_aria.json --out ckpt_cloud.pt \
  --n_layer 24 --n_head 16 --n_embd 1536 --block_size 1024 --use_rope --checkpoint \
  --batch_size 24 --grad_accum 2 --steps 20000 --warmup 500 --save_every 1000 \
  --lr 3e-4 --resume 2>&1 | tee train.log
```

**Run it so it survives disconnects** — start inside `tmux` (or `nohup ... &`) so
closing your laptop doesn't kill the job:
```bash
tmux new -s train        # run the training command inside; detach with Ctrl-b then d
tmux attach -t train     # reattach later to check on it
```

Because of `--resume` + `--save_every`, a preempted spot instance just picks up
from the last checkpoint when you relaunch the same command.

---

## 4. Bring the model home

Download the finished checkpoint (Runpod/Vast have a file browser, or use `scp`):

```bash
# from your PC:
scp root@<instance-ip>:/workspace/aria/ckpt_cloud.pt .
# also grab its tokenizer:
scp root@<instance-ip>:/workspace/aria/bpe_aria.json bpe_cloud.json
```

Then **shut the instance down** so you stop paying.

---

## 5. Use it locally

Make a lean copy (drops the optimizer state) and add it to the model list
(see `models.json`) so it shows up in the app's model dropdown:

```bash
python -c "import torch; ck=torch.load('ckpt_cloud.pt',map_location='cpu'); \
  torch.save({'model':ck['model'],'config':ck['config'],'tokenizer':'bpe_cloud.json','tokenizer_type':'bpe'},'ckpt_cloud.pt')"
```

Add an entry to `models.json`, restart `python api.py`, and pick it in the dropdown.

---

## Going *really* big (multiple GPUs)

One 80 GB GPU (+ `--checkpoint`) handles ~1–3B params. To use **several GPUs at
once**, the training loop needs `torch.nn.parallel.DistributedDataParallel` — a
moderate, well-documented change. Do a single-GPU run first; add multi-GPU only if
you actually outgrow it.
