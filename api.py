"""
api.py -- a tiny local HTTP API for our chatbot, using only the standard library.

Start it:   python api.py
Then any app can talk to the model over HTTP:

  POST http://127.0.0.1:8000/chat
  body: {"message": "hello!", "history": [["hi","hey there"]], "temperature": 0.8}
  resp: {"reply": "..."}

  GET  http://127.0.0.1:8000/health  ->  {"status": "ok", ...}

`history` is an optional list of [user, bot] pairs so the model has context.
No frameworks -- just http.server, so it runs anywhere Python does.
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch

from bpe import BPETokenizer
from data import CharTokenizer
from model import GPT, GPTConfig
from retrieval import needs_search, grounded_user_text, retrieve, _load_keys, _KEYS_FILE
from tools import solve_math, recall_answer

USER, BOT, EOT = "<|user|>", "<|bot|>", "<|eot|>"


def _messages_to_history(messages):
    """Convert an OpenAI-style messages list into our (history_pairs, current) form."""
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    current = ""
    if turns and turns[-1]["role"] == "user":
        current = turns[-1].get("content", "")
        turns = turns[:-1]
    history, i = [], 0
    while i < len(turns) - 1:
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            history.append([turns[i].get("content", ""), turns[i + 1].get("content", "")])
            i += 2
        else:
            i += 1
    return history, current


class ChatEngine:
    """Loads a trained checkpoint and turns messages into replies."""

    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.cfg = GPTConfig(**ckpt["config"])
        self.model = GPT(self.cfg).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device
        self.ttype = ckpt.get("tokenizer_type", "char")
        if self.ttype == "bpe":
            self.tok = BPETokenizer.load(ckpt["tokenizer"])
            self.eot = self.tok.special_tokens[EOT]
        else:
            self.tok = CharTokenizer.load(ckpt["tokenizer"])
            self.eot = None

    def _build_ids(self, message, history):
        history = history or []
        if self.ttype == "bpe":
            prompt = ""
            for u, b in history:
                prompt += f"{USER}{u}{EOT}{BOT}{b}{EOT}"
            prompt += f"{USER}{message}{EOT}{BOT}"
            return self.tok.encode(prompt)[-(self.cfg.block_size - 130):]
        return self.tok.encode(message) or [0]

    def prepare(self, message, history=None, use_search=True, force_search=False):
        """Decide chat vs. grounded(web-search). Returns (input_ids, source_title|None).

        force_search=True (the UI's "Always" mode) skips the heuristic and looks up
        every message; the relevance gate still falls back to chat if nothing fits.
        """
        if self.ttype == "bpe" and use_search and (force_search or needs_search(message)):
            context, source, source_url = retrieve(message)   # Wikipedia or general web
            if context:
                prompt = f"{USER}{grounded_user_text(context, message)}{EOT}{BOT}"
                ids = self.tok.encode(prompt)[-(self.cfg.block_size - 130):]
                return ids, source, source_url
        return self._build_ids(message, history), None, None

    def reply(self, message, history=None, temperature=0.8, top_k=40, max_new_tokens=120,
              repetition_penalty=1.15, use_search=True, force_search=False):
        ids, source, source_url = self.prepare(message, history, use_search, force_search)
        min_new, no_repeat = 0, 0
        if source:  # grounded: precise, informative, and no repeated phrases
            temperature = min(temperature, 0.3)
            max_new_tokens, min_new, no_repeat = max(max_new_tokens, 180), 24, 3
            repetition_penalty = max(repetition_penalty, 1.2)
        ctx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            ctx, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, stop_token=self.eot, repetition_penalty=repetition_penalty,
            min_new_tokens=min_new, no_repeat_ngram_size=no_repeat,
        )[0]
        text = self.tok.decode(out.tolist()[len(ids):]).replace(EOT, "").strip()
        return text, source, source_url

    def stream_reply(self, ids, temperature=0.8, top_k=40, max_new_tokens=120,
                     repetition_penalty=1.15, min_new_tokens=0, no_repeat_ngram_size=0):
        """Yield the reply piece by piece, given already-prepared input ids."""
        ctx = torch.tensor([ids], dtype=torch.long, device=self.device)
        gen_ids = []
        emitted = ""
        for tok_id in self.model.generate_stream(
            ctx, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, stop_token=self.eot, repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens, no_repeat_ngram_size=no_repeat_ngram_size,
        ):
            gen_ids.append(tok_id)
            # decode the whole thing so far; only emit if it ends on a complete char
            text = self.tok.decode(gen_ids).replace(EOT, "")
            if text.endswith("�"):   # incomplete multi-byte char, wait for more
                continue
            delta = text[len(emitted):]
            if delta:
                emitted = text
                yield delta


class ModelRegistry:
    """Holds the list of available models (from models.json) and lazily loads them.

    Only `max_loaded` engines are kept in VRAM at once (default 1) -- switching models
    evicts the oldest, so a dropdown of several models never blows up GPU memory.
    """

    def __init__(self, device, default_ckpt, max_loaded=1):
        self.device = device
        self.max_loaded = max_loaded
        self.models = self._manifest(default_ckpt)
        self.cache = OrderedDict()

    def _manifest(self, default_ckpt):
        try:
            with open("models.json", encoding="utf-8") as f:
                m = json.load(f)
            m = [x for x in m if os.path.exists(x.get("ckpt", ""))]
            if m:
                return m
        except Exception:
            pass
        return [{"id": "aria", "name": "Aria", "ckpt": default_ckpt}]

    @property
    def default_id(self):
        return self.models[0]["id"]

    def list_payload(self):
        return {"default": self.default_id,
                "models": [{"id": x["id"], "name": x["name"]} for x in self.models]}

    def _entry(self, model_id):
        for x in self.models:
            if x["id"] == model_id:
                return x
        return self.models[0]

    def get(self, model_id=None):
        entry = self._entry(model_id or self.default_id)
        mid = entry["id"]
        if mid in self.cache:
            self.cache.move_to_end(mid)
            return self.cache[mid]
        engine = ChatEngine(entry["ckpt"], self.device)
        self.cache[mid] = engine
        while len(self.cache) > self.max_loaded:        # evict oldest -> free VRAM
            _, old = self.cache.popitem(last=False)
            del old
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return engine


def make_handler(registry):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # needed for chunked streaming responses

        def _json(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")  # let browser apps call it
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # CORS preflight
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        # static files that make up the web app (served from this folder)
        STATIC = {
            "/": ("webchat.html", "text/html; charset=utf-8"),
            "/index.html": ("webchat.html", "text/html; charset=utf-8"),
            "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
            "/sw.js": ("sw.js", "text/javascript"),
            "/icon.svg": ("icon.svg", "image/svg+xml"),
        }

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            q = parse_qs(parsed.query)
            if path == "/health":
                engine = registry.get(q.get("model", [None])[0])   # optional ?model=
                self._json(200, {"status": "ok", "model": engine.ttype,
                                 "params_millions": round(engine.model.num_params() / 1e6, 2),
                                 "context_tokens": engine.cfg.block_size,
                                 "reply_reserve": 130})
            elif path == "/models":                # list models for the app dropdown
                self._json(200, registry.list_payload())
            elif path == "/settings/keys":         # which search API keys are set (masked)
                keys = _load_keys()
                self._json(200, {"brave": "brave" in keys, "tavily": "tavily" in keys})
            elif path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": m["id"], "object": "model", "owned_by": "from-scratch"}
                    for m in registry.models]})
            elif path in self.STATIC:
                self._serve_static(*self.STATIC[path])
            else:
                self._json(404, {"error": "not found"})

        def _serve_static(self, filename, content_type):
            try:
                with open(os.path.join(os.path.dirname(__file__), filename), "rb") as f:
                    body = f.read()
            except FileNotFoundError:
                return self._json(404, {"error": f"{filename} not found"})
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == "/chat":
                return self._handle_chat()
            if self.path == "/chat/stream":
                return self._handle_stream()
            if self.path == "/count":
                return self._handle_count()
            if self.path == "/v1/chat/completions":
                return self._handle_openai()
            if self.path == "/settings/keys":
                return self._handle_setkeys()
            return self._json(404, {"error": "not found"})

        def _handle_setkeys(self):
            """Save Brave/Tavily search API keys (a local file). Empty value clears one."""
            try:
                data = self._read_body()
                current = {}
                try:
                    with open(_KEYS_FILE, encoding="utf-8") as f:
                        current = json.load(f)
                except Exception:
                    pass
                for k in ("brave", "tavily"):
                    if k in data:
                        v = (data[k] or "").strip()
                        if v:
                            current[k] = v
                        else:
                            current.pop(k, None)   # empty value clears the key
                with open(_KEYS_FILE, "w", encoding="utf-8") as f:
                    json.dump(current, f)
                keys = _load_keys()
                self._json(200, {"ok": True, "brave": "brave" in keys, "tavily": "tavily" in keys})
            except Exception as e:
                self._json(500, {"error": str(e)})

        def _handle_openai(self):
            """OpenAI-compatible /v1/chat/completions -- so any OpenAI client can use Aria."""
            try:
                data = self._read_body()
                engine = registry.get(data.get("model"))
                messages = data.get("messages", [])
                history, message = _messages_to_history(messages)
                if not message:
                    return self._json(400, {"error": {"message": "no user message"}})
                temperature = float(data.get("temperature", 0.8))
                stream = bool(data.get("stream", False))
                cid = "chatcmpl-" + str(int(time.time() * 1000))
                calc = solve_math(message)

                if not stream:
                    if calc is not None:
                        reply = calc
                    else:
                        reply, _, _ = engine.reply(message, history=history, temperature=temperature)
                    return self._json(200, {
                        "id": cid, "object": "chat.completion", "created": int(time.time()),
                        "model": "aria",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": reply}}],
                    })

                # streaming: OpenAI-style chunked deltas
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def chunk(delta=None, finish=None):
                    payload = {"id": cid, "object": "chat.completion.chunk",
                               "created": int(time.time()), "model": "aria",
                               "choices": [{"index": 0, "delta": delta or {},
                                            "finish_reason": finish}]}
                    msg = ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")
                    self.wfile.write(f"{len(msg):X}\r\n".encode()); self.wfile.write(msg + b"\r\n")
                    self.wfile.flush()

                chunk(delta={"role": "assistant"})
                if calc is not None:
                    chunk(delta={"content": calc})
                else:
                    ids, _ = engine.prepare(message, history)
                    for piece in engine.stream_reply(ids, temperature=temperature):
                        chunk(delta={"content": piece})
                chunk(finish="stop")
                done = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(done):X}\r\n".encode()); self.wfile.write(done + b"\r\n")
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            except Exception as e:
                try:
                    self._json(500, {"error": {"message": str(e)}})
                except Exception:
                    pass

        def _handle_count(self):
            """Exact token count of a conversation, so the UI can show context usage."""
            try:
                data = self._read_body()
                engine = registry.get(data.get("model"))
                text = ""
                for u, b in data.get("history", []):
                    text += f"{USER}{u}{EOT}{BOT}{b}{EOT}"
                if data.get("message"):
                    text += f"{USER}{data['message']}{EOT}{BOT}"
                n = len(engine.tok.encode(text)) if engine.ttype == "bpe" else len(text)
                self._json(200, {"tokens": n, "context_tokens": engine.cfg.block_size,
                                 "reply_reserve": 130})
            except Exception as e:
                self._json(500, {"error": str(e)})

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def _handle_chat(self):
            try:
                data = self._read_body()
                message = data.get("message", "")
                if not message:
                    return self._json(400, {"error": "missing 'message'"})
                calc = solve_math(message)               # exact math beats guessing
                if calc is not None:
                    return self._json(200, {"reply": calc, "tool": "calculator"})
                recalled = recall_answer(message, data.get("history", []))
                if recalled is not None:                 # answer recall from real history
                    return self._json(200, {"reply": recalled, "tool": "memory"})
                engine = registry.get(data.get("model"))
                mode = data.get("search_mode", "auto")   # auto | always | off
                reply, source, source_url = engine.reply(
                    message,
                    history=data.get("history", []),
                    temperature=float(data.get("temperature", 0.8)),
                    top_k=int(data.get("top_k", 40)),
                    max_new_tokens=int(data.get("max_new_tokens", 120)),
                    use_search=(mode != "off"),
                    force_search=(mode == "always"),
                )
                payload = {"reply": reply}
                if source:
                    payload["source"] = source
                    if source_url:
                        payload["source_url"] = source_url
                self._json(200, payload)
            except Exception as e:
                self._json(500, {"error": str(e)})

        def _handle_stream(self):
            """Stream the reply token-by-token using chunked Server-Sent Events."""
            try:
                data = self._read_body()
                message = data.get("message", "")
                if not message:
                    return self._json(400, {"error": "missing 'message'"})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def send_event(payload):
                    msg = ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")
                    self.wfile.write(f"{len(msg):X}\r\n".encode())  # chunk size in hex
                    self.wfile.write(msg + b"\r\n")
                    self.wfile.flush()

                # exact tools beat guessing -> answer directly, no model needed
                direct = solve_math(message)
                direct_tool = "calculator"
                if direct is None:
                    direct = recall_answer(message, data.get("history", []))
                    direct_tool = "memory"
                if direct is not None:
                    send_event({"tool": direct_tool})
                    send_event({"token": direct})
                    send_event({"done": True})
                    self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
                    return

                # decide chat vs. web-search up front so we can announce the source
                engine = registry.get(data.get("model"))
                mode = data.get("search_mode", "auto")   # auto | always | off
                ids, source, source_url = engine.prepare(message, data.get("history", []),
                                             use_search=(mode != "off"),
                                             force_search=(mode == "always"))
                temp = float(data.get("temperature", 0.8))
                max_new = int(data.get("max_new_tokens", 120))
                min_new, no_repeat, rep = 0, 0, 1.15
                if source:  # grounded: precise, informative, no repeated phrases
                    temp = min(temp, 0.3)
                    max_new, min_new, no_repeat, rep = max(max_new, 180), 24, 3, 1.2
                    send_event({"source": source, "url": source_url})
                for piece in engine.stream_reply(
                    ids,
                    temperature=temp,
                    top_k=int(data.get("top_k", 40)),
                    max_new_tokens=max_new,
                    min_new_tokens=min_new,
                    repetition_penalty=rep,
                    no_repeat_ngram_size=no_repeat,
                ):
                    send_event({"token": piece})
                send_event({"done": True})
                self.wfile.write(b"0\r\n\r\n")  # end of chunked stream
                self.wfile.flush()
            except Exception as e:
                try:
                    self._json(500, {"error": str(e)})
                except Exception:
                    pass

        def log_message(self, *a):  # quieter console
            pass

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_chat.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_loaded", type=int, default=1,
                   help="how many models to keep in VRAM at once (dropdown switching)")
    args = p.parse_args()

    registry = ModelRegistry(args.device, args.ckpt, args.max_loaded)
    print(f"Models available: {[m['name'] for m in registry.models]}")
    engine = registry.get()          # eagerly load the default so first reply is fast
    print(f"  default: {engine.model.num_params()/1e6:.1f}M-param model on {args.device}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(registry))
    print(f"Chatbot API listening on http://{args.host}:{args.port}")
    print(f"  GET  /models   (list)   GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
