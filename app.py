"""
app.py -- Aria as a native desktop app.

Starts the chatbot API server in the background and opens the web UI in its own
window (no browser needed). Double-click friendly.

Run:  python app.py
"""

import argparse
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import torch
import webview   # pywebview

from api import ModelRegistry, make_handler


def start_server(ckpt, host, port, device):
    registry = ModelRegistry(device, ckpt)
    server = ThreadingHTTPServer((host, port), make_handler(registry))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return registry, server


def wait_until_up(url, timeout=30):
    for _ in range(timeout * 2):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_chat.pt")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading Aria ({args.ckpt}) on {args.device} ...")
    registry, _ = start_server(args.ckpt, "127.0.0.1", args.port, args.device)
    url = f"http://127.0.0.1:{args.port}"
    wait_until_up(url + "/health")
    print(f"  {registry.get().model.num_params()/1e6:.1f}M-param model ready. Opening window...")

    webview.create_window("Aria", url, width=460, height=780, min_size=(380, 560))
    webview.start()   # blocks until the window is closed


if __name__ == "__main__":
    main()
