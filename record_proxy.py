#!/usr/bin/env python3
"""Recording proxy for an OpenAI-compatible server. Standard library only.

    client -> http://127.0.0.1:8091/v1 -> this proxy -> upstream server

Everything passes through unchanged, except with --rules (below). Each POST is
also saved as it streams, so a turn that loops or gets killed still leaves its text:

    <dir>/<HHMMSS>-<n>.request.json   the request body
    <dir>/<HHMMSS>-<n>.stream.txt     reasoning, answer and tool-call text, as it arrives
    <dir>/live.txt                    the current turn only (tail -f this)

usage: record_proxy.py --upstream http://HOST:8090 [--port 8091] [--dir ~/dsv4-proxy]
                       [--rules FILE]
Binds to 127.0.0.1 only. Recordings contain your prompts and code: keep them local.

--rules FILE puts the file's text at the top of the first user message of every
request that offers tools (agent turns; OpenCode's title requests have none),
unless that message already starts with it. That is where the Test B runs pasted
the working rules, so the model sees the same thing without anyone pasting them.
The saved request is the one sent upstream, rules included.
"""
import argparse
import http.client
import itertools
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

COUNTER = itertools.count(1)
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "upgrade",
       "proxy-authenticate", "proxy-authorization", "trailer", "host", "content-length"}


def extract(line):
    """Readable text from one SSE line, or '' if it has none."""
    if not line.startswith(b"data: {"):
        return ""
    try:
        choices = json.loads(line[6:]).get("choices") or []
    except ValueError:
        return ""
    out = []
    for choice in choices:
        delta = choice.get("delta") or {}
        out.append(delta.get("reasoning_content") or "")
        out.append(delta.get("content") or "")
        for call in delta.get("tool_calls") or []:
            fn = call.get("function") or {}
            out.append(f"\n[tool {fn.get('name') or ''}] {fn.get('arguments') or ''}\n")
        if choice.get("finish_reason"):
            out.append(f"\n[finish: {choice['finish_reason']}]\n")
    return "".join(out)


def add_rules(body, rules):
    """Return (body, added): `rules` put before the first user message's text."""
    if not rules:
        return body, False
    try:
        data = json.loads(body)
    except ValueError:
        return body, False
    if not isinstance(data, dict) or not data.get("tools"):
        return body, False
    user = next((x for x in data.get("messages") or [] if isinstance(x, dict) and x.get("role") == "user"), None)
    if user is None:
        return body, False
    content = user.get("content")
    if isinstance(content, str):
        if content.startswith(rules):
            return body, False
        user["content"] = rules + "\n\n" + content
    elif isinstance(content, list):
        part = next((x for x in content if isinstance(x, dict) and x.get("type") == "text"), None)
        if part is None or str(part.get("text", "")).startswith(rules):
            return body, False
        part["text"] = rules + "\n\n" + part.get("text", "")
    else:
        return body, False
    return json.dumps(data, ensure_ascii=False).encode(), True


def make_handler(upstream, folder, rules=None):
    target = urlsplit(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _forward(self, body=None):
            conn = http.client.HTTPConnection(target.hostname, target.port, timeout=None)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            conn.request(self.command, self.path, body=body, headers=headers)
            return conn, conn.getresponse()

        def _start_reply(self, resp):
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def do_GET(self):
            try:
                conn, resp = self._forward()
            except OSError as e:
                return self.send_error(502, f"upstream: {e}")
            self._start_reply(resp)
            self.wfile.write(resp.read())
            conn.close()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            stem = os.path.join(folder, f"{time.strftime('%H%M%S')}-{next(COUNTER):04d}")
            body, added = add_rules(body, rules)
            if added:
                print(f"{os.path.basename(stem)}: added working rules", file=sys.stderr, flush=True)
            with open(stem + ".request.json", "wb") as f:
                f.write(body)
            try:
                conn, resp = self._forward(body)
            except OSError as e:
                return self.send_error(502, f"upstream: {e}")
            self._start_reply(resp)
            client_ok = True
            with open(stem + ".stream.txt", "w", buffering=1) as rec, \
                    open(os.path.join(folder, "live.txt"), "w", buffering=1) as live:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    text = extract(line)
                    if text:
                        rec.write(text)
                        live.write(text)
                    if client_ok:
                        try:
                            self.wfile.write(line)
                            self.wfile.flush()
                        except OSError:
                            client_ok = False  # keep recording what the model produces
                            rec.write("\n[client disconnected]\n")
            conn.close()

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--dir", default="~/dsv4-proxy")
    ap.add_argument("--rules", help="file whose text goes before the first user message (agent turns)")
    args = ap.parse_args()
    rules = open(os.path.expanduser(args.rules)).read().strip() if args.rules else None
    folder = os.path.join(os.path.expanduser(args.dir), time.strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    print(f"127.0.0.1:{args.port} -> {args.upstream}, recording to {folder}"
          + (f", adding rules from {args.rules}" if rules else ""), file=sys.stderr, flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.upstream, folder, rules)).serve_forever()


if __name__ == "__main__":
    main()
