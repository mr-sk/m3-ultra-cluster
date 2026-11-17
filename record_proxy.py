#!/usr/bin/env python3
"""Recording proxy for an OpenAI-compatible server. Standard library only.

    client -> http://127.0.0.1:8091/v1 -> this proxy -> upstream server

Everything passes through unchanged, except with --rules (below). Each POST is
also saved as it streams, so a turn that loops or gets killed still leaves its text:

    <dir>/<HHMMSS>-<n>.request.json   the request body
    <dir>/<HHMMSS>-<n>.stream.txt     reasoning, answer and tool-call text, as it arrives
    <dir>/live.txt                    the current turn only (tail -f this)

usage: record_proxy.py --upstream http://HOST:8090 [--port 8091] [--dir ~/dsv4-proxy]
                       [--rules FILE] [--fill-edit-path]
Binds to 127.0.0.1 only. Recordings contain your prompts and code: keep them local.

--fill-edit-path (off by default) fills a missing `path` in an `edit` tool call
when the session history makes it unambiguous: exactly one file the session has
read, edited or written contains the call's `oldString` (20+ characters). Each fill
is logged. Otherwise the call passes unchanged and the client rejects it.

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
import re
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


# --- --fill-edit-path ------------------------------------------------------------
# Test B, 2026-11-15: a quarter of generated tokens went into edit calls OpenCode
# rejected, most missing `path` with oldString and newString present; the model
# then regenerates the whole edit. Checked against recordings: 51 of 53 rejected
# no-path edits whose retry shows the intended file were inferred right, 0 wrong,
# 2 left alone.
MIN_OLD = 20  # shorter oldStrings ("}", "    }\n") match too much to be evidence
_READ_HEAD = re.compile(r"^Read file (\S.*?), lines \d+-\d+\n")
_LINE_NO = re.compile(r"^\d+: ", re.M)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def known_text(messages):
    """{path: [text fragments the file is known to contain]} from reads, edits, writes."""
    calls, known = {}, {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            calls[c.get("id")] = (fn.get("name"), args)
        if m.get("role") != "tool":
            continue
        name, args = calls.get(m.get("tool_call_id"), (None, {}))
        out = _text(m.get("content"))
        path = args.get("path") if isinstance(args, dict) else None
        if not path or out.lstrip().startswith('{"error"'):
            continue
        if name == "read":
            head = _READ_HEAD.match(out)
            body = out[head.end():] if head else out
            known.setdefault(path, []).append(_LINE_NO.sub("", body))
        elif name == "edit" and out.startswith("Edited") and args.get("newString"):
            known.setdefault(path, []).append(args["newString"])
        elif name == "write" and args.get("content") is not None:
            known[path] = [args["content"]]
    return known


def infer_edit_path(messages, old_string):
    if not isinstance(old_string, str) or len(old_string.strip()) < MIN_OLD:
        return None
    hits = [p for p, parts in known_text(messages).items() if any(old_string in t for t in parts)]
    return hits[0] if len(hits) == 1 else None


def _requires_path(tools, name):
    for t in tools or []:
        f = t.get("function", t) if isinstance(t, dict) else {}
        if f.get("name") == name:
            return "path" in ((f.get("parameters") or {}).get("required") or [])
    return False


def fill_edit_path(line, messages, tools):
    """Return (line, [filled paths]): an SSE line with missing edit paths filled in."""
    if not line.startswith(b"data: {") or b"tool_calls" not in line:
        return line, []
    try:
        data = json.loads(line[6:])
    except ValueError:
        return line, []
    filled = []
    for choice in data.get("choices") or []:
        for call in (choice.get("delta") or {}).get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") != "edit" or not _requires_path(tools, "edit"):
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                continue
            if not isinstance(args, dict) or "path" in args:
                continue
            path = infer_edit_path(messages, args.get("oldString"))
            if path:
                fn["arguments"] = json.dumps({"path": path, **args}, ensure_ascii=False)
                filled.append(path)
    if not filled:
        return line, []
    ending = line[len(line.rstrip(b"\r\n")):]
    return b"data: " + json.dumps(data, ensure_ascii=False).encode() + ending, filled


def make_handler(upstream, folder, rules=None, fill_path=False):
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
            history = tools = None
            if fill_path:
                try:
                    req = json.loads(body)
                    history, tools = req.get("messages") or [], req.get("tools") or []
                except (ValueError, AttributeError):
                    history = None
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
                    if history is not None:
                        line, filled = fill_edit_path(line, history, tools)
                        for path in filled:
                            print(f"{os.path.basename(stem)}: filled missing edit path {path}",
                                  file=sys.stderr, flush=True)
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
    ap.add_argument("--fill-edit-path", action="store_true",
                    help="fill an edit call's missing path when the history makes it unambiguous")
    args = ap.parse_args()
    rules = open(os.path.expanduser(args.rules)).read().strip() if args.rules else None
    folder = os.path.join(os.path.expanduser(args.dir), time.strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    print(f"127.0.0.1:{args.port} -> {args.upstream}, recording to {folder}"
          + (f", adding rules from {args.rules}" if rules else "")
          + (", filling missing edit paths" if args.fill_edit_path else ""), file=sys.stderr, flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port),
                        make_handler(args.upstream, folder, rules, args.fill_edit_path)).serve_forever()


if __name__ == "__main__":
    main()
