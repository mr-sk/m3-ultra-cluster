"""OpenAI-compatible chat server for an mlx-vlm model pipelined across an MLX cluster.

mlx-vlm 0.7.12 can split a model across machines (utils.sharded_load) but its own
server cannot, so this fills the gap. Every rank loads its share of the model.
Rank 0 serves HTTP; for each request it broadcasts the job to the other ranks
over mx.distributed, then all ranks generate in lockstep while rank 0 streams
the text back.

Run on the node listed first in the hostfile:
  mlx.launch --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- \
    ~/vlm-env/bin/python ~/vlm_cluster_server.py --model mlx-community/DeepSeek-V4-Flash-4bit

Endpoints (rank 0, default port 8090):
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions   (stream true/false; text only)

Limits, on purpose:
  - One request at a time. Others queue behind it.
  - Tool calling. For DeepSeek-V4 the prompt is built with DeepSeek's own
    reference encoder (dsv4_encoding.py, next to this file), which renders tools,
    tool calls and tool results in the DSML format the model was trained on, and
    its output is parsed back into OpenAI `tool_calls` (several per turn allowed).
    The mlx-community chat template ignores `tools`, so it is not used. Other
    models fall back to an emulated <tool_call>{json}</tool_call> format.
  - Generation always runs until EOS or max_tokens on every rank, so the ranks
    stay in step. `stop` strings are applied to what is sent, not to generation.
  - Prompt cache, one slot. After each prompt is read, a copy of the model's cache
    is kept. A request whose tokens start with that saved prompt (the normal case
    for chat and agents, which resend the conversation plus a new turn) skips
    re-reading it. Anything else is read from scratch.
"""
import argparse
import copy
import json
import random
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx
import dsv4_encoding  # DeepSeek's reference DeepSeek-V4 encoder, vendored (MIT)
import numpy as np
from mlx_vlm import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load, sharded_load
import importlib
# mlx_vlm rebinds the name `mlx_vlm.generate` to a function, so load by full name.
_dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
PromptCacheState = importlib.import_module("mlx_vlm.generate.common").PromptCacheState

DEFAULT_MAX_TOKENS = 8192  # Cline sends no max_tokens; 2048 cut its answers off
THINK_MAX_TOKENS = 32768   # default when thinking: the reasoning alone can run to thousands of tokens
DEFAULT_REASONING_EFFORT = "none"  # --default-reasoning-effort; "none" = chat mode
THINK_END = "</think>"
PROMPT_CACHE = True
# 2048-token prefill chunks hit a Metal GPU timeout on DeepSeek-V4-Flash at ~4.5k
# tokens in pipeline mode; 512 avoids it and is also ~2.6x faster.
PREFILL_STEP = 512


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- prompt cache
#
# DeepSeek-V4's cache (sliding window + compressed pooling layers) cannot be
# trimmed back to an earlier length, so mlx-vlm's own turn-to-turn reuse almost
# never applies. Instead we take snapshots of the cache during prefill, before any
# generated token goes in: one at the end of every prompt, plus checkpoints at
# turn boundaries inside long prompts. A request reuses the longest snapshot whose
# tokens are an exact prefix of its own.
#
# Several snapshots are kept (agents interleave side requests with the main
# conversation, and sometimes rewrite earlier turns). Eviction is least-recently
# used under a token budget. Everything here depends only on token ids and the
# command line, which are identical on every rank, so all ranks keep, pick and
# evict the same snapshots and stay in step.

SNAPS = []        # [{"ids", "cache", "used", "nbytes"}], all deep copies
_tick = [0]       # LRU clock
_req = {}         # per-request bookkeeping for the hooks below
BOUNDARY_IDS = set()   # token ids that start a turn; checkpoints go just before them
CACHE_MAX_SNAPS = 16
CACHE_BUDGET_TOKENS = 600_000
CACHE_CKPT_GAP = 4096  # minimum tokens between checkpoints inside one prompt
CACHE_CKPT_MAX = 4     # most checkpoints per prompt, besides the final one
_orig_prepare = _dispatch._prepare_generation_inputs
_orig_generate_step = _dispatch.generate_step


def _nbytes(cache):
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten([c.state for c in cache]) if hasattr(v, "nbytes"))


def cache_stats():
    return {"snapshots": len(SNAPS), "tokens": sum(len(x["ids"]) for x in SNAPS),
            "gib": round(sum(x["nbytes"] for x in SNAPS) / 2**30, 3),
            "longest": max((len(x["ids"]) for x in SNAPS), default=0),
            "active_gib": round(mx.get_active_memory() / 2**30, 2),
            "peak_gib": round(mx.get_peak_memory() / 2**30, 2)}


def _prepare_hook(*a, **kw):
    out = _orig_prepare(*a, **kw)
    if _req.get("active"):
        ids = out[0].flatten().tolist()
        _req["full_ids"] = ids
        best = None
        for snap in SNAPS:
            n = len(snap["ids"])
            if n < len(ids) and (best is None or n > len(best["ids"])) and ids[:n] == snap["ids"]:
                best = snap
        pcs = _req["pcs"]
        if best is not None:
            _tick[0] += 1
            best["used"] = _tick[0]
            # Hand over a copy: generation writes into it, and the snapshot must
            # stay intact for the next request that shares this prefix.
            pcs.token_ids = best["ids"]
            pcs.cache = copy.deepcopy(best["cache"])
            mx.eval([c.state for c in pcs.cache])
            _req["reused"] = len(best["ids"])
        else:
            pcs.cache = None
    return out


def _add_snapshot(ids, cache):
    snap = copy.deepcopy(cache)
    mx.eval([c.state for c in snap])
    _tick[0] += 1
    SNAPS[:] = [x for x in SNAPS if x["ids"] != ids]
    SNAPS.append({"ids": ids, "cache": snap, "used": _tick[0], "nbytes": _nbytes(snap)})
    newest = SNAPS[-1]
    while len(SNAPS) > 1 and (len(SNAPS) > CACHE_MAX_SNAPS or
                              sum(len(x["ids"]) for x in SNAPS) > CACHE_BUDGET_TOKENS):
        SNAPS.remove(min((x for x in SNAPS if x is not newest), key=lambda x: x["used"]))


def _snapshot(prefix_len, prompt_cache):
    _add_snapshot(_req["full_ids"][: _req["reused"] + prefix_len], prompt_cache)
    _req["snapshots"] += 1


def _checkpoint_lengths(full, reused, n_input):
    """Where to snapshot, relative to the tokens actually being prefilled."""
    final = n_input - 1  # all but the last prompt token, which _step processes
    if final <= 0:
        return []
    picks, last = [], reused + final
    for p in range(reused + final - 1, reused, -1):  # walk back from the end
        if len(picks) >= CACHE_CKPT_MAX:
            break
        if full[p] in BOUNDARY_IDS and last - p >= CACHE_CKPT_GAP and p - reused >= CACHE_CKPT_GAP // 2:
            picks.append(p)
            last = p
    return sorted(p - reused for p in picks) + [final]


def _generate_step_hook(input_ids, *a, **kw):
    if _req.get("active"):
        pts = _checkpoint_lengths(_req["full_ids"], _req["reused"], int(input_ids.shape[1]))
        if pts:
            kw["prompt_cache_checkpoint"] = _snapshot
            kw["prompt_cache_checkpoint_len"] = pts[-1]
            kw["prompt_cache_checkpoint_lengths"] = pts
    return _orig_generate_step(input_ids, *a, **kw)


_dispatch._prepare_generation_inputs = _prepare_hook
_dispatch.generate_step = _generate_step_hook


# ---------------------------------------------------------------- cluster sync

def broadcast(world, obj=None):
    """Send a JSON-serialisable object from rank 0 to every rank.

    mx.distributed has no broadcast, so rank 0 contributes the bytes to an
    all_sum and every other rank contributes zeros.
    """
    if world.size() == 1:
        return obj
    data = json.dumps(obj).encode() if world.rank() == 0 else b""
    n = mx.distributed.all_sum(mx.array([len(data)], dtype=mx.int32), group=world, stream=mx.cpu)
    n = int(n.item())
    if world.rank() == 0:
        buf = mx.array(np.frombuffer(data, dtype=np.uint8).astype(np.int32))
    else:
        buf = mx.zeros((n,), dtype=mx.int32)
    buf = mx.distributed.all_sum(buf, group=world, stream=mx.cpu)
    return json.loads(np.array(buf).astype(np.uint8).tobytes().decode())


def run_job(model, processor, job, on_chunk):
    """Generate for one job. Runs identically on every rank."""
    mx.random.seed(job["seed"])  # same seed everywhere, so sampled tokens match
    last = None
    marker = job.get("stop_marker")
    text = ""
    use_cache = job.get("cache", True) and PROMPT_CACHE
    pcs = PromptCacheState() if use_cache else None  # filled in by _prepare_hook
    _req.clear()
    _req.update(active=use_cache, pcs=pcs, reused=0, snapshots=0)
    extra = {"prompt_cache_state": pcs} if pcs is not None else {}
    try:
        for r in stream_generate(
            model,
            processor,
            job["prompt"],
            max_tokens=job["max_tokens"],
            temperature=job["temperature"],
            top_p=job["top_p"],
            prefill_step_size=job.get("prefill_step", PREFILL_STEP),
            **extra,
        ):
            last = r
            on_chunk(r)
            if marker:
                # Every rank sees the same tokens, so every rank stops on the same one.
                # In thinking mode, only the answer after </think> counts.
                text += r.text
                tail = text[text.find(THINK_END):] if job.get("think") else text
                if (not job.get("think") or THINK_END in text) and marker in tail:
                    break
    finally:
        if sys.exc_info()[0]:
            SNAPS.clear()  # after a failure, trust nothing
        job["_reused"] = _req.get("reused", 0)
        job["_new_snaps"] = _req.get("snapshots", 0)
        _req.clear()
    return last


# ------------------------------------------------------------ request parsing

class BadRequest(Exception):
    pass


def flatten_content(content):
    """OpenAI content may be a string or a list of parts. Keep the text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                parts.append(p.get("text", ""))
            elif isinstance(p, dict):
                raise BadRequest(f"unsupported content part type: {p.get('type')!r} (text only)")
        return "\n".join(parts)
    raise BadRequest("message content must be a string or a list of parts")


TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
# Stop on the closing tag's stem: the model sometimes writes </tool_calls>.
TOOL_STOP = "</tool_call"


def tools_prompt(tools):
    specs = []
    for t in tools:
        f = t.get("function", t)
        specs.append({"name": f.get("name"), "description": f.get("description", ""),
                      "parameters": f.get("parameters", {})})
    return (
        "# Tools\n\nYou can call these tools. Their JSON schemas:\n\n"
        + "\n".join(json.dumps(x, ensure_ascii=False) for x in specs)
        + "\n\nTo call a tool, reply with optional brief text, then exactly one block:\n"
        f"{TOOL_OPEN}\n{{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n{TOOL_CLOSE}\n"
        "The arguments must be valid JSON matching the schema. Call one tool per reply "
        "and stop right after the block; the result comes back in the next message as "
        "<tool_result>. When no tool is needed, reply normally without a block."
    )


def parse_tool_call(text):
    """Return (content before the call, [openai tool_call]) or (text, [])."""
    i = text.find(TOOL_OPEN)
    if i == -1:
        return text, []
    raw = text[i + len(TOOL_OPEN):].lstrip()
    if raw.startswith("```"):
        raw = raw[3:].removeprefix("json").lstrip()
    try:
        # Read one JSON object and ignore whatever closes it (</tool_call>,
        # </tool_calls>, a code fence, nothing at all).
        obj, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        return text, []  # malformed: hand the raw text back so the client sees it
    if not isinstance(obj, dict) or not obj.get("name"):
        return text, []
    args = obj.get("arguments", obj.get("parameters", {}))
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    call = {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
            "function": {"name": obj.get("name", ""), "arguments": args}}
    return text[:i].rstrip(), [call]


DSML_OPEN = "<｜DSML｜tool_calls"     # the model writes "\n\n" + this + ">\n<｜DSML｜invoke ..."
DSML_END = "</｜DSML｜tool_calls>"
DSML_STOP = "</｜DSML｜tool_calls"


def is_dsv4(config):
    return (config or {}).get("model_type") == "deepseek_v4"


def reasoning_mode(body):
    """Return (thinking?, effort for the encoder) from an OpenAI-style request.

    Accepts `reasoning_effort` (OpenAI / Cline), `reasoning: {"effort": ...}`, and
    `chat_template_kwargs: {"thinking": bool}` (vLLM / SGLang style). Falls back
    to --default-reasoning-effort.
    """
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    kw = body.get("chat_template_kwargs") or {}
    if effort is None and isinstance(kw, dict):
        flag = kw.get("thinking", kw.get("enable_thinking"))
        if flag is not None:
            effort = "high" if flag else "none"
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    effort = str(effort).lower()
    if effort in ("none", "minimal", "off", "false"):
        return False, None
    if effort not in ("low", "medium", "high", "max", "xhigh"):
        raise BadRequest(f"unsupported reasoning_effort: {effort!r}")
    # The vendored preview encoder knows one effort prompt, which it calls "max"
    # (the 0731 release encoder renames that text "high" and adds a stronger
    # "max"). Map the top levels onto it; everything else thinks with no prompt.
    return True, ("max" if effort in ("max", "xhigh") else None)


def _strip_think(content):
    """Drop <think>…</think> a client may have left inside an earlier assistant turn."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.replace("<think>", "").replace("</think>", "")


def dsv4_prompt(messages, tools, thinking=False, effort=None):
    """Build the prompt with DeepSeek's encoder, from OpenAI-format messages."""
    msgs = []
    for m in messages:
        role = m.get("role")
        role = {"developer": "system", "function": "tool"}.get(role, role)
        if role not in ("system", "user", "assistant", "tool"):
            raise BadRequest(f"unsupported role: {role!r}")
        nm = {"role": role, "content": flatten_content(m.get("content"))}
        if role == "assistant":
            nm["content"] = _strip_think(nm["content"])
            if m.get("reasoning_content"):
                # The encoder keeps or drops earlier reasoning by its own rules.
                nm["reasoning_content"] = m["reasoning_content"]
        if role == "assistant" and m.get("tool_calls"):
            nm["tool_calls"] = []
            for c in m["tool_calls"]:
                f = c.get("function", {})
                args = f.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {}, ensure_ascii=False)
                nm["tool_calls"].append({"id": c.get("id", ""), "type": "function",
                                         "function": {"name": f.get("name", ""), "arguments": args}})
        if role == "tool":
            nm["tool_call_id"] = m.get("tool_call_id", "")
        msgs.append(nm)
    if tools:
        tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        for nm in msgs:
            if nm["role"] == "system":
                nm["tools"] = tools
                break
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    return dsv4_encoding.encode_messages(
        msgs, thinking_mode="thinking" if thinking else "chat", reasoning_effort=effort)


def parse_dsml(text):
    """Return (content before the calls, [openai tool_call, ...]) or (text, [])."""
    i = text.find(DSML_OPEN)
    if i == -1:
        return text, []
    body = text[i + len(DSML_OPEN):]
    j = body.find(DSML_STOP)
    # Generation stops on the closing tag's stem, so put the full tag back.
    body = (body[:j] if j != -1 else body) + DSML_END
    try:
        _, _, calls = dsv4_encoding.parse_tool_calls(0, body)
    except Exception as e:
        log(f"DSML tool-call parse failed ({e}); returning raw text")
        return text, []
    if not calls:
        return text, []
    return text[:i].rstrip(), [
        {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
         "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]


def build_job(body, processor, config):
    tools = body.get("tools") or [{"type": "function", "function": f}
                                  for f in (body.get("functions") or [])]
    if body.get("tool_choice") == "none":
        tools = []
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("`messages` must be a non-empty list")
    if is_dsv4(config):
        thinking, effort = reasoning_mode(body)
        return finish_job(body, dsv4_prompt(messages, tools, thinking, effort), tools, "dsml",
                          think=thinking)
    clean = []
    call_names = {}  # tool_call_id -> tool name, so results can be labelled
    for m in messages:
        role = m.get("role")
        content = flatten_content(m.get("content"))
        if role in ("system", "developer"):
            role = "system"
        elif role == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                f = c.get("function", {})
                call_names[c.get("id")] = f.get("name", "")
                try:
                    a = json.loads(f.get("arguments") or "{}")
                except json.JSONDecodeError:
                    a = f.get("arguments")
                content += ("\n" if content else "") + TOOL_OPEN + "\n" + json.dumps(
                    {"name": f.get("name"), "arguments": a}, ensure_ascii=False) + "\n" + TOOL_CLOSE
        elif role in ("tool", "function"):
            role = "user"
            content = f"<tool_result name=\"{m.get('name') or call_names.get(m.get('tool_call_id'), '')}\">\n{content}\n</tool_result>"
        elif role not in ("user", "assistant"):
            raise BadRequest(f"unsupported role: {role!r}")
        if clean and clean[-1]["role"] == role and role != "assistant":
            clean[-1]["content"] += "\n\n" + content  # merge consecutive user/system turns
        else:
            clean.append({"role": role, "content": content})
    if tools:
        block = tools_prompt(tools)
        if clean[0]["role"] == "system":
            clean[0]["content"] += "\n\n" + block
        else:
            clean.insert(0, {"role": "system", "content": block})

    prompt = apply_chat_template(processor, config, clean)
    return finish_job(body, prompt, tools, "json")


def finish_job(body, prompt, tools, tool_format, think=False):
    temperature, top_p = body.get("temperature"), body.get("top_p")
    if think:
        # DeepSeek's recommendation: temperature 1.0, top_p 0.95 for agent work.
        # Greedy decoding tends to loop in long reasoning.
        temperature = 1.0 if temperature is None else temperature
        top_p = (0.95 if tools else 1.0) if top_p is None else top_p
    return {
        "prompt": prompt,
        "think": think,
        "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens")
                          or (THINK_MAX_TOKENS if think else DEFAULT_MAX_TOKENS)),
        "temperature": 0.0 if temperature is None else float(temperature),
        "top_p": 1.0 if top_p is None else float(top_p),
        "seed": int(body["seed"]) if body.get("seed") is not None else random.randint(0, 2**31 - 1),
        "tool_format": tool_format,
        "tool_open": (DSML_OPEN if tool_format == "dsml" else TOOL_OPEN) if tools else None,
        "stop_marker": (DSML_STOP if tool_format == "dsml" else TOOL_STOP) if tools else None,
        "cache": body.get("cache", True) is not False,  # {"cache": false} forces a cold read
    }


def split_think(raw, think, final=False):
    """Split generated text into (reasoning, answer, reasoning finished?).

    In thinking mode the prompt ends with "<think>", so the model writes its
    reasoning, then "</think>", then the answer (and any tool calls). The preview
    model sometimes closes its reasoning, keeps reasoning, and closes it again,
    so the final split uses the LAST "</think>"; while streaming, the first one
    is the earliest point the reasoning can have ended.
    """
    if not think:
        return "", raw, True
    i = raw.rfind(THINK_END) if final else raw.find(THINK_END)
    if i == -1:
        for k in range(len(THINK_END) - 1, 0, -1):  # hold back a partial "</think"
            if raw.endswith(THINK_END[:k]):
                return raw[:-k], "", False
        return raw, "", False
    return raw[:i], raw[i + len(THINK_END):].lstrip("\n"), True


def apply_stop(text, stops):
    """Return (text cut at the first stop string, whether a stop was hit)."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut], cut < len(text)


# --------------------------------------------------------------------- server

def make_handler(world, model, processor, config, model_id):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter default logging
            log(self.address_string(), fmt % args)

        def _json(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status, message):
            self._json(status, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok", "nodes": world.size(), "model": model_id,
                                 "prompt_cache": cache_stats()})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}]})
            else:
                self._error(404, "not found")

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._error(404, "not found")
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                job = build_job(body, processor, config)
            except BadRequest as e:
                return self._error(400, str(e))
            except Exception as e:
                return self._error(400, f"bad request: {e}")

            stream = bool(body.get("stream"))
            stops = body.get("stop") or []
            stops = [stops] if isinstance(stops, str) else list(stops)
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            broadcast(world, job)  # every rank now starts the same job

            state = {"text": "", "sent": 0, "rsent": 0, "stopped": False, "client_gone": False}

            def send_sse(obj):
                if state["client_gone"]:
                    return
                try:
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    state["client_gone"] = True  # keep generating so ranks stay in step

            def chunk_obj(delta, finish=None):
                return {"id": rid, "object": "chat.completion.chunk", "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                send_sse(chunk_obj({"role": "assistant", "content": ""}))

            def on_chunk(r):
                if state["stopped"]:
                    return
                state["text"] += r.text
                reasoning, src, _ = split_think(state["text"], job["think"])
                if stream and len(reasoning) > state["rsent"]:
                    send_sse(chunk_obj({"reasoning_content": reasoning[state["rsent"]:]}))
                state["rsent"] = len(reasoning)
                if job["think"]:
                    # Stream the reasoning live, but hold the answer until the end:
                    # a second "</think>" can still turn what follows the first
                    # one back into reasoning.
                    return
                # Stops and tool calls apply to the answer only, never to the reasoning.
                topen = job["tool_open"]
                visible, hit = apply_stop(src, stops + ([topen] if topen else []))
                if topen and not hit:
                    # don't stream a partial opening tag that may still be completing
                    for k in range(len(topen) - 1, 0, -1):
                        if visible.endswith(topen[:k]):
                            visible = visible[:-k]
                            break
                    hit = False
                if topen and topen in src:
                    visible = visible.rstrip()  # the blank line before the calls block
                if hit and not (topen and topen in src
                                and not any(st in src for st in stops)):
                    state["stopped"] = True
                if stream and len(visible) > state["sent"]:
                    send_sse(chunk_obj({"content": visible[state["sent"]:]}))
                state["sent"] = len(visible)

            t0 = time.perf_counter()
            last = run_job(model, processor, job, on_chunk)
            elapsed = time.perf_counter() - t0
            reasoning, src, think_done = split_think(state["text"], job["think"], final=True)
            if job["think"] and not think_done:
                reasoning = state["text"]  # ran out of tokens mid-thought: it is all reasoning
            text, _ = apply_stop(src, stops)
            tool_calls = []
            if job["tool_open"]:
                text, tool_calls = (parse_dsml if job["tool_format"] == "dsml" else parse_tool_call)(text)

            if tool_calls:
                finish = "tool_calls"
            elif state["stopped"]:
                finish = "stop"
            elif last is not None and last.generation_tokens >= job["max_tokens"]:
                finish = "length"
            else:
                finish = "stop"
            usage = {
                "prompt_tokens": last.prompt_tokens if last else 0,
                "completion_tokens": last.generation_tokens if last else 0,
                "total_tokens": (last.prompt_tokens + last.generation_tokens) if last else 0,
            }
            log(f"{rid}: {f'[think {len(reasoning)} chars] ' if job['think'] else ''}"
                f"{'[tool ' + ', '.join(c['function']['name'] for c in tool_calls) + '] ' if tool_calls else ''}"
                f"{usage['prompt_tokens']} prompt ({job.get('_reused', 0)} cached) + {usage['completion_tokens']} gen tokens "
                f"in {elapsed:.1f}s (prefill {last.prompt_tps if last else 0:.0f} tok/s, "
                f"decode {last.generation_tps if last else 0:.1f} tok/s){' [client gone]' if state['client_gone'] else ''}"
                f" | cache: +{job.get('_new_snaps', 0)} -> {cache_stats()}")

            if stream:
                if len(reasoning) > state["rsent"]:  # anything held back at the end
                    send_sse(chunk_obj({"reasoning_content": reasoning[state["rsent"]:]}))
                if len(text) > state["sent"] and (job["think"] or (job["tool_open"] and not tool_calls)):
                    # thinking: the whole answer (held until now); otherwise any unparsed remainder
                    send_sse(chunk_obj({"content": text[state["sent"]:]}))
                if tool_calls:
                    send_sse(chunk_obj({"tool_calls": [dict(c, index=i) for i, c in enumerate(tool_calls)]}))
                final = chunk_obj({}, finish)
                final["usage"] = usage
                send_sse(final)
                if not state["client_gone"]:
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                self.close_connection = True
            else:
                self._json(200, {
                    "id": rid, "object": "chat.completion", "created": created, "model": model_id,
                    "choices": [{"index": 0, "message": dict(
                                     {"role": "assistant", "content": text or (None if tool_calls else "")},
                                     **({"reasoning_content": reasoning} if job["think"] else {}),
                                     **({"tool_calls": tool_calls} if tool_calls else {})),
                                 "finish_reason": finish}],
                    "usage": usage,
                })

    return Handler


def main():
    global PREFILL_STEP, PROMPT_CACHE, CACHE_MAX_SNAPS, CACHE_BUDGET_TOKENS, CACHE_CKPT_GAP
    global DEFAULT_REASONING_EFFORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--prefill-step", type=int, default=PREFILL_STEP)
    ap.add_argument("--no-prompt-cache", action="store_true")
    ap.add_argument("--default-reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                    choices=["none", "low", "medium", "high", "max"],
                    help="thinking mode for requests that don't ask (DeepSeek-V4 only)")
    ap.add_argument("--cache-max-snapshots", type=int, default=CACHE_MAX_SNAPS)
    ap.add_argument("--cache-budget-tokens", type=int, default=CACHE_BUDGET_TOKENS)
    ap.add_argument("--cache-checkpoint-gap", type=int, default=CACHE_CKPT_GAP)
    args = ap.parse_args()
    PREFILL_STEP = args.prefill_step
    PROMPT_CACHE = not args.no_prompt_cache
    DEFAULT_REASONING_EFFORT = args.default_reasoning_effort
    CACHE_MAX_SNAPS = args.cache_max_snapshots
    CACHE_BUDGET_TOKENS = args.cache_budget_tokens
    CACHE_CKPT_GAP = args.cache_checkpoint_gap

    world = mx.distributed.init()
    rank = world.rank()
    t0 = time.perf_counter()
    if world.size() > 1:
        model, processor = sharded_load(args.model, pipeline_group=world)
    else:
        model, processor = load(args.model)
    config = model.config.to_dict() if hasattr(model.config, "to_dict") else model.config
    if is_dsv4(config):
        # Checkpoints go just before each user turn (tool results are user turns too).
        tok = getattr(processor, "tokenizer", processor)
        uid = tok.convert_tokens_to_ids(dsv4_encoding.USER_SP_TOKEN)
        if isinstance(uid, int) and uid != getattr(tok, "unk_token_id", None):
            BOUNDARY_IDS.add(uid)
    log(f"prompt cache: {'on' if PROMPT_CACHE else 'off'}, {CACHE_MAX_SNAPS} snapshots, "
        f"{CACHE_BUDGET_TOKENS} token budget, checkpoint ids {sorted(BOUNDARY_IDS)}")
    log(f"rank {rank}/{world.size()}: loaded {args.model} in {time.perf_counter() - t0:.0f}s, "
        f"{mx.get_active_memory() / 2**30:.1f} GiB active")

    if rank != 0:
        # Worker: wait for jobs from rank 0 and generate in lockstep, forever.
        while True:
            job = broadcast(world)
            if job.get("cmd") == "shutdown":
                break
            run_job(model, processor, job, lambda r: None)
            log(f"rank {rank}: job done, {job.get('_reused', 0)} cached, cache {cache_stats()}")
        return

    server = HTTPServer((args.host, args.port), make_handler(world, model, processor, config, args.model))
    log(f"serving on http://{args.host}:{args.port}/v1  ({world.size()} node(s))")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        broadcast(world, {"cmd": "shutdown"})


if __name__ == "__main__":
    main()
