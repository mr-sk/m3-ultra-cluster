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
import contextlib
import copy
import json
import os
import random
import re
import socket
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import queue as _queue
import collections as _collections
import threading
from types import SimpleNamespace

# macOS drops GPU residency after ~2 s idle (a macOS bug; ml-explore/mlx#4633).
# This makes MLX re-request it every second. MLX reads it when it creates the
# Metal device, so it is set before mlx is imported. It only covers memory inside
# the wired limit, which pin_weights() holds raised. Set it yourself to override.
os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "1000")

import mlx.core as mx
import dsv4_encoding  # DeepSeek's reference DeepSeek-V4 encoder, vendored (MIT)
import numpy as np
from mlx_vlm import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load, sharded_load
from mlx_vlm.models import cache as _vlm_cache
from mlx_vlm.models.base import create_attention_mask as _create_attention_mask
from mlx_vlm.models.cache import CacheList as _CacheList
from mlx_vlm.models.linear import linear as _linear
from mlx_vlm.sample_utils import make_sampler as _make_sampler
from mlx_vlm.sample_utils import apply_top_p as _apply_top_p
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.tokenizer_utils import make_streaming_detokenizer as _make_detokenizer
import importlib
# mlx_vlm rebinds the name `mlx_vlm.generate` to a function, so load by full name.
_dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
PromptCacheState = importlib.import_module("mlx_vlm.generate.common").PromptCacheState

DEFAULT_MAX_TOKENS = 8192  # Cline sends no max_tokens; 2048 cut its answers off
THINK_MAX_TOKENS = 32768   # default when thinking: the reasoning alone can run to thousands of tokens
DEFAULT_REASONING_EFFORT = "none"  # --default-reasoning-effort; "none" = chat mode
THINK_END = "</think>"
THINK_START = "<think>"
# Reasoning budget per effort level, in tokens. When it runs out, mlx-vlm forces
# "\n</think>" so the model must move on to its answer. Seen 2026-11-08: a Cline
# turn reasoned for 8,488 tokens (32k chars, 4.5 min) and Cline timed out.
THINK_BUDGETS = {"low": 1024, "medium": 2048, "high": 4096, "max": 12288, "xhigh": 12288}
# Loop guard: stop when the last LOOP_TAIL characters already appeared LOOP_REPEATS
# times within the last LOOP_WINDOW characters. Text-only, so every rank stops on
# the same token. Seen 2026-11-08: a 382-token side request ran to 8,192 tokens.
LOOP_TAIL, LOOP_REPEATS, LOOP_WINDOW, LOOP_EVERY = 200, 4, 6000, 32


def looping(text):
    if len(text) < LOOP_TAIL * LOOP_REPEATS:
        return False
    tail = text[-LOOP_TAIL:]
    return text[-LOOP_WINDOW:].count(tail) >= LOOP_REPEATS
# Status lines the server adds to the reasoning stream (Cline shows them in its
# "Thinking" section). Marked so they can be stripped from history before the
# model ever sees them. --no-stats-in-reasoning turns them off.
STATS_IN_REASONING = True
STATS_MARK = "\u27e6cluster\u27e7"   # ⟦cluster⟧
PREFILL_EST = [730.0]                 # new-token prefill rate, tok/s; updated as we go


def _fmt_secs(t):
    return f"{t:.0f}s" if t < 90 else f"{t / 60:.1f} min"


def strip_stats(text):
    """Remove the server's own status lines from text a client sent back."""
    if not text or STATS_MARK not in text:
        return text
    return re.sub(r"[^\n]*" + re.escape(STATS_MARK) + r"[^\n]*\n?", "", text).strip()


def stats_start_line(total, reused):
    new = total - reused
    eta = new / PREFILL_EST[0]
    est = f" \u00b7 ~{_fmt_secs(eta)} to read" if eta >= 3 else ""
    return f"{STATS_MARK} reading {new:,} new token{'' if new == 1 else 's'} ({reused:,} cached){est}\n\n"


def stats_end_line(prompt_tokens, reused, gen_tokens, prompt_tps, gen_tps, elapsed, cs, lead="\n\n"):
    new = prompt_tokens - reused
    read_s = prompt_tokens / prompt_tps if prompt_tps else 0.0
    return (f"{lead}{STATS_MARK} {gen_tokens:,} tokens at {gen_tps:.1f} tok/s \u00b7 "
            f"read {new:,} new + {reused:,} cached in {_fmt_secs(read_s)} \u00b7 "
            f"{_fmt_secs(elapsed)} total \u00b7 cache {cs['snapshots']} saved "
            f"({cs['tokens'] / 1000:.0f}k tok) \u00b7 peak {cs['peak_gib']} GiB")


def update_prefill_estimate(prompt_tokens, reused, prompt_tps):
    """Learn the cold prefill rate from turns that read a lot of new tokens."""
    new = prompt_tokens - reused
    if new >= 2048 and prompt_tps:
        rate = new / (prompt_tokens / prompt_tps)
        PREFILL_EST[0] = 0.7 * PREFILL_EST[0] + 0.3 * rate
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
    if _req.get("active") or _req.get("on_start"):
        ids = out[0].flatten().tolist()
        _req["full_ids"] = ids
    if _req.get("active"):
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
    if _req.get("on_start"):
        _req["on_start"](len(_req["full_ids"]), _req.get("reused", 0))
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
    if WAKE is not None:
        if world.rank() == 0:
            WAKE.signal()
        elif not WAKE.wait():
            log(f"rank {world.rank()}: rank 0 closed the wake socket, shutting down")
            return {"cmd": "shutdown"}
    data = json.dumps(obj).encode() if world.rank() == 0 else b""
    n = mx.distributed.all_sum(mx.array([len(data)], dtype=mx.int32), group=world, stream=mx.cpu)
    n = int(n.item())
    if world.rank() == 0:
        buf = mx.array(np.frombuffer(data, dtype=np.uint8).astype(np.int32))
    else:
        buf = mx.zeros((n,), dtype=mx.int32)
    buf = mx.distributed.all_sum(buf, group=world, stream=mx.cpu)
    return json.loads(np.array(buf).astype(np.uint8).tobytes().decode())


# Wake socket: lets the other ranks wait for a job without spinning a core.
# broadcast() is a CPU all_sum, and JACCL waits for one by polling the
# Thunderbolt RDMA completion queue, so a rank waiting for the next job held
# one core at 100% for as long as the server was idle (seen 2026-11-13). Rank 0
# now sends one byte over TCP before each broadcast; the others block on it,
# at no CPU, and only then enter the all_sum. If rank 0 dies, they see the
# socket close and shut down instead of polling forever.
WAKE = None
WAKE_PORT = 8095
WAKE_HELLO = b"dsv4-wake\n"


def wake_address(env, port):
    """(rank 0's host, port), from the coordinator address mlx.launch sets."""
    coord = env.get("MLX_JACCL_COORDINATOR")
    if not coord:
        return None
    return coord.rsplit(":", 1)[0], port


class Wake:
    def __init__(self, socks):
        self.socks = socks

    @staticmethod
    def _tune(s):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.settimeout(None)
        return s

    @classmethod
    def listen(cls, port, n, host="0.0.0.0", timeout=600):
        """Rank 0: accept n workers. Connections without the hello are dropped."""
        srv = socket.create_server((host, port))
        srv.settimeout(timeout)
        socks = []
        try:
            while len(socks) < n:
                s, _ = srv.accept()
                s.settimeout(5)
                try:
                    hello = s.recv(len(WAKE_HELLO))
                except OSError:
                    hello = b""
                if hello == WAKE_HELLO:
                    socks.append(cls._tune(s))
                else:
                    s.close()
        finally:
            srv.close()
        return cls(socks)

    @classmethod
    def connect(cls, host, port, rank, timeout=600):
        """A worker: connect to rank 0, retrying until it is listening."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                s = socket.create_connection((host, port), timeout=5)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        s.sendall(WAKE_HELLO)
        return cls([cls._tune(s)])

    def signal(self):
        for s in self.socks:
            s.sendall(b"j")

    def wait(self):
        """Block until rank 0 signals (True) or closes the socket (False)."""
        try:
            return self.socks[0].recv(1) == b"j"
        except OSError:
            return False

    def close(self):
        for s in self.socks:
            s.close()


# Page the weights back in after an idle gap: a backstop for pin_weights().
# macOS unwires idle GPU memory within seconds (even with mx.set_wired_limit
# raised, unless MLX's residency refresh is on; measured on macOS 27.2)
# and may then compress or swap it. The first job after a long idle used to page
# it back in inside GPU work while the other rank's GPU waited on this one, and
# the ~5 s GPU watchdog killed rank 1 (kIOGPUCommandBufferCallbackErrorTimeout,
# 2026-11-09 after 12 h idle, 2026-11-11 after ~2 days). Rank 0 decides and sends
# the decision with the job, so every rank does the same thing.
PREFAULT_IDLE = 60.0  # seconds idle before a job prefaults; 0 = every job, <0 = never
LAST_JOB_END = None   # time.monotonic() when the last job finished (rank 0 decides)
WORLD = None


def needs_prefault(now, last_end, idle=None):
    idle = PREFAULT_IDLE if idle is None else idle
    if idle < 0:
        return False
    return last_end is None or now - last_end >= idle


def cpu_barrier():
    """Wait for every rank on the CPU, where the GPU watchdog cannot see the wait."""
    if WORLD is not None and WORLD.size() > 1:
        mx.eval(mx.distributed.all_sum(mx.array([1]), group=WORLD, stream=mx.cpu))


def prefault(model, barrier=None):
    """Touch every weight on this rank, one tensor per GPU command buffer, so no
    single buffer has to page in more than one tensor. Then wait for the other
    ranks, so no rank's GPU sits waiting on a rank that is still paging in.
    Returns (tensors touched, seconds touching, seconds waiting for the others)."""
    from mlx.utils import tree_flatten
    t = time.perf_counter()
    params = tree_flatten(model.parameters())
    for _, p in params:
        mx.eval(p.sum())
    touched = time.perf_counter()
    if barrier is not None:
        barrier()
    return len(params), touched - t, time.perf_counter() - touched


def run_job(model, processor, job, on_chunk, on_start=None):
    """Generate for one job. Runs identically on every rank."""
    global LAST_JOB_END
    if job.get("prefault"):
        n, dt, waited = prefault(model, cpu_barrier)
        log(f"rank {WORLD.rank() if WORLD else 0}: paged in {n} weight tensors in {dt:.1f}s, "
            f"then waited {waited:.1f}s for the other ranks")
    mx.random.seed(job["seed"])  # same seed everywhere, so sampled tokens match
    last = None
    marker = job.get("stop_marker")
    text = ""
    use_cache = job.get("cache", True) and PROMPT_CACHE
    pcs = PromptCacheState() if use_cache else None  # filled in by _prepare_hook
    _req.clear()
    _req.update(active=use_cache, pcs=pcs, reused=0, snapshots=0, on_start=on_start)
    extra = {"prompt_cache_state": pcs} if pcs is not None else {}
    if job.get("think") and job.get("think_budget"):
        extra.update(thinking_budget=job["think_budget"], enable_thinking=True,
                     thinking_start_token=THINK_START, thinking_end_token=THINK_END)
    n_tok = 0
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
            # Every rank sees the same tokens, so every rank stops on the same one.
            text += r.text
            n_tok += 1
            if marker and marker in text:
                # Includes a tool call written inside unfinished reasoning: the
                # preview model sometimes calls a tool without closing </think>.
                break
            if n_tok % LOOP_EVERY == 0 and looping(text):
                job["_looped"] = True
                break
    finally:
        if sys.exc_info()[0]:
            SNAPS.clear()  # after a failure, trust nothing
        job["_reused"] = _req.get("reused", 0)
        job["_text"] = text
        job["_new_snaps"] = _req.get("snapshots", 0)
        _req.clear()
        LAST_JOB_END = time.monotonic()
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
        return False, None, None
    if effort not in THINK_BUDGETS:
        raise BadRequest(f"unsupported reasoning_effort: {effort!r}")
    budget = int(body.get("thinking_budget") or THINK_BUDGETS[effort])
    # The vendored preview encoder knows one effort prompt, which it calls "max"
    # (the 0731 release encoder renames that text "high" and adds a stronger
    # "max"). Map the top levels onto it; everything else thinks with no prompt.
    return True, ("max" if effort in ("max", "xhigh") else None), budget


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
                rc = strip_stats(m["reasoning_content"])
                if rc:
                    nm["reasoning_content"] = rc
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


_dsml_fixups = []  # names of arguments rescued by the lenient decoder, per parse


def _lenient_decode(tool_name, tool_args):
    """DeepSeek's decode_dsml_to_arguments, tolerant of mislabelled values.

    The reference decoder pastes string="false" values into the JSON verbatim.
    When the model labels code as string="false", the result is invalid JSON
    and clients drop the argument (Cline: "expected string, received undefined
    at new_text"). Values that don't parse as JSON are kept as strings.
    """
    args = {}
    for key, (value, is_str) in tool_args.items():
        if is_str == "true":
            args[key] = value
            continue
        try:
            args[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            args[key] = value
            _dsml_fixups.append(key)
    return dict(name=tool_name, arguments=json.dumps(args, ensure_ascii=False))


dsv4_encoding.decode_dsml_to_arguments = _lenient_decode  # parse_tool_calls looks it up at call time


def check_required(tool_calls, tools, raw=""):
    """Log required arguments the model left out. Deliberately does NOT fill them.

    Seen 2026-11-08: DeepSeek called Cline's editor with path and old_text but no
    new_text. Filling new_text with "" (tried briefly) turns that into a
    deletion: in one case the model had put its NEW text in old_text, so a
    match would have deleted tests. A client-side validation error is the safe
    failure: the model sees it and retries.
    """
    schemas = {}
    for t in tools or []:
        f = t.get("function", t)
        schemas[f.get("name")] = f.get("parameters") or {}
    for c in tool_calls:
        sch = schemas.get(c["function"]["name"]) or {}
        props, req = sch.get("properties") or {}, sch.get("required") or []
        try:
            args = json.loads(c["function"]["arguments"])
        except json.JSONDecodeError:
            continue
        missing = [k for k in req if k not in args]
        if not missing:
            continue
        log(f"tool call {c['function']['name']}: missing required {missing} (left for the client "
            f"to reject); raw tail {raw[-300:]!r}")
    return tool_calls


_DSML_PARAM = re.compile(
    r'<｜DSML｜parameter name="([^"]*)" string="(true|false)">(.*?)</｜DSML｜parameter>\n?', re.S)


def _drop_identical_duplicates(body):
    """Remove repeats of a parameter that has the SAME value within one invoke.

    After being told it missed `path`, DeepSeek sent `path` twice with the same
    value (2026-11-08). The reference parser rejects any duplicate, so a complete
    edit reached the client as plain text and ended the task. Repeats with a
    DIFFERENT value are left in place: the parser still rejects those, because
    choosing between them would be a guess.
    """
    removed = []
    parts = re.split(r"(?=<｜DSML｜invoke)", body)
    for i, part in enumerate(parts):
        seen = {}

        def keep_first(m):
            name, value = m.group(1), (m.group(2), m.group(3))
            if seen.get(name) == value:
                removed.append(name)
                return ""
            seen.setdefault(name, value)
            return m.group(0)

        parts[i] = _DSML_PARAM.sub(keep_first, part)
    if removed:
        log(f"DSML: dropped identical duplicate parameters {removed}")
    return "".join(parts)


def _drop_conflicting_required(body, tools):
    """Leave out a required parameter that was sent twice with different values.

    Sampled replays of a stuck edit sometimes sent two different `newString`s
    (2026-11-09). The parser rejects that, so the call reached the client as
    plain text and ended the task. Without the parameter the call fails the
    client's validation and the model retries, with no value guessed. Optional
    parameters are left alone: a call missing one would run.

    A parameter the tool's schema does not list is left out too: the tool
    does not take it, so nothing is lost. In B7 (2026-11-13) a complete `write`
    (path + content) carried two different `oldString`s, an `edit` parameter,
    and the whole call reached the client as text, ending the session.
    """
    required = {}
    for t in tools or []:
        f = t.get("function", t)
        params = f.get("parameters") or {}
        props = params.get("properties")
        req = set(params.get("required") or [])
        # Unknown names are dropped only when the schema lists its properties.
        required[f.get("name")] = (req, set(props) if isinstance(props, dict) else None)
    dropped = []
    parts = re.split(r"(?=<｜DSML｜invoke)", body)
    for i, part in enumerate(parts):
        invoke = re.match(r'<｜DSML｜invoke name="([^"]*)"', part)
        values = {}
        for name, _, value in _DSML_PARAM.findall(part):
            values.setdefault(name, set()).add(value)
        conflicts = {k for k, v in values.items() if len(v) > 1}
        req, known = required.get(invoke.group(1), (set(), None)) if invoke else (set(), None)
        conflicts = {k for k in conflicts if k in req or (known is not None and k not in known)}
        if conflicts:
            dropped += sorted(conflicts)
            parts[i] = _DSML_PARAM.sub(
                lambda m: "" if m.group(1) in conflicts else m.group(0), part)
    if dropped:
        log(f"DSML: left out required or unknown parameters sent with conflicting values {dropped}")
    return "".join(parts)


_PARAM_END, _INVOKE_END = "</｜DSML｜parameter>", "</｜DSML｜invoke>"


def _close_cut_off(body):
    """Close a tool-calls block that stopped partway, after its last finished part.

    The loop guard stops a turn mid-call. In W5 (2026-11-14) the model was
    repeating `oldString` inside an edit; the cut-off call reached OpenCode as
    text with finish `length`, three turns running, and the headless session
    gave up. Kept to the parameters that finished, the call goes through the
    usual duplicate and conflict handling: a complete call runs, an incomplete
    one fails the client's validation and the model retries. Returns None if
    no parameter finished.
    """
    p, v = body.rfind(_PARAM_END), body.rfind(_INVOKE_END)
    if max(p, v) == -1:
        return None
    if v > p:
        return body[:v + len(_INVOKE_END)] + "\n"
    return body[:p + len(_PARAM_END)] + "\n" + _INVOKE_END + "\n"


def _close_unclosed_invokes(body):
    """Close a tool call that is still open when the next one starts.

    In X3 (2026-11-14) the model stopped partway into an edit's `newString` and
    started the same edit again as a new call. The parser read both as one call
    with two `oldString`s, the whole reply reached OpenCode as text and the
    session ended. The open call keeps only its finished parameters (as in
    _close_cut_off), so it fails the client's validation; the next call is
    unaffected. No value is guessed.
    """
    parts = re.split(r"(?=<｜DSML｜invoke )", body)
    fixed = 0
    for k in range(len(parts) - 1):
        if parts[k].startswith("<｜DSML｜invoke ") and _INVOKE_END not in parts[k]:
            closed = _close_cut_off(parts[k])
            if closed is None:  # no parameter finished: an empty call
                closed = parts[k].split("\n", 1)[0] + "\n" + _INVOKE_END + "\n"
            parts[k] = closed
            fixed += 1
    if fixed:
        log(f"DSML: closed {fixed} tool call(s) left open before the next one")
    return "".join(parts)


def parse_dsml(text, tools=None, truncated=False):
    """Return (content before the calls, [openai tool_call, ...]) or (text, []).

    truncated: generation was cut off (loop guard), so a call without its
    closing tag is closed after its last finished parameter.
    """
    i = text.find(DSML_OPEN)
    if i == -1:
        return text, []
    body = text[i + len(DSML_OPEN):]
    j = body.find(DSML_STOP)
    if j == -1 and truncated:
        closed = _close_cut_off(body)
        if closed is None:
            return text, []
        log("DSML: generation stopped inside a tool call; kept its finished parameters")
        body, j = closed, len(closed)
    # Generation stops on the closing tag's stem, so put the full tag back.
    body = _close_unclosed_invokes(body[:j] if j != -1 else body)
    body = _drop_identical_duplicates(body + DSML_END)
    body = _drop_conflicting_required(body, tools)
    _dsml_fixups.clear()
    try:
        _, _, calls = dsv4_encoding.parse_tool_calls(0, body)
        for c in calls:
            keys = list(json.loads(c["arguments"]).keys())
            log(f"tool call {c['name']}: args {keys}"
                + (f" (rescued non-JSON string=false values: {_dsml_fixups})" if _dsml_fixups else ""))
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
        thinking, effort, budget = reasoning_mode(body)
        job = finish_job(body, dsv4_prompt(messages, tools, thinking, effort), tools, "dsml",
                         think=thinking)
        if not thinking:
            job["_pc_src"] = (messages, tools)  # for queue_precache; never sent to other ranks
        if budget:
            # Leave room for the answer: a client's max_tokens can be close to the
            # budget (Cline's ~4k cut a turn off mid-reasoning, 2026-11-08), and the
            # budget only forces "</think>" once it is exceeded.
            budget = max(256, min(budget, int(job["max_tokens"] * 0.75)))
        job["think_budget"] = budget
        job["client_max_tokens"] = body.get("max_tokens") or body.get("max_completion_tokens")
        return job
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


# Greedy decoding resends a failed call verbatim, even when told exactly what was
# wrong (replays of Test B's stuck turn, 2026-11-09); sampling is what breaks it.
RESAMPLE_ON_REPEAT = True  # --no-resample-on-repeat
REPEAT_STREAK = 2          # same tool, same error, this many times in a row


def _tool_error(content):
    """The error in a tool result, as a comparable string, or None if it has none."""
    try:
        result = json.loads(flatten_content(content))
    except (TypeError, ValueError):
        return None
    error = result.get("error") if isinstance(result, dict) else None
    return json.dumps(error, sort_keys=True) if error else None


def failure_streak(messages):
    """How many of the latest tool results are the same tool failing the same way.

    Only tool results count: other messages between them neither add to the
    streak nor end it.
    """
    names, streak, last = {}, 0, None
    for m in messages:
        for call in m.get("tool_calls") or []:
            names[call.get("id")] = (call.get("function") or {}).get("name")
        if m.get("role") not in ("tool", "function"):
            continue
        error = _tool_error(m.get("content"))
        key = (names.get(m.get("tool_call_id")), error) if error else None
        streak = 0 if key is None else streak + 1 if key == last else 1
        last = key
    return streak


def finish_job(body, prompt, tools, tool_format, think=False):
    temperature, top_p = body.get("temperature"), body.get("top_p")
    client_max = body.get("max_tokens") or body.get("max_completion_tokens")
    sources = {"temperature": "client" if temperature is not None else "default",
               "top_p": "client" if top_p is not None else "default",
               "max_tokens": "client" if client_max else "default"}
    if think:
        # DeepSeek's recommendation: temperature 1.0, top_p 0.95 for agent work.
        # Greedy decoding tends to loop in long reasoning.
        temperature = 1.0 if temperature is None else temperature
        top_p = (0.95 if tools else 1.0) if top_p is None else top_p
    streak = failure_streak(body.get("messages") or []) if RESAMPLE_ON_REPEAT else 0
    if streak >= REPEAT_STREAK and temperature is None:
        temperature, sources["temperature"] = 1.0, "repeat"
        top_p = 0.95 if top_p is None else top_p
        log(f"tool failed the same way {streak} times in a row: sampling this turn")
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
        "param_source": sources,
        "repeat_streak": streak,
    }


TRACE_DIR = os.path.expanduser("~/dsv4-traces")  # --trace-dir; "" disables (--no-trace)


def write_trace(rid, body, job, raw, reasoning, answer, tool_calls, finish, usage, timing):
    """Save one request/response pair for later analysis and replay (rank 0 only).

    Written after generation finishes, so it cannot affect the lockstep between
    ranks. Contains the client's full request (messages, tools, sampling) and the
    model's raw output: local files only, never commit them.
    """
    if not TRACE_DIR:
        return
    try:
        day = time.strftime("%Y-%m-%d")
        folder = os.path.join(TRACE_DIR, day)
        os.makedirs(folder, exist_ok=True)
        record = {
            "id": rid, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request": body,
            "effective": {k: job.get(k) for k in ("think", "think_budget", "max_tokens", "temperature",
                                                  "top_p", "seed", "tool_format", "cache", "param_source")},
            "prompt": job.get("prompt"),
            "output": {"raw": raw, "reasoning": reasoning, "answer": answer,
                       "tool_calls": tool_calls, "finish_reason": finish,
                       "looped": bool(job.get("_looped"))},
            "usage": usage, "cached_tokens": job.get("_reused", 0), "timing": timing,
        }
        path = os.path.join(folder, f"{time.strftime('%H%M%S')}-{rid[-8:]}.json")
        with open(path, "w") as f:
            json.dump(record, f, ensure_ascii=False)
    except Exception as e:  # tracing must never break serving
        log(f"{rid}: trace write failed: {e}")


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

# ------------------------------------------------ two requests at once (--concurrent 2)
#
# With one request at a time each rank idles half of every decoded token: rank 1
# runs the first layers while rank 0 waits, then the reverse. With --concurrent 2
# two requests are decoded interleaved: each step rank 1 runs its layers for one
# request while rank 0 runs its layers for the other, then one all_gather swaps
# rank 1's hidden state and rank 0's sampled token (proto_interleave.py: 1.87x the
# combined throughput, tokens identical to one at a time). Only the engine thread
# uses MLX; on rank 0 the HTTP handlers run in their own threads and hand jobs over
# through ENGINE_QUEUE. Prefill is done here rather than by stream_generate:
# generate_step works one token ahead, so stopping it after the first token would
# leave that token's KV in the cache before we feed it again.
#
# Protocol, identical on every rank: "change" (finished slots, jobs to admit) ->
# admit each job (prefill + first token, sampled with the job's seed as usual) ->
# "go" (slots that finished on their first token) -> interleave() until rank 0
# calls for an event -> "change" again. Rank 0 alone samples after the first
# token, decides when a request stops, and streams it.

CONCURRENT = 1        # --concurrent
ENGINE_QUEUE = None   # rank 0: (job, events queue, cancel Event); None to shut down
PRECACHE = _collections.deque()  # rank 0: (prompt after a tool result, prompt after a user turn)
PRECACHE_MAX_NEW = 8192          # most tokens one precache prefills


def queue_precache(messages, tools, reply):
    """Rank 0 handler threads: after a reply, ask the engine to cache the conversation with
    the reply in it, as the client will send it back.

    The reply's encoding ends where the next turn begins, and that depends on what comes
    next, so encode it followed by a tool result and by a user turn: the engine keeps
    their common tokens. If the next request differs anyway (the client edited the
    reply), the lookup falls back to the prompt's own snapshot, as before.
    """
    msgs = list(messages) + [reply]
    call_id = (reply.get("tool_calls") or [{}])[0].get("id", "x")
    after_tool = dsv4_prompt(msgs + [{"role": "tool", "tool_call_id": call_id, "content": "x"}], tools)
    after_user = dsv4_prompt(msgs + [{"role": "user", "content": "y"}], tools)
    PRECACHE.append((after_tool, after_user))
    ENGINE_QUEUE.put(("precache",))  # wakes the engine; the work is in PRECACHE


class Slot:
    """One request being decoded. Every rank keeps a Slot per request with the same tokens."""

    def __init__(self, sid, first_token):
        self.sid = sid
        self.tokens = [first_token]
        self.finished = False


def interleave(rank, lanes, first_half, second_half, exchange, on_token=None, want_event=None):
    """Decode up to two requests at once on a 2-rank pipeline, until rank 0 calls for an event.

    Runs on both ranks with the same `lanes` ([slot, slot-or-None]). Each step rank 1
    runs the first layers for lane (step % 2) while rank 0 runs the last layers for the
    slot rank 1 worked on the step before, samples its next token and decides whether
    it stops. One exchange per step carries rank 1's hidden state and rank 0's token
    and control flag to both ranks. When rank 0 sets the flag (a slot finished, or
    want_event() says a request is waiting), one "drain" step finishes the hidden state
    still in flight, so nothing is pending on return. Rank 0 appends tokens in
    on_token(slot, token) -> stop?; rank 1 appends the same tokens from the exchange.
    Tested offline by interleave_test.py.
    """
    pending = None
    received = None
    draining = False
    step = 0
    while True:
        lane = None if draining else lanes[step % 2]
        if lane is not None and lane.finished:
            lane = None
        hidden = first_half(lane) if rank == 1 and lane is not None else None
        token, control = None, 0
        if rank == 0 and pending is not None:
            token = second_half(pending, received)
            if on_token(pending, token):
                control = 1
        if rank == 0 and not draining and want_event is not None and want_event():
            control = 1
        token_seen, control_seen, hidden_seen = exchange(hidden, token, control)
        if rank == 1 and pending is not None:
            pending.tokens.append(token_seen)
            if on_token is not None:
                on_token(pending, token_seen)  # rank 1 mirrors the text; rank 0 decides stops
        step += 1
        if draining:
            return step
        if control_seen:
            draining = True
        pending, received = lane, hidden_seen
        if pending is None and draining:
            return step


def _layers_here(inner, ids, h, caches):
    """This rank's layers for one request, as DeepseekV4Model.__call__ runs them (text only)."""
    if h is None:  # the rank with the first layers embeds the token
        h = inner.embed_tokens(ids)
        h = mx.contiguous(mx.broadcast_to(h[:, :, None, :],
                                          (h.shape[0], h.shape[1], inner.args.hc_mult, h.shape[2])))
    first = caches[0]
    mask = _create_attention_mask(h[:, :, 0, :], first[0] if isinstance(first, _CacheList) else first,
                                  window_size=inner.args.sliding_window, return_array=True)
    for layer, c in zip(inner.pipeline_layers, caches):
        h = layer(h, mask, c, ids, causal=True)
    return h


def _prefill(model, processor, job, on_start=None):
    """Prompt-cache lookup, chunked prefill with the usual snapshots, and the first token's
    logits. Runs on every rank in lockstep, like stream_generate's prefill."""
    lm = model.language_model
    ids = _orig_prepare(model, processor, job["prompt"], None, None, None, {})[0].flatten().tolist()
    use_cache = job.get("cache", True) and PROMPT_CACHE
    reused, cache = 0, None
    if use_cache:
        best = None
        for snap in SNAPS:
            n = len(snap["ids"])
            if n < len(ids) and (best is None or n > len(best["ids"])) and ids[:n] == snap["ids"]:
                best = snap
        if best is not None:
            _tick[0] += 1
            best["used"] = _tick[0]
            cache = copy.deepcopy(best["cache"])
            mx.eval([c.state for c in cache])
            reused = len(best["ids"])
    if cache is None:
        cache = _vlm_cache.make_prompt_cache(lm)
    if on_start is not None:
        on_start(len(ids), reused)
    rest = ids[reused:]
    points = _checkpoint_lengths(ids, reused, len(rest)) if use_cache else []
    step = job.get("prefill_step", PREFILL_STEP)
    done, target, snaps = 0, len(rest) - 1, 0
    while done < target:
        end = min(done + step, target)
        if points and points[0] > done:
            end = min(end, points[0])
        lm(mx.array(rest[done:end])[None], cache=cache, skip_logits=True)
        mx.eval([c.state for c in cache])
        done = end
        if points and done == points[0]:
            _add_snapshot(ids[: reused + done], cache)
            snaps += 1
            points.pop(0)
    logits = lm(mx.array(rest[-1:])[None], cache=cache).logits[:, -1, :]
    return ids, reused, snaps, cache, logits


class Engine:
    """Runs on every rank's main thread; rank 0 also streams to the waiting handlers."""

    def __init__(self, world, model, processor):
        self.world, self.model, self.processor = world, model, processor
        self.rank = world.rank()
        self.lm = model.language_model
        self.inner = self.lm.model
        self.tok = getattr(processor, "tokenizer", processor)
        self.hdt = self.inner.embed_tokens(mx.array([[0]])).dtype
        self.shape = (1, 1, self.inner.args.hc_mult, self.inner.args.hidden_size)
        self.slots = []
        self.next_sid = 0
        self.side = {}  # rank 0: sid -> (events queue, cancel Event)
        try:
            from mlx_vlm.models.deepseek_v4 import language as _dl
            self.indexers = [m for _, m in self.lm.named_modules() if isinstance(m, _dl.Indexer)]
        except ImportError:
            self.indexers = []

    # --- per-step pieces for interleave()
    def first_half(self, slot):
        return _layers_here(self.inner, mx.array([[slot.tokens[-1]]]), None, slot.cache)

    def second_half(self, slot, hidden):
        h = _layers_here(self.inner, mx.array([[slot.tokens[-1]]]), hidden.astype(self.hdt), slot.cache)
        logits = _linear(self.lm.lm_head, self.inner.norm(self.inner.hc_head(h)))[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return int(slot.sampler(logprobs).item())

    def exchange(self, hidden, token, control):
        if self.rank == 0:
            head = mx.array([-1.0 if token is None else float(token), float(control)], dtype=mx.float32)
            payload = mx.concatenate([head, mx.zeros((self.shape[2] * self.shape[3] - 2,), mx.float32)])
            payload = payload.reshape(self.shape)
        else:
            payload = mx.zeros(self.shape, mx.float32) if hidden is None else hidden.astype(mx.float32)
        g = mx.distributed.all_gather(payload, group=self.world)
        mx.eval(g)
        row0 = g[0].reshape(-1)
        t, c = int(row0[0].item()), int(row0[1].item())
        return (None if t < 0 else t), c, g[1:2]

    # --- rank 0: what happens to each sampled token
    def advance(self, slot, token):
        """Detokenize one token and apply the stop checks run_job applies. Deterministic, so
        every rank reaches the same decision from the same tokens; only rank 0 streams.
        Returns the stop reason or None. (A client that went away is rank 0's to decide.)"""
        job = slot.job
        if self.tok.stopping_criteria(token):
            return "stop"
        slot.detok.add_token(token, skip_special_token_ids=[])
        seg = slot.detok.last_segment
        slot.text += seg
        slot.n += 1
        if self.rank == 0:
            self.side[slot.sid][0].put(("chunk", SimpleNamespace(text=seg, token=token)))
        marker = job.get("stop_marker")
        if marker and marker in slot.text:
            return "stop"
        if slot.n % LOOP_EVERY == 0 and looping(slot.text):
            job["_looped"] = True
            return "loop"
        if slot.n >= job["max_tokens"]:
            return "length"
        return None

    def accept(self, slot, token):
        """Rank 0: advance(), and finish the slot if it stops or its client went away."""
        why = self.advance(slot, token)
        if why is None and self.side[slot.sid][1].is_set():
            why = "cancel"
        return self.finish(slot, why) if why else False

    def mirror(self, slot, token):
        """Rank 1: keep the slot's text and count in step with rank 0."""
        if self.advance(slot, token):
            slot.finished = True

    def on_token(self, slot, token):
        slot.tokens.append(token)
        return self.accept(slot, token)

    def finish(self, slot, why):
        global LAST_JOB_END
        slot.finished = True
        events, _ = self.side.pop(slot.sid)
        slot.detok.finalize()
        if slot.detok.last_segment:
            slot.text += slot.detok.last_segment
            events.put(("chunk", SimpleNamespace(text=slot.detok.last_segment, token=None)))
        job = slot.job
        job["_text"], job["_reused"], job["_new_snaps"] = slot.text, slot.reused, slot.snaps
        if slot.look is not None and slot.look.passes:
            lk = slot.look
            log(f"copy-spec: {lk.passes} verify passes, {lk.accepted}/{lk.drafted} draft tokens accepted, "
                f"{slot.n} tokens")
        LAST_JOB_END = time.monotonic()
        dt = time.perf_counter() - slot.t_decode
        events.put(("done", SimpleNamespace(prompt_tokens=slot.prompt_tokens, generation_tokens=slot.n,
                                            prompt_tps=slot.prompt_tps,
                                            generation_tps=slot.n / dt if dt > 0 else 0.0)))
        return True

    # --- both ranks
    def admit(self, sid, job, on_start=None):
        if job.get("prefault"):
            n, dt, waited = prefault(self.model, cpu_barrier)
            log(f"rank {self.rank}: paged in {n} weight tensors in {dt:.1f}s, then waited {waited:.1f}s")
        mx.random.seed(job["seed"])  # same seed everywhere, so the first token matches
        t0 = time.perf_counter()
        ids, reused, snaps, cache, logits = _prefill(self.model, self.processor, job, on_start)
        sampler = _make_sampler(temp=job["temperature"], top_p=job["top_p"])
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first = int(sampler(logprobs).item())
        slot = Slot(sid, first)
        slot.job, slot.cache, slot.sampler = job, cache, sampler
        slot.prompt_tokens, slot.reused, slot.snaps = len(ids), reused, snaps
        slot.prompt_tps = len(ids) / max(time.perf_counter() - t0, 1e-9)
        slot.t_decode = time.perf_counter()
        slot.text, slot.n = "", 0
        slot.seed = int(job["seed"])
        slot.prompt_ids = ids
        slot.look = CopyLookup() if COPY_SPEC else None
        slot.detok = _make_detokenizer(self.processor)
        if self.rank == 0:
            self.accept(slot, first)
        else:
            self.mirror(slot, first)
        return slot

    SYNC_EVERY = 8  # lone request: tokens between checks for rank 0's own events

    def _sample_lockstep(self, slot, token, index):
        """Pipeline forward of `token` (both ranks, as stream_generate does), then a sample
        drawn with a per-token seed, so both ranks get the same token. Lazy."""
        logits = self.lm(mx.array([[token]]) if isinstance(token, int) else token.reshape(1, 1),
                         cache=slot.cache).logits[:, -1, :]
        mx.random.seed((slot.seed * 1_000_003 + index) % (2 ** 31))
        return slot.sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))

    def decode_single(self, slot):
        """One request: lockstep like normal mode, building one token ahead.

        Both ranks sample the same token and make the same stop decision (advance() is
        deterministic). Every SYNC_EVERY tokens a CPU all_sum tells rank 1 whether rank 0
        wants to stop for its own reasons (a request is waiting, or the client went away).
        """
        nxt = self._sample_lockstep(slot, slot.tokens[-1], len(slot.tokens))
        mx.async_eval(nxt)
        since = 0
        while True:
            cur = nxt
            nxt = self._sample_lockstep(slot, cur, len(slot.tokens) + 1)  # build ahead
            mx.async_eval(nxt)
            tok = int(cur.item())
            slot.tokens.append(tok)
            why = self.advance(slot, tok)
            since += 1
            event = False
            if why is None and since >= self.SYNC_EVERY:
                since = 0
                flag = 0
                if self.rank == 0:
                    cancelled = self.side[slot.sid][1].is_set()
                    waiting = len(self.slots) < CONCURRENT and not ENGINE_QUEUE.empty()
                    flag = 2 if cancelled else (1 if waiting else 0)
                flag = int(mx.distributed.all_sum(mx.array([flag]), group=self.world, stream=mx.cpu).item())
                if flag == 2:
                    why = "cancel"
                elif flag == 1:
                    event = True
            if why or event:
                # The token built ahead is already in the cache on both ranks: take it too.
                ahead = int(nxt.item())
                if why:
                    if self.rank == 0:
                        self.finish(slot, why)
                    else:
                        slot.finished = True
                    return
                slot.tokens.append(ahead)
                why = self.advance(slot, ahead)
                if why:
                    if self.rank == 0:
                        self.finish(slot, why)
                    else:
                        slot.finished = True
                return

    def _spec_step(self, slot):
        """One step of copy-spec decoding, on every rank: returns the new tokens (at least one).
        Invariant kept: the cache holds every token of the request except the last."""
        index = len(slot.tokens)
        ctx = slot.prompt_ids + slot.tokens
        draft = []
        if HISA_MIN_POOLED and len(ctx) + 2 * CopyLookup.MAX < 4 * HISA_MIN_POOLED:
            draft = slot.look.draft(ctx)  # (beyond that, decode uses HISA: no exact match)
        if not draft:
            logits = self.lm(mx.array([[slot.tokens[-1]]]), cache=slot.cache).logits[:, -1, :]
            mx.random.seed((slot.seed * 1_000_003 + index) % (2 ** 31))
            return [int(slot.sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item())]
        inp = [slot.tokens[-1]] + draft
        for m in self.indexers:  # exact top-k in the multi-token pass too (its HISA kernel differs)
            m.index_block = 0
        tx = start_speculative_cache(slot.cache, len(inp))
        try:
            logits = self.lm(mx.array(inp)[None], cache=slot.cache).logits[0].astype(mx.float32)
        finally:
            for m in self.indexers:
                m.index_block = 64
        job = slot.job
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        d = mx.array(draft)
        if job["temperature"] <= 0:
            preds = mx.argmax(lp, axis=-1).tolist()
            a = 0
            while a < len(draft) and preds[a] == draft[a]:
                a += 1
            bonus = preds[a]
        else:
            # Speculative sampling with one-token drafts: accept draft token j with probability
            # p_j(d_j); at the first rejection sample from p_j without d_j. Exact for p.
            if 0 < job["top_p"] < 1:
                lp = _apply_top_p(lp, job["top_p"])
            probs = mx.softmax(lp / max(job["temperature"], 0.01), axis=-1)
            p_d = mx.take_along_axis(probs[:-1], d[:, None], axis=-1)[:, 0]
            u = mx.random.uniform(shape=(len(draft),), key=mx.random.key((slot.seed * 1_000_003 + index) % (2 ** 31)))
            ok = (u < p_d).tolist()
            a = 0
            while a < len(draft) and ok[a]:
                a += 1
            row = probs[a]
            if a < len(draft):
                row = mx.where(mx.arange(row.shape[0]) == draft[a], 0.0, row)
            key = mx.random.key((slot.seed * 1_000_003 + index + 7919) % (2 ** 31))
            bonus = int(mx.random.categorical(mx.log(row + 1e-30), key=key).item())
        tx.commit(a + 1)  # keep inp[0..a]
        slot.look.update(len(draft), a)
        return draft[:a] + [bonus]

    def decode_single_spec(self, slot):
        """One request with copy-spec: like decode_single (same stop and event handling), but
        each step may add several tokens, and nothing is built ahead."""
        since = 0
        while True:
            new = self._spec_step(slot)
            why = None
            for tok in new:
                slot.tokens.append(tok)
                why = self.advance(slot, tok)
                if why:
                    break
            since += len(new)
            event = False
            if why is None and since >= self.SYNC_EVERY:
                since = 0
                flag = 0
                if self.rank == 0:
                    cancelled = self.side[slot.sid][1].is_set()
                    waiting = len(self.slots) < CONCURRENT and not ENGINE_QUEUE.empty()
                    flag = 2 if cancelled else (1 if waiting else 0)
                flag = int(mx.distributed.all_sum(mx.array([flag]), group=self.world, stream=mx.cpu).item())
                if flag == 2:
                    why = "cancel"
                elif flag == 1:
                    event = True
            if why:
                if self.rank == 0:
                    self.finish(slot, why)
                else:
                    slot.finished = True
                return
            if event:
                return

    def decode(self):
        live = [s for s in self.slots if not s.finished]
        if not live:
            return
        if len(live) == 1:
            if live[0].look is not None:
                return self.decode_single_spec(live[0])
            return self.decode_single(live[0])
        lanes = [live[0], live[1] if len(live) > 1 else None]
        if self.rank == 0:
            def want_event():  # a waiting request and room for it (cancel is handled per token)
                return len(live) < CONCURRENT and not ENGINE_QUEUE.empty()
            interleave(0, lanes, None, self.second_half, self.exchange, self.on_token, want_event)
        else:
            interleave(1, lanes, self.first_half, None, self.exchange, self.mirror)

    def tokenize(self, prompt):
        return _orig_prepare(self.model, self.processor, prompt, None, None, None, {})[0].flatten().tolist()

    def precache_ids(self):
        """Rank 0: the token ids to precache, from what the handlers queued."""
        out = []
        while PRECACHE:
            a, b = (self.tokenize(x) for x in PRECACHE.popleft())
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            out.append(a[:n])
        return out

    def precache(self, ids):
        """Both ranks: extend the longest snapshot that is a prefix of `ids` to all of
        them, and keep that as a snapshot. Returns the tokens prefilled."""
        best = None
        for snap in SNAPS:
            n = len(snap["ids"])
            if n < len(ids) and (best is None or n > len(best["ids"])) and ids[:n] == snap["ids"]:
                best = snap
        if best is None or len(ids) - len(best["ids"]) > PRECACHE_MAX_NEW:
            return 0
        cache = copy.deepcopy(best["cache"])
        rest = ids[len(best["ids"]):]
        step = PREFILL_STEP
        for i in range(0, len(rest), step):
            self.lm(mx.array(rest[i:i + step])[None], cache=cache, skip_logits=True)
            mx.eval([c.state for c in cache])
        _add_snapshot(list(ids), cache)
        return len(rest)

    def run_precache(self, lists):
        for ids in lists:
            t = time.perf_counter()
            n = self.precache(ids)
            if self.rank == 0:
                log(f"precache: {n} tokens on top of {len(ids) - n} in {time.perf_counter() - t:.2f}s"
                    if n else f"precache: skipped ({len(ids)} tokens, no snapshot to build on)")

    def run_rank0(self):
        while True:
            finished = [s.sid for s in self.slots if s.finished]
            self.slots = [s for s in self.slots if not s.finished]
            admits = []
            if not self.slots:
                item = ENGINE_QUEUE.get()
                if item is None:
                    broadcast(self.world, {"cmd": "shutdown"})
                    return
                if item[0] != "precache":
                    admits.append(item)
            while len(self.slots) + len(admits) < CONCURRENT:
                try:
                    item = ENGINE_QUEUE.get_nowait()
                except _queue.Empty:
                    break
                if item is None:
                    ENGINE_QUEUE.put(None)  # finish the current requests first
                    break
                if item[0] != "precache":
                    admits.append(item)
            pre = self.precache_ids()
            plan = []
            for job, events, cancel in admits:
                sid = self.next_sid
                self.next_sid += 1
                self.side[sid] = (events, cancel)
                plan.append((sid, job, events))
            broadcast(self.world, {"cmd": "change", "finished": finished, "precache": pre,
                                   "admit": [[sid, job] for sid, job, _ in plan]})
            try:
                self.run_precache(pre)
            except Exception:
                SNAPS.clear()
                raise
            for sid, job, events in plan:
                try:
                    slot = self.admit(sid, job, lambda total, reused, ev=events: ev.put(("start", (total, reused))))
                except Exception as e:
                    SNAPS.clear()
                    events.put(("error", e))
                    raise
                self.slots.append(slot)
            first_done = [s.sid for s in self.slots if s.finished]
            broadcast(self.world, {"cmd": "go", "finished": first_done})
            self.slots = [s for s in self.slots if not s.finished]
            self.decode()

    def run_follower(self):
        while True:
            cmd = broadcast(self.world)
            if cmd.get("cmd") == "shutdown":
                return
            gone = set(cmd.get("finished", []))
            self.slots = [s for s in self.slots if s.sid not in gone]
            self.run_precache(cmd.get("precache", []))
            for sid, job in cmd.get("admit", []):
                self.slots.append(self.admit(sid, job))
            go = broadcast(self.world)
            gone = set(go.get("finished", []))
            self.slots = [s for s in self.slots if s.sid not in gone]
            self.decode()


def submit(job):
    """Rank 0 handler threads: queue a job for the engine. Returns (events queue, cancel Event)."""
    events, cancel = _queue.Queue(), threading.Event()
    ENGINE_QUEUE.put((job, events, cancel))
    return events, cancel


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

            pc_src = job.pop("_pc_src", None)
            job["prefault"] = needs_prefault(time.monotonic(), LAST_JOB_END)
            if CONCURRENT <= 1:
                broadcast(world, job)  # every rank now starts the same job

            state = {"text": "", "sent": 0, "rsent": 0, "stopped": False, "client_gone": False,
                     "content": ""}  # content as streamed: what the client will send back

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
                    state["content"] += visible[state["sent"]:]
                state["sent"] = len(visible)

            t0 = time.perf_counter()
            show_stats = stream and STATS_IN_REASONING

            def on_start(total, reused):
                send_sse(chunk_obj({"reasoning_content": stats_start_line(total, reused)}))

            if CONCURRENT > 1:
                # The engine thread decodes; this thread streams what it sends back.
                events, cancel = submit(job)
                last = None
                while True:
                    kind, val = events.get()
                    if kind == "start":
                        if show_stats:
                            on_start(*val)
                    elif kind == "chunk":
                        on_chunk(val)
                        if state["client_gone"]:
                            cancel.set()
                    elif kind == "done":
                        last = val
                        break
                    else:
                        raise val
            else:
                last = run_job(model, processor, job, on_chunk, on_start if show_stats else None)
            elapsed = time.perf_counter() - t0
            reasoning, src, think_done = split_think(state["text"], job["think"], final=True)
            topen_ = job["tool_open"]
            if job["think"] and not think_done:
                k = state["text"].find(topen_) if topen_ else -1
                if k != -1:
                    # Tool call written inside reasoning that never closed: keep the
                    # call, treat everything before it as reasoning.
                    reasoning, src = state["text"][:k].rstrip(), state["text"][k:]
                else:
                    reasoning = state["text"]  # ran out mid-thought: all reasoning
            text, _ = apply_stop(src, stops)
            tool_calls = []
            if job["tool_open"]:
                text, tool_calls = (parse_dsml(text, body.get("tools"), truncated=bool(job.get("_looped")))
                                    if job["tool_format"] == "dsml" else parse_tool_call(text))
                if tool_calls:
                    check_required(tool_calls, body.get("tools"), state["text"])

            if tool_calls:
                # Includes a call kept from a looped turn: the client then runs it
                # or rejects it, instead of seeing `length` (see _close_cut_off).
                finish = "tool_calls"
            elif job.get("_looped"):
                finish = "length"  # cut short; tell the client it is incomplete
            elif state["stopped"]:
                finish = "stop"
            elif last is not None and last.generation_tokens >= job["max_tokens"]:
                finish = "length"
            else:
                finish = "stop"
            if last is not None:
                update_prefill_estimate(last.prompt_tokens, job.get("_reused", 0), last.prompt_tps)
                if job.get("_looped") or last.generation_tokens >= 2000:
                    raw = job.get("_text", "")
                    log(f"{rid}: {'LOOP stopped' if job.get('_looped') else 'long output'} after "
                        f"{last.generation_tokens} tokens; head {raw[:200]!r} ... tail {raw[-200:]!r}")
            usage = {
                "prompt_tokens": last.prompt_tokens if last else 0,
                "completion_tokens": last.generation_tokens if last else 0,
                "total_tokens": (last.prompt_tokens + last.generation_tokens) if last else 0,
            }
            think_tag = (f"[think {len(reasoning)} chars, budget {job.get('think_budget')}, "
                         f"max_tokens {job.get('client_max_tokens') or 'default'}] ") if job["think"] else ""
            log(f"{rid}: {think_tag}"
                f"{'[LOOP] ' if job.get('_looped') else ''}"
                f"{'[tool ' + ', '.join(c['function']['name'] for c in tool_calls) + '] ' if tool_calls else ''}"
                f"{usage['prompt_tokens']} prompt ({job.get('_reused', 0)} cached) + {usage['completion_tokens']} gen tokens "
                f"in {elapsed:.1f}s (prefill {last.prompt_tps if last else 0:.0f} tok/s, "
                f"decode {last.generation_tps if last else 0:.1f} tok/s){' [client gone]' if state['client_gone'] else ''}"
                f" | temp {job['temperature']}({job['param_source']['temperature'][0]}) "
                f"top_p {job['top_p']}({job['param_source']['top_p'][0]})"
                f" | cache: +{job.get('_new_snaps', 0)} -> {cache_stats()}")
            write_trace(rid, body, job, job.get("_text", state["text"]), reasoning, text, tool_calls, finish,
                        usage, {"elapsed_s": round(elapsed, 2),
                                "prompt_tps": round(last.prompt_tps, 1) if last else None,
                                "generation_tps": round(last.generation_tps, 1) if last else None,
                                "client_gone": state["client_gone"]})

            if stream:
                if len(reasoning) > state["rsent"]:  # anything held back at the end
                    send_sse(chunk_obj({"reasoning_content": reasoning[state["rsent"]:]}))
                if show_stats and last is not None:
                    send_sse(chunk_obj({"reasoning_content": stats_end_line(
                        last.prompt_tokens, job.get("_reused", 0), last.generation_tokens,
                        last.prompt_tps, last.generation_tps, elapsed, cache_stats(),
                        lead="\n\n" if reasoning else "")}))
                if len(text) > state["sent"] and (job["think"] or (job["tool_open"] and not tool_calls)):
                    # thinking: the whole answer (held until now); otherwise any unparsed remainder
                    send_sse(chunk_obj({"content": text[state["sent"]:]}))
                    state["content"] += text[state["sent"]:]
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

            if (CONCURRENT > 1 and PROMPT_CACHE and pc_src is not None and pc_src[1]
                    and job.get("cache", True) and finish in ("tool_calls", "stop")
                    and not state["client_gone"]):
                content = state["content"] if stream else text
                reply = dict({"role": "assistant", "content": content},
                             **({"tool_calls": tool_calls} if tool_calls else {}))
                try:
                    queue_precache(pc_src[0], pc_src[1], reply)
                except Exception as e:  # an optimisation only: never fail the request for it
                    log(f"{rid}: precache not queued: {e!r}")

    return Handler


HISA_MIN_POOLED = 12_000  # --hisa-min-pooled; 0 keeps mlx-vlm's HISA everywhere
COPY_SPEC = False         # --copy-spec: draft copies from the context (engine, lone request)


class CopyLookup:
    """Copy drafts for speculative decoding: earlier occurrences of the last KEY tokens; the
    one whose match with the current text reaches furthest back wins (ties: most recent),
    if that match is at least MIN_MATCH tokens, and the draft is what followed it.
    Deterministic, so every rank drafts the same tokens."""
    KEY, MIN_MATCH = 6, 8
    MIN_DRAFT, START, MAX, COOLDOWN = 8, 12, 64, 4  # passes of 2-8 tokens take mlx-vlm's slow path

    def __init__(self):
        self.pos, self.done = {}, self.KEY - 1
        self.dlen, self.cool = self.START, 0
        self.passes = self.drafted = self.accepted = 0

    def draft(self, ctx):
        if self.cool > 0:
            self.cool -= 1
            return []
        k = self.KEY
        while self.done < len(ctx) - 2:  # index every position except the current end
            self.done += 1
            self.pos.setdefault(tuple(ctx[self.done - k + 1:self.done + 1]), []).append(self.done)
        best, best_len, end = None, 0, len(ctx) - 1
        for p in reversed(self.pos.get(tuple(ctx[-k:]), [])[-16:]):
            n = k
            while n < 256 and p - n >= 0 and ctx[p - n] == ctx[end - n]:
                n += 1
            if n > best_len:
                best, best_len = p, n
        if best is None or best_len < self.MIN_MATCH:
            return []
        d = ctx[best + 1:best + 1 + self.dlen]
        return d if len(d) >= self.MIN_DRAFT else []

    def update(self, drafted, accepted):
        self.passes += 1; self.drafted += drafted; self.accepted += accepted
        if accepted == drafted:
            self.dlen = min(self.dlen * 2, self.MAX)
        else:
            self.dlen = self.START
            if accepted < 3:
                self.cool = self.COOLDOWN


def install_exact_indexer():
    """DeepSeek-V4 decode: choose the indexer's top-k exactly while the compressed context is
    short, and use mlx-vlm's HISA (coarse block filter, then fine top-k) only beyond that.

    HISA approximates the exact selection the model was trained with and is meant for long
    contexts; on M3 Ultra it was slower than the exact selection up to 32k tokens (0.6-0.7
    ms per half at 15k and 32k) and equal at 60k (2026-11-20). Prefill is unchanged.
    """
    try:
        from mlx_vlm.models.deepseek_v4 import language as dl
    except ImportError:
        return
    orig = dl.Indexer._hisa_select

    seen = set()

    def select(self, q, pooled, x, k):
        exact = pooled.shape[1] < HISA_MIN_POOLED
        if exact not in seen:
            seen.add(exact)
            log(f"indexer: {'exact top-k' if exact else 'HISA'} in decode "
                f"({pooled.shape[1]} compressed positions; switch at {HISA_MIN_POOLED})")
        if not exact:
            return orig(self, q, pooled, x, k)
        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads ** -0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]

    dl.Indexer._hisa_select = select
    log(f"indexer: exact top-k in decode below {HISA_MIN_POOLED} compressed positions")


def fixed_pipeline(self, group):
    """Pipeline split in which every layer runs on exactly one rank (mlx-lm 0.32.2's).

    mlx-vlm 0.7.12's PipelineMixin.pipeline gives each rank ceil or floor of
    layers/ranks but computes start_idx from that rank's own count, so when the
    layers don't divide evenly one layer runs nowhere. DeepSeek-V4-Flash has 43
    layers: on 2 ranks layer 21 was never loaded or run (2026-11-19). Measured on
    a 3,071-token text: 1.8246 nats/token with the dropped layer, 1.7956 with it
    back (tensor parallel, all layers: 1.7986). Rank 0 still runs the last layers.
    """
    self.pipeline_rank, self.pipeline_size = group.rank(), group.size()
    base, extra = divmod(len(self.layers), self.pipeline_size)
    split = [base + (1 if r < extra else 0) for r in range(self.pipeline_size)]
    self.start_idx = sum(split[self.pipeline_rank + 1:])
    self.end_idx = self.start_idx + split[self.pipeline_rank]
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx


PIN_WEIGHTS = True  # --no-pin-weights
STOCK_WIRED_LIMIT = importlib.import_module("mlx_vlm.generate.common").wired_limit


@contextlib.contextmanager
def _keep_wired(model, streams=None):
    """Stand-in for mlx-vlm's wired_limit once the weights are pinned: leave the limit alone."""
    yield


def pin_weights():
    """Hold the wired limit raised for the life of the server.

    mlx-vlm's stream_generate raises the limit for each call and sets it back to
    the old value (0) afterwards. At 0, MLX's residency refresh has nothing to
    keep, so idle weights get unwired and may be compressed or swapped. Measured
    on an M3 Ultra (macOS 27.2.1, 60 GiB): with the refresh on and the limit
    held, 65 GiB stayed wired through 60 s idle and the next pass took 0.11 s;
    with the limit back at 0, 4.9 GiB stayed wired and the pass took 0.58 s.
    mlx-lm's server does the same thing (raises the limit once at startup).
    Returns the limit in bytes.
    """
    limit = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(limit)
    # Several mlx_vlm modules import wired_limit by name; replace every copy.
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_vlm") and getattr(mod, "wired_limit", None) is STOCK_WIRED_LIMIT:
            mod.wired_limit = _keep_wired
    return limit


def main():
    global PREFILL_STEP, PROMPT_CACHE, CACHE_MAX_SNAPS, CACHE_BUDGET_TOKENS, CACHE_CKPT_GAP
    global DEFAULT_REASONING_EFFORT, STATS_IN_REASONING, TRACE_DIR, RESAMPLE_ON_REPEAT
    global PREFAULT_IDLE, WORLD, PIN_WEIGHTS, WAKE, CONCURRENT, ENGINE_QUEUE, HISA_MIN_POOLED, COPY_SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--prefill-step", type=int, default=PREFILL_STEP)
    ap.add_argument("--no-prompt-cache", action="store_true")
    ap.add_argument("--trace-dir", default=TRACE_DIR, help="where rank 0 saves request/response traces")
    ap.add_argument("--no-trace", action="store_true", help="don't save traces")
    ap.add_argument("--no-resample-on-repeat", action="store_true",
                    help="stay greedy even when a tool keeps failing the same way")
    ap.add_argument("--no-pin-weights", action="store_true",
                    help="let mlx-vlm lower the wired limit after each request (its default)")
    ap.add_argument("--prefault-idle", type=float, default=PREFAULT_IDLE,
                    help="seconds idle after which a job first pages the weights back in "
                         "(0 = before every job, negative = never)")
    ap.add_argument("--no-stats-in-reasoning", action="store_true",
                    help="don't add the server's status lines to the streamed reasoning")
    ap.add_argument("--default-reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                    choices=["none", "low", "medium", "high", "max"],
                    help="thinking mode for requests that don't ask (DeepSeek-V4 only)")
    ap.add_argument("--concurrent", type=int, default=1,
                    help="decode up to this many requests at once, interleaved across the pipeline (1 or 2)")
    ap.add_argument("--wake-port", type=int, default=WAKE_PORT,
                    help="TCP port on rank 0 that wakes the other ranks for each job")
    ap.add_argument("--no-wake", action="store_true",
                    help="wait for jobs inside the all_sum instead (spins a core per waiting rank)")
    ap.add_argument("--cache-max-snapshots", type=int, default=CACHE_MAX_SNAPS)
    ap.add_argument("--copy-spec", action="store_true",
                    help="--concurrent: speculative decoding with drafts copied from the context "
                         "(lone request; verified in one pass, exact for the sampler)")
    ap.add_argument("--hisa-min-pooled", type=int, default=HISA_MIN_POOLED,
                    help="DeepSeek-V4 decode: exact top-k below this many compressed positions "
                         "(4 tokens each), HISA from there on; 0 = HISA always")
    ap.add_argument("--cache-budget-tokens", type=int, default=CACHE_BUDGET_TOKENS)
    ap.add_argument("--cache-checkpoint-gap", type=int, default=CACHE_CKPT_GAP)
    args = ap.parse_args()
    PREFILL_STEP = args.prefill_step
    PROMPT_CACHE = not args.no_prompt_cache
    TRACE_DIR = "" if args.no_trace else os.path.expanduser(args.trace_dir)
    DEFAULT_REASONING_EFFORT = args.default_reasoning_effort
    STATS_IN_REASONING = not args.no_stats_in_reasoning
    RESAMPLE_ON_REPEAT = not args.no_resample_on_repeat
    PREFAULT_IDLE = args.prefault_idle
    CONCURRENT = args.concurrent
    PIN_WEIGHTS = not args.no_pin_weights
    CACHE_MAX_SNAPS = args.cache_max_snapshots
    CACHE_BUDGET_TOKENS = args.cache_budget_tokens
    CACHE_CKPT_GAP = args.cache_checkpoint_gap

    world = WORLD = mx.distributed.init()
    rank = world.rank()
    t0 = time.perf_counter()
    HISA_MIN_POOLED = args.hisa_min_pooled
    COPY_SPEC = args.copy_spec and args.concurrent > 1
    if HISA_MIN_POOLED:
        install_exact_indexer()
    if world.size() > 1:
        import mlx_vlm.models.pipeline as vlm_pipeline
        vlm_pipeline.PipelineMixin.pipeline = fixed_pipeline
        model, processor = sharded_load(args.model, pipeline_group=world)
        inner = getattr(model.language_model, "model", model.language_model)
        if hasattr(inner, "start_idx"):
            log(f"rank {rank}: runs layers {inner.start_idx}-{inner.end_idx - 1} "
                f"({inner.end_idx - inner.start_idx} of the model's layers)")
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
    if PIN_WEIGHTS:
        log(f"rank {rank}: wired limit held at {pin_weights() / 2**30:.1f} GiB, residency refresh "
            f"every {os.environ.get('MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS')} ms")
    addr = wake_address(os.environ, args.wake_port)
    if world.size() > 1 and not args.no_wake and addr is not None:
        if rank == 0:
            WAKE = Wake.listen(addr[1], world.size() - 1, host=addr[0])
        else:
            WAKE = Wake.connect(addr[0], addr[1], rank)
        log(f"rank {rank}: wake socket up on port {addr[1]}")
    elif world.size() > 1:
        log(f"rank {rank}: no wake socket; waiting ranks will spin a core")

    # Warm up before serving: the first pass compiles kernels and touches every
    # weight. In pipeline mode rank 0's GPU work waits on the other ranks, so a
    # slow first pass can hit macOS's GPU watchdog (seen 2026-11-08 on a 9-token
    # request that arrived the second the server came up). All ranks run it.
    t1 = time.perf_counter()
    # One attempt only: a GPU timeout can leave the ranks out of step, so fail
    # loudly at startup rather than retry.
    # Page in first: the ranks finish loading at different times.
    run_job(model, processor, {"prompt": "Hello", "max_tokens": 4, "temperature": 0.0,
                               "top_p": 1.0, "seed": 0, "cache": False,
                               "prefault": PREFAULT_IDLE >= 0}, lambda r: None)
    log(f"rank {rank}: warm-up done in {time.perf_counter() - t1:.1f}s")

    if CONCURRENT > 1:
        if world.size() != 2 or CONCURRENT != 2:
            raise SystemExit("--concurrent 2 needs exactly 2 ranks")
        engine = Engine(world, model, processor)
        if rank != 0:
            engine.run_follower()
            return
        ENGINE_QUEUE = _queue.Queue()
        server = ThreadingHTTPServer((args.host, args.port), make_handler(world, model, processor, config, args.model))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log(f"serving on http://{args.host}:{args.port}/v1  ({world.size()} node(s), up to 2 requests at once)")
        engine.run_rank0()
        return

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
