"""Prototype: decode two requests at once on a 2-rank pipeline, interleaved.

Normally one request at a time: rank 1 runs layers 0-20, then rank 0 runs 21-42, and
each waits while the other works. Here, each step rank 1 runs its layers for one request
while rank 0 runs its layers for the other, then one all_gather swaps rank 1's hidden
state and rank 0's sampled token (packed in float32). Each request advances every two
steps, but both ranks are busy every step.

Run on both ranks with the server stopped:
  cluster_launch.py ... -- python proto_interleave.py --model M PROMPT_A PROMPT_B [--tokens 64]

Checks: greedy tokens from the interleaved run must equal one-request-at-a-time decoding.
"""
import argparse
import os
import time

os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "1000")

import mlx.core as mx
from mlx.utils import tree_flatten
import mlx_vlm.models.pipeline as vpipe
from mlx_vlm.models import cache as vcache
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.cache import CacheList
from mlx_vlm.models.linear import linear
from mlx_vlm.utils import sharded_load

PREFILL = 512
STATE_EVAL = True  # --no-state-eval: evaluate only what the next step needs, as mlx-vlm's generate does


def fixed_pipeline(self, group):
    self.pipeline_rank, self.pipeline_size = group.rank(), group.size()
    base, extra = divmod(len(self.layers), self.pipeline_size)
    split = [base + (1 if r < extra else 0) for r in range(self.pipeline_size)]
    self.start_idx = sum(split[self.pipeline_rank + 1:])
    self.end_idx = self.start_idx + split[self.pipeline_rank]
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx


def local_layers(inner, ids, h, caches):
    """This rank's layers for one request, as DeepseekV4Model.__call__ runs them (no images)."""
    if h is None:  # first-layers rank: embed
        h = inner.embed_tokens(ids)
        h = mx.contiguous(mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], inner.args.hc_mult, h.shape[2])))
    first = caches[0]
    mask = create_attention_mask(h[:, :, 0, :], first[0] if isinstance(first, CacheList) else first,
                                 window_size=inner.args.sliding_window, return_array=True)
    for layer, c in zip(inner.pipeline_layers, caches):
        h = layer(h, mask, c, ids, causal=True)
    return h


def head(lm, h):
    inner = lm.model
    return linear(lm.lm_head, inner.norm(inner.hc_head(h)))


def lockstep_logits(lm, ids, caches):
    """The normal pipelined forward (both ranks, all_gather inside): logits of the last position."""
    out = lm(mx.array(ids)[None], cache=caches)
    return out.logits[:, -1, :]


def prefill(lm, ids, caches):
    logits = None
    for s in range(0, len(ids), PREFILL):
        logits = lockstep_logits(lm, ids[s:s + PREFILL], caches)
        mx.eval(logits, [c.state for c in caches])
    return int(mx.argmax(logits, axis=-1).item())


def reference(lm, prompt_ids, n):
    """One request at a time, lockstep: greedy tokens and decode seconds."""
    caches = vcache.make_prompt_cache(lm)
    tok = prefill(lm, prompt_ids, caches)
    toks = [tok]
    t = time.perf_counter()
    for _ in range(n):
        logits = lockstep_logits(lm, [toks[-1]], caches)
        mx.eval(logits, *([[c.state for c in caches]] if STATE_EVAL else []))
        toks.append(int(mx.argmax(logits, axis=-1).item()))
    return toks, time.perf_counter() - t


def interleaved(lm, world, prompts, n):
    """Two requests at once. Returns per-request greedy tokens and decode seconds."""
    rank = world.rank()
    inner = lm.model
    caches = [vcache.make_prompt_cache(lm) for _ in prompts]
    toks = [[prefill(lm, ids, c)] for ids, c in zip(prompts, caches)]
    hdt = inner.embed_tokens(mx.array([[0]])).dtype
    shape = (1, 1, inner.args.hc_mult, inner.args.hidden_size)
    received_h = None  # rank 0: hidden state from rank 1, for request (step - 1) % 2
    steps = 2 * n + 1
    prof = {"build": 0.0, "eval": 0.0, "read": 0.0}
    t = time.perf_counter()
    for step in range(steps):
        t_a = time.perf_counter()
        if rank == 1:  # first layers, for request step % 2
            x = step % 2
            if step < 2 * n:
                h = local_layers(inner, mx.array([[toks[x][-1]]]), None, caches[x])
                payload = h.astype(mx.float32)
            else:
                payload = mx.zeros(shape, mx.float32)
        else:          # last layers, for the request rank 1 worked on last step
            payload = mx.zeros(shape, mx.float32)
            if step >= 1:
                y = (step - 1) % 2
                h = local_layers(inner, mx.array([[toks[y][-1]]]), received_h.astype(hdt), caches[y])
                tok = mx.argmax(head(lm, h)[:, -1, :], axis=-1).astype(mx.float32)
                payload = mx.concatenate([tok.reshape(1), payload.reshape(-1)[1:]]).reshape(shape)
        g = mx.distributed.all_gather(payload)  # [2, 1, hc, D]: row 0 from rank 0, row 1 from rank 1
        t_b = time.perf_counter()
        mx.eval(g, *([[c.state for cs in caches for c in cs]] if STATE_EVAL else []))
        t_c = time.perf_counter()
        if step >= 1:  # both ranks learn rank 0's token for request (step - 1) % 2
            toks[(step - 1) % 2].append(int(g[0].reshape(-1)[0].item()))
        received_h = g[1:2]
        t_d = time.perf_counter()
        prof["build"] += t_b - t_a; prof["eval"] += t_c - t_b; prof["read"] += t_d - t_c
    print(f"[rank {rank}] per step: build {prof['build'] / steps * 1e3:.1f} ms, eval+swap {prof['eval'] / steps * 1e3:.1f} ms, "
          f"read token {prof['read'] / steps * 1e3:.1f} ms", flush=True)
    return toks, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--no-state-eval", action="store_true")
    ap.add_argument("prompts", nargs=2)
    args = ap.parse_args()
    global STATE_EVAL
    STATE_EVAL = not args.no_state_eval

    world = mx.distributed.init()
    say = (lambda *a: print(*a, flush=True)) if world.rank() == 0 else (lambda *a: None)
    vpipe.PipelineMixin.pipeline = fixed_pipeline
    model, processor = sharded_load(args.model, pipeline_group=world)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    for _, p in tree_flatten(model.parameters()):  # as the server: page in, one tensor per buffer
        mx.eval(p.sum())
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    lm = model.language_model
    tokenizer = getattr(processor, "tokenizer", processor)
    prompts = [tokenizer.encode(open(p).read(), add_special_tokens=False) for p in args.prompts]
    say(f"loaded; prompts {[len(p) for p in prompts]} tokens; layers {lm.model.start_idx}-{lm.model.end_idx - 1} on rank 0")
    reference(lm, prompts[0][:64], 4)  # warm-up

    # This rank's half alone, no communication: the real cost of half a token.
    inner = lm.model
    scratch = vcache.make_prompt_cache(lm)
    hdt = inner.embed_tokens(mx.array([[0]])).dtype
    ids = mx.array([[1000]])
    h0 = mx.random.normal((1, 1, inner.args.hc_mult, inner.args.hidden_size)).astype(hdt)
    times = []
    for i in range(24):
        t0 = time.perf_counter()
        if world.rank() == 1:
            out = local_layers(inner, ids, None, scratch)
        else:
            out = mx.argmax(head(lm, local_layers(inner, ids, h0, scratch))[:, -1, :], axis=-1)
        t1 = time.perf_counter()
        mx.eval(out, [c.state for c in scratch])
        t2 = time.perf_counter()
        if i >= 4:
            times.append((t1 - t0, t2 - t1))
    b = sum(x[0] for x in times) / len(times) * 1e3
    e = sum(x[1] for x in times) / len(times) * 1e3
    print(f"[rank {world.rank()}] half alone: build {b:.1f} ms + GPU {e:.1f} ms = {b + e:.1f} ms", flush=True)
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))

    ref, ref_s = [], 0.0
    for ids in prompts:
        t, s = reference(lm, ids, args.tokens)
        ref.append(t)
        ref_s += s
    say(f"one at a time: {2 * args.tokens} tokens in {ref_s:.2f}s decode = {2 * args.tokens / ref_s:.1f} tok/s")

    got, s = interleaved(lm, world, prompts, args.tokens)
    say(f"interleaved:   {2 * args.tokens} tokens in {s:.2f}s decode = {2 * args.tokens / s:.1f} tok/s combined, "
        f"{args.tokens / s:.1f} tok/s per request")
    for i in range(2):
        same = got[i] == ref[i]
        k = next((j for j, (a, b) in enumerate(zip(got[i], ref[i])) if a != b), None)
        say(f"request {'AB'[i]}: tokens identical to one-at-a-time: {same}" + ("" if same else f" (first diff at {k})"))
    say("sample A:", repr(tokenizer.decode(got[0][:40])))


if __name__ == "__main__":
    main()
