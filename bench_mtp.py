"""Speculative decoding with DeepSeek-V4's MTP drafter on the pipelined cluster.

Runs on every rank (start it like the server, with the server stopped):

  ~/cluster-env/bin/python ~/cluster_launch.py --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- \\
    $HOME/vlm-env/bin/python $HOME/mtp-bench/bench_mtp.py \\
    --model $HOME/models/DeepSeek-V4-Flash-0731-4bit --draft $HOME/models/DeepSeek-V4-Flash-0731-mtp \\
    $HOME/mtp-bench/prompt1.txt ...

Each prompt file holds an encoded prompt (the server trace's "prompt" field) and
must exist on every rank. For each prompt: greedy and temperature 1.0, each
without and with the drafter. Greedy outputs must match token for token.
"""
import argparse
import os
import time

# As the server does, before MLX is imported: keep wired weights resident.
os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "1000")

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import stream_generate
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
from mlx_vlm.speculative.utils import format_speculative_stats
from mlx_vlm.utils import sharded_load
import mlx_vlm.models.deepseek_v4.language as dsv4_lang

# DSpark reads the hidden states of layers 40-42. In the pipelined forward pass
# only the rank that owns a layer records it (unlike the final hidden state,
# which is all_gathered), so the other rank's drafter would get nothing and the
# ranks would fall out of lockstep. Share each capture: the owner contributes
# it, every other rank zeros, and all_sum leaves the same list on every rank.
_orig_model_call = dsv4_lang.DeepseekV4Model.__call__


TRACE = {"file": None, "n": 0}


def _caller():
    """The nearest mlx_vlm frame outside the model files: who asked for this forward pass."""
    import traceback
    for fr in reversed(traceback.extract_stack()[:-3]):
        if "mlx_vlm" in fr.filename and "/models/" not in fr.filename:
            return f"{fr.filename.split('mlx_vlm/')[-1]}:{fr.lineno}:{fr.name}"
    return "?"


def _trace(self, inputs, capture_layer_ids):
    if TRACE["file"] is not None:
        TRACE["n"] += 1
        shape = tuple(inputs.shape) if inputs is not None else None
        TRACE["file"].write(f"{TRACE['n']} t={time.perf_counter():.4f} model call inputs={shape} "
                            f"capture={capture_layer_ids} caller={_caller()}\n")
        TRACE["file"].flush()


def _shared_capture_call(self, inputs, cache=None, inputs_embeds=None, hidden_sink=None,
                         skip_final_norm=False, capture_layer_ids=None, capture_sink=None):
    _trace(self, inputs, capture_layer_ids)
    if not capture_layer_ids or getattr(self, "pipeline_size", 1) <= 1 or capture_sink is None:
        return _orig_model_call(self, inputs, cache, inputs_embeds, hidden_sink, skip_final_norm,
                                capture_layer_ids, capture_sink)
    local = []
    out = _orig_model_call(self, inputs, cache, inputs_embeds, hidden_sink, skip_final_norm,
                           capture_layer_ids, local)
    lo, hi = self.start_idx, self.start_idx + len(self.pipeline_layers)
    mine = iter(local)
    for layer_id in sorted(set(capture_layer_ids)):
        if lo <= layer_id < hi:
            value = next(mine)
        else:
            value = mx.zeros((*out.shape[:2], self.args.hidden_size), dtype=out.dtype)
        # MLX runs collectives in dependency order. The zeros depend on nothing, so
        # without this the non-owner's all_sum went out before its pipeline send while
        # the owner waited in recv for that send: deadlock (2026-11-15). Depending on
        # the final output puts send/recv, then all_gather, then these, on every rank.
        value = mx.depends(value, out)
        if TRACE["file"] is not None:
            TRACE["file"].write(f"   capture {layer_id}: {'own' if lo <= layer_id < hi else 'zeros'} "
                                f"shape={tuple(value.shape)} dtype={value.dtype}\n")
        capture_sink.append(mx.distributed.all_sum(value))
    return out


dsv4_lang.DeepseekV4Model.__call__ = _shared_capture_call


def fixed_pipeline(self, group):
    """mlx-lm 0.32.2's split. mlx-vlm 0.7.12's formula drops a layer when the layer count
    doesn't divide evenly (43 layers on 2 ranks: layer 21 runs nowhere)."""
    self.pipeline_rank, self.pipeline_size = group.rank(), group.size()
    base, extra = divmod(len(self.layers), self.pipeline_size)
    split = [base + (1 if r < extra else 0) for r in range(self.pipeline_size)]
    self.start_idx = sum(split[self.pipeline_rank + 1:])
    self.end_idx = self.start_idx + split[self.pipeline_rank]
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx


def mean_nll(model, processor, text, n_tokens):
    """Average negative log-likelihood per token of `text`, read in 512-token chunks."""
    from mlx_vlm.models import cache as vcache
    tok = getattr(processor, "tokenizer", processor)
    ids = tok.encode(text)[:n_tokens]
    lm = model.language_model
    c = vcache.make_prompt_cache(lm)
    total, n = 0.0, 0
    for s in range(0, len(ids) - 1, 512):
        chunk = ids[s:s + 512]
        targets = ids[s + 1:s + 1 + len(chunk)]
        out = lm(mx.array(chunk)[None], cache=c)
        lg = out.logits[0, :len(targets)].astype(mx.float32)
        nll = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, mx.array(targets)[:, None], axis=-1)[:, 0]
        total += nll.sum().item()
        n += len(targets)
    return total / n, n

PARTS = {}


def _arrays(tree):
    import mlx.utils as mu
    if isinstance(tree, mx.array):
        return [tree]
    if hasattr(tree, "__dict__") and not isinstance(tree, (list, tuple, dict)):
        tree = {k: v for k, v in vars(tree).items()}
    return [v for _, v in mu.tree_flatten(tree) if isinstance(v, mx.array)]


def time_part(name, fn):
    """Wrap fn: force its outputs to finish and add up the wall time (attribution only)."""
    def wrapped(*a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        mx.eval(_arrays(out))
        n, tot = PARTS.get(name, (0, 0.0))
        PARTS[name] = (n + 1, tot + time.perf_counter() - t0)
        return out
    return wrapped


def install_timers():
    import mlx_vlm.speculative.dflash as dflash
    from mlx_vlm.speculative.drafters.deepseek_v4_dspark.deepseek_v4_dspark import DeepseekV4DsparkDraftModel
    dflash.verify_forward = time_part("target verify (5 tokens, both Studios)", dflash.verify_forward)
    DeepseekV4DsparkDraftModel.draft_block = time_part("drafter draft_block", DeepseekV4DsparkDraftModel.draft_block)


def run(model, processor, prompt, max_tokens, temperature, seed, draft=None, kind=None, block=None):
    mx.random.seed(seed)
    kw = {"draft_model": draft, "draft_kind": kind} if draft is not None else {}
    if draft is not None and block:
        kw["draft_block_size"] = block
    text, t_first, first_n, last = "", None, 0, None
    for r in stream_generate(model, processor, prompt, max_tokens=max_tokens, temperature=temperature,
                             top_p=0.95 if temperature else 1.0, prefill_step_size=512, **kw):
        if t_first is None:  # decode timing starts after the first yield (prefill done)
            t_first, first_n = time.perf_counter(), r.generation_tokens
        text += r.text
        last = r
    wall = time.perf_counter() - t_first
    n = last.generation_tokens if last else 0
    return text, n, (n - first_n) / wall if wall > 0 and n > first_n else 0.0, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--no-draft", action="store_true", help="load only the target (diagnostics)")
    ap.add_argument("--draft-bits", type=int, default=4, help="quantize the drafter (0: keep as loaded)")
    ap.add_argument("--diag", action="store_true", help="stop after the warm-up")
    ap.add_argument("--trace", action="store_true", help="per-rank call trace and stack dumps")
    ap.add_argument("--time-parts", action="store_true", help="time verify and drafter separately")
    ap.add_argument("--tensor", action="store_true", help="tensor parallel (tensor_group) instead of pipeline")
    ap.add_argument("--allsum-bench", action="store_true", help="time small all_sums before loading")
    ap.add_argument("--prefault", action="store_true", help="touch every weight before the warm-up")
    ap.add_argument("--fix-split", action="store_true", help="mlx-lm's pipeline split (no dropped layer)")
    ap.add_argument("--nll", default="", help="score this text file: mean negative log-likelihood per token")
    ap.add_argument("--nll-tokens", type=int, default=3072)
    ap.add_argument("--save-text", default="", help="rank 0 writes each run's output to PREFIX-<label>.txt")
    ap.add_argument("--only", choices=["greedy-draft", "greedy", "nll"], help="greedy-draft: drafter only; "
                    "greedy: base greedy, then the drafter at each --blocks size")
    ap.add_argument("--blocks", default="", help="comma list of draft block sizes for --only greedy")
    ap.add_argument("prompts", nargs="*")
    args = ap.parse_args()

    world = mx.distributed.init()
    say = (lambda *a: print(*a, flush=True)) if world.rank() == 0 else (lambda *a: None)
    if args.trace:
        import faulthandler
        base = os.path.expanduser(f"~/mtp-bench/rank{world.rank()}")
        TRACE["file"] = open(base + ".trace.txt", "w")
        stacks = open(base + ".stacks.txt", "w")
        faulthandler.dump_traceback_later(240, repeat=True, file=stacks)  # stuck 4 min: dump stacks

    def mem(stage):  # every rank reports, so a memory problem shows on the rank it hits
        gib = 2 ** 30
        print(f"[rank {world.rank()}] {stage}: active {mx.get_active_memory() / gib:.1f} GiB, "
              f"peak {mx.get_peak_memory() / gib:.1f} GiB", flush=True)

    if args.allsum_bench:
        for shape in ((1, 1, 4096), (1, 1, 4, 4096), (1, 512, 4096)):
            x = mx.ones(shape, dtype=mx.bfloat16)
            mx.eval(mx.distributed.all_sum(x))
            t0 = time.perf_counter()
            for _ in range(200):
                mx.eval(mx.distributed.all_sum(x))
            one = (time.perf_counter() - t0) / 200
            t0 = time.perf_counter()
            y = x
            for _ in range(86):  # one token's worth, chained in one graph
                y = mx.distributed.all_sum(y) * 0.5
            mx.eval(y)
            chain = time.perf_counter() - t0
            say(f"all_sum {shape} bf16 ({x.nbytes / 1024:.0f} KiB): {one * 1e6:.0f} us each with a sync; "
                f"86 chained: {chain * 1e3:.1f} ms")
    if args.fix_split:
        import mlx_vlm.models.pipeline as vpipe
        vpipe.PipelineMixin.pipeline = fixed_pipeline
    if args.tensor:
        # mlx-vlm 0.7.12 defines shard() on DeepSeek-V4's LanguageModel, but sharded_load
        # looks for it on the outer Model and refuses tensor parallelism without it.
        import mlx_vlm.models.deepseek_v4.deepseek_v4 as dsv4_top
        if not hasattr(dsv4_top.Model, "shard"):
            dsv4_top.Model.shard = lambda self, group=None: self.language_model.shard(group)
        model, processor = sharded_load(args.model, tensor_group=world)
    else:
        model, processor = sharded_load(args.model, pipeline_group=world)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])  # as the server
    inner = model.language_model.model
    if not args.tensor:
        held = [i for i, l in enumerate(inner.layers) if l is not None and inner.start_idx <= i < (inner.end_idx or 10**9)]
        print(f"[rank {world.rank()}] layers {inner.start_idx}..{inner.end_idx} -> holds {len(held)}: "
              f"{held[0]}-{held[-1]}", flush=True)
    mem("model loaded")
    draft, kind = (None, None) if args.no_draft else load_drafter(args.draft)
    if draft is not None:
        if args.draft_bits:
            # Unquantized the drafter is ~10 GiB, too much next to a 70 GiB half-model
            # under an 84 GiB wired limit (GPU timeout in the warm-up, 2026-11-15).
            nn.quantize(draft, group_size=64, bits=args.draft_bits,
                        class_predicate=lambda _, m: hasattr(m, "to_quantized") and not isinstance(m, nn.Embedding))
        mx.eval(draft.parameters())
        validate_drafter_compatibility(model, draft, kind)
    mem("drafter loaded")
    if args.prefault:
        # As the server: touch every weight, one tensor per command buffer, then wait for
        # the other rank, so no GPU buffer pages in more than one tensor (GPU timeout otherwise).
        from mlx.utils import tree_flatten
        t0 = time.perf_counter()
        for _, prm in tree_flatten(model.parameters()):
            mx.eval(prm.sum())
        print(f"[rank {world.rank()}] prefaulted in {time.perf_counter() - t0:.1f}s", flush=True)
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    mx.eval(mx.distributed.all_sum(mx.array(1.0)))  # both ranks here before any pipeline traffic
    say(f"loaded; drafter kind {kind}; ranks {world.size()}")
    run(model, processor, "Hi", 4, 0.0, 0)  # warm-up, as the server does
    mem("after warm-up")
    if args.nll:
        nll, n = mean_nll(model, processor, open(args.nll).read(), args.nll_tokens)
        say(f"NLL over {n} tokens: {nll:.4f} nats/token (perplexity {__import__('math').exp(nll):.3f})")
    if args.diag or args.only == "nll":
        return
    if args.time_parts:
        install_timers()

    for path in args.prompts:
        prompt = open(path).read()
        say(f"\n== {path}")
        base_greedy = None
        if args.only == "greedy-draft":
            configs = [(0.0, True, args.block_size)]
        elif args.only == "greedy":
            configs = [(0.0, False, None)] + [(0.0, True, int(b)) for b in args.blocks.split(",") if b]
        else:
            configs = [(t, u, args.block_size) for t in (0.0, 1.0) for u in (False, True)]
        for temp, use, block in configs:
            if True:
                toks, n, tps, last = run(model, processor, prompt, args.max_tokens, temp, 1234,
                                         draft if use else None, kind if use else None, block)
                label = f"temp {temp} {'MTP block ' + str(block) if use else 'base'}"
                extra = ""
                if use:
                    extra = f" | {format_speculative_stats(draft)}"
                if temp == 0.0:
                    if not use:
                        base_greedy = toks
                    elif base_greedy is not None:
                        same = toks == base_greedy
                        k = next((i for i, (a, b) in enumerate(zip(toks, base_greedy)) if a != b),
                                 min(len(toks), len(base_greedy)))
                        extra += f" | greedy match: {same}" + ("" if same else f" (first diff at char {k})")
                if args.save_text and world.rank() == 0:
                    with open(f"{args.save_text}-{label.replace(' ', '_')}.txt", "w") as f:
                        f.write(toks)
                say(f"{label}: {n} tokens, {tps:.1f} tok/s (decode wall), "
                    f"prompt {last.prompt_tokens if last else '?'}{extra}")
                for part, (cnt, tot) in sorted(PARTS.items()):
                    print(f"[rank {world.rank()}]   {part}: {cnt} calls, {tot / max(cnt, 1) * 1000:.1f} ms each, "
                          f"{tot:.2f} s total", flush=True)
                PARTS.clear()


if __name__ == "__main__":
    main()
