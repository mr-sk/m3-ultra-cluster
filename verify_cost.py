"""How much does one forward pass over n tokens cost, and does decode slow down as it goes?
Runs on both ranks with the server stopped:
  cluster_launch.py ... -- python verify_cost.py --model M [--buffer-ops N --buffer-mb N] PROMPT

1. c(n): from a ~15k-token cache, one pass over n tokens copied from earlier in the text,
   with logits for every position (what verifying n draft tokens costs), and the same n
   without logits through the prefill path, for n = 1..128. Median of 3, per rank.
2. Decode rate over a long greedy generation (decay points to compile-cache churn or leaks).
"""
import argparse, copy, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--context", type=int, default=15000)
ap.add_argument("--buffer-ops", type=int, default=0, help="MLX_MAX_OPS_PER_BUFFER (0 = MLX default)")
ap.add_argument("--buffer-mb", type=int, default=0, help="MLX_MAX_MB_PER_BUFFER (0 = MLX default)")
ap.add_argument("--decode-tokens", type=int, default=1200)
ap.add_argument("--skip-sweep", action="store_true")
ap.add_argument("--fuse-l1", action="store_true", help="single-token decode also takes mlx-vlm's fused gate+up / down expert kernels")
ap.add_argument("prompt")
args = ap.parse_args()
if args.buffer_ops:
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = str(args.buffer_ops)
if args.buffer_mb:
    os.environ["MLX_MAX_MB_PER_BUFFER"] = str(args.buffer_mb)
os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "1000")

import mlx.core as mx
from mlx.utils import tree_flatten
import mlx_vlm.models.pipeline as vpipe
from mlx_vlm.models import cache as vcache
from mlx_vlm.utils import sharded_load
from proto_interleave import fixed_pipeline, prefill

NS = [1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32, 48, 64, 96, 128]


def fuse_l1():
    """Let one-token decode use the fused expert kernels mlx-vlm keeps for 2-8 token passes."""
    import inspect, textwrap
    import mlx_vlm.models.fast_ops as fo
    import mlx_vlm.models.switch_layers as sl
    src = inspect.getsource(fo.exact_affine_switch_gate_up)
    assert "or x.shape[1] <= 1\n" in src
    exec(src.replace("or x.shape[1] <= 1\n", "\n"), fo.__dict__)
    src = textwrap.dedent(inspect.getsource(sl.SwitchGLU.__call__))
    assert "1 < x.shape[1] <= DECODE_BLOCK_SIZE" in src
    ns = dict(sl.__dict__)
    exec(src.replace("1 < x.shape[1] <= DECODE_BLOCK_SIZE", "1 <= x.shape[1] <= DECODE_BLOCK_SIZE"), ns)
    sl.SwitchGLU.__call__ = ns["__call__"]


def main():
    world = mx.distributed.init(); rank = world.rank()
    say = lambda *a: print(f"[rank {rank}]", *a, flush=True)
    vpipe.PipelineMixin.pipeline = fixed_pipeline
    if args.fuse_l1:
        fuse_l1()
        say("fused expert kernels on for one-token decode")
    model, processor = sharded_load(args.model, pipeline_group=world)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    for _, p in tree_flatten(model.parameters()):
        mx.eval(p.sum())
    barrier = lambda: mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    barrier()
    lm = model.language_model
    tok = getattr(processor, "tokenizer", processor)
    ids = tok.encode(open(args.prompt).read(), add_special_tokens=False)
    ids = (ids * (args.context // len(ids) + 1))[: args.context]
    base = vcache.make_prompt_cache(lm)
    prefill(lm, ids, base)
    barrier()
    say(f"buffer ops={os.environ.get('MLX_MAX_OPS_PER_BUFFER', 'default')} "
        f"mb={os.environ.get('MLX_MAX_MB_PER_BUFFER', 'default')}; context {len(ids)}")

    def timed(n, logits):
        toks = ids[2000:2000 + n]  # text that already appears in the context: a copy draft
        times = []
        for _ in range(3):
            c = copy.deepcopy(base)
            mx.eval([x.state for x in c])
            barrier()
            t = time.perf_counter()
            out = lm(mx.array(toks)[None], cache=c, skip_logits=not logits)
            mx.eval(out.logits if logits else [x.state for x in c])
            barrier()
            times.append(time.perf_counter() - t)
            del c
        return sorted(times)[1]

    if not args.skip_sweep:
        timed(1, True)  # warm-up
        one = timed(1, True)
        for n in NS:
            v = timed(n, True)
            p = timed(n, False) if n >= 2 else v
            say(f"c({n:3d}): verify {v * 1e3:7.1f} ms = {v / one:5.2f}x one step, {v * 1e3 / n:6.2f} ms/token"
                f" | prefill path {p * 1e3:7.1f} ms, {p * 1e3 / n:6.2f} ms/token")

    # Decode decay: greedy, one token at a time from the same cache.
    c = copy.deepcopy(base)
    y = mx.array([[ids[-1]]])
    barrier()
    t0 = t = time.perf_counter()
    rates = []
    toks = []
    for i in range(1, args.decode_tokens + 1):
        y = mx.argmax(lm(y, cache=c).logits[:, -1, :], axis=-1).reshape(1, 1)
        mx.eval(y)
        toks.append(int(y.item()))
        if i % 200 == 0:
            now = time.perf_counter()
            rates.append(200 / (now - t)); t = now
    say("decode tok/s per 200 tokens: " + ", ".join(f"{r:.1f}" for r in rates)
        + f" | overall {args.decode_tokens / (time.perf_counter() - t0):.1f}"
        + f" | tokens hash {hash(tuple(toks)) & 0xffffffff:08x}, first 12 {toks[:12]}")


main()
