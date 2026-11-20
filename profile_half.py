"""Where does each rank's half of a decoded token spend its time?  (run on both ranks, server stopped)
usage: cluster_launch.py ... -- python profile_half.py --model M PROMPT"""
import argparse, os, time
os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "1000")
import mlx.core as mx
from mlx.utils import tree_flatten
import mlx_vlm.models.pipeline as vpipe
import mlx_vlm.models.deepseek_v4.language as dl
from mlx_vlm.models import cache as vcache
from mlx_vlm.utils import sharded_load
from proto_interleave import fixed_pipeline, local_layers, head, prefill

N = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("prompt")
    args = ap.parse_args()
    world = mx.distributed.init(); rank = world.rank()
    vpipe.PipelineMixin.pipeline = fixed_pipeline
    model, processor = sharded_load(args.model, pipeline_group=world)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    for _, p in tree_flatten(model.parameters()):
        mx.eval(p.sum())
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    lm = model.language_model; inner = lm.model
    tok = getattr(processor, "tokenizer", processor)
    ids = tok.encode(open(args.prompt).read(), add_special_tokens=False)[: args.context]
    caches = vcache.make_prompt_cache(lm)
    prefill(lm, ids, caches)  # lockstep, both ranks: a realistic cache
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    hdt = inner.embed_tokens(mx.array([[0]])).dtype
    x = mx.array([[1000]])
    h0 = mx.random.normal((1, 1, inner.args.hc_mult, inner.args.hidden_size)).astype(hdt)
    mx.eval(h0)

    def step():
        if rank == 1:
            return local_layers(inner, x, None, caches)
        return mx.argmax(head(lm, local_layers(inner, x, h0, caches))[:, -1, :], axis=-1)

    for _ in range(3):
        mx.eval(step())
    t = time.perf_counter()
    for _ in range(N):
        mx.eval(step())
    synced = (time.perf_counter() - t) / N
    t = time.perf_counter()
    out = None
    for _ in range(N):
        out = step()
        mx.async_eval(out)
    mx.eval(out)
    pipelined = (time.perf_counter() - t) / N

    # per component: force an eval after attention and after the expert layer, in every layer
    parts = {"attn": 0.0, "ffn": 0.0}
    by_type = {}  # attention time per layer kind: (class name, compress ratio) -> [seconds, layers]
    orig = dl.DeepseekV4Block.__call__

    def timed(self, h, mask, cache, input_ids, position_offset=None, causal=False):
        mx.eval(h)
        t0 = time.perf_counter()
        h = self.attn_hc.apply_branch(h, self.attn_norm, self.attn, mask=mask, cache=cache,
                                      position_offset=position_offset, causal=causal)
        mx.eval(h)
        t1 = time.perf_counter()
        h = self.ffn_hc.apply_branch(h, self.ffn_norm, self.ffn, input_ids)
        mx.eval(h)
        t2 = time.perf_counter()
        parts["attn"] += t1 - t0; parts["ffn"] += t2 - t1
        key = (type(self.attn).__name__, getattr(self.attn, "compress_ratio", None))
        slot = by_type.setdefault(key, [0.0, 0])
        slot[0] += t1 - t0; slot[1] += 1
        return h

    dl.DeepseekV4Block.__call__ = timed
    for _ in range(N):
        mx.eval(step())
    dl.DeepseekV4Block.__call__ = orig
    # cost of one forced eval of a tiny op, to subtract
    t = time.perf_counter()
    for _ in range(200):
        mx.eval(h0 * 1.0)
    one_eval = (time.perf_counter() - t) / 200
    n_layers = len([l for l in inner.pipeline_layers])
    a = parts["attn"] / N - n_layers * one_eval
    f = parts["ffn"] / N - n_layers * one_eval
    print(f"[rank {rank}] {n_layers} layers, context {len(ids)}: synced {synced * 1e3:.1f} ms | pipelined {pipelined * 1e3:.1f} ms | "
          f"attention {a * 1e3:.1f} ms, experts {f * 1e3:.1f} ms (forced eval {one_eval * 1e6:.0f} us each, subtracted)", flush=True)
    for (cls, ratio), (sec, calls) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        layers = calls // N
        per_layer = sec / calls - one_eval
        print(f"[rank {rank}]   {cls} ratio {ratio}: {layers} layers, {per_layer * 1e3:.2f} ms per layer, "
              f"{per_layer * layers * 1e3:.1f} ms per half", flush=True)


if __name__ == "__main__":
    main()
