"""DeepSeek-V4 indexer, one-token decode: fused scoring kernel (exact top-k).

Stock exact scoring runs ~6 kernels per layer: pooled.astype(float32) (a copy of every
compressed key), q @ pooled^T, ReLU, * scale, * per-head weight, sum over heads; then
argpartition. Here one Metal kernel computes every position's score
    s[p] = sum_h w[h] * relu(scale * q[h] . k[p])
with one SIMD group (32 lanes x 4 dims = 128) per position, then argpartition as before.

install(mode): "fused" uses the kernel for one-token decode; "skip" skips scoring and top-k
entirely (returns a fixed selection: an upper bound on what fusing them could save; output
is wrong in that mode). Other shapes (prefill, multi-token passes) keep mlx-vlm's code.
"""
import mlx.core as mx

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="dsv4_indexer_scores",
            input_names=["q", "w", "k", "n_pos"],
            output_names=["s"],
            source=r"""
            // one simdgroup per position; lane handles dims [4*lane, 4*lane+4)
            uint p = threadgroup_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            if (p >= (uint)n_pos[0]) return;
            const device bfloat16_t* kp = k + p * 128 + lane * 4;
            float4 kv = float4(float(kp[0]), float(kp[1]), float(kp[2]), float(kp[3]));
            float acc = 0.0f;
            for (int h = 0; h < 64; ++h) {
                const device float* qh = q + h * 128 + lane * 4;
                float part = qh[0] * kv.x + qh[1] * kv.y + qh[2] * kv.z + qh[3] * kv.w;
                float dotv = simd_sum(part);
                acc += w[h] * max(dotv, 0.0f);
            }
            if (lane == 0) s[p] = acc;
            """,
        )
    return _KERNEL


def fused_scores(q, w, pooled):
    """q: (64, 128) float32 (already scaled); w: (64,) float32; pooled: (Np, 128) bfloat16.
    Returns (Np,) float32 scores."""
    n = pooled.shape[0]
    return _kernel()(
        inputs=[q, w, pooled, mx.array([n], dtype=mx.int32)],
        grid=(n * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]


def reference_scores(q, w, pooled):
    s = mx.maximum(q.astype(mx.float32) @ pooled.astype(mx.float32).T, 0)  # (64, Np)
    return (s * w[:, None]).sum(axis=0)


def install(mode):
    """Patch the DeepSeek-V4 Indexer's one-token exact path ('fused' or 'skip')."""
    from mlx_vlm.models.deepseek_v4 import language as dl
    orig = dl.Indexer.__call__

    def call(self, x, q_residual, position_rope, pool_cache, offset):
        B, L, _ = x.shape
        if L != 1 or B != 1 or self.index_block != 0 or pool_cache is None:
            return orig(self, x, q_residual, position_rope, pool_cache, offset)
        pooled = self.compressor(x, pool_cache, offset)
        if pooled.shape[1] == 0:
            return None
        Np = pooled.shape[1]
        k = min(self.index_topk, Np)
        pm = pool_cache.make_mask(L, offset)
        if pm is not None:
            pooled_cache_mask[0] = pm
            return orig_tail(self, x, q_residual, position_rope, pooled, offset, k)
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = position_rope(q.transpose(0, 2, 1, 3), offset)  # (1, H, 1, D)
        fixed = mx.arange(k, dtype=mx.uint32)
        if mode == "skip":
            return mx.broadcast_to(fixed[None, None], (1, 1, k))
        w = (self.weights_proj(x).astype(mx.float32) * (self.n_heads ** -0.5)).reshape(-1)
        if mode == "topk-only":  # top-k of a cheap stand-in score: times the selection alone
            s = pooled.reshape(Np, self.head_dim)[:, 0].astype(mx.float32) + w[0]
        elif mode in ("stock-scores", "stock-scores-only"):
            s = (mx.maximum(q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32), 0)
                 * self.scale * w.reshape(1, -1, 1, 1)).sum(axis=1).reshape(-1)
        else:
            s = fused_scores((q.astype(mx.float32) * self.scale).reshape(self.n_heads, self.head_dim),
                             w, pooled.reshape(Np, self.head_dim).astype(mx.bfloat16))
        if mode in ("scores-only", "stock-scores-only"):  # scores computed, selection skipped
            return (fixed + (s[:1] * 0).astype(mx.uint32))[None, None]
        return mx.argpartition(-s, kth=k - 1)[:k].reshape(1, 1, k)

    def orig_tail(self, x, q_residual, position_rope, pooled, offset, k):
        """The rare masked case: mlx-vlm's flat exact scoring, unchanged (the compressor
        has already run, so the original __call__ can't be called again)."""
        B, L, _ = x.shape
        pmask = pooled_cache_mask[0]
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = position_rope(q.transpose(0, 2, 1, 3), offset)
        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads ** -0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        scores = mx.where(pmask if pmask.ndim == 3 else pmask[None], scores, mx.finfo(scores.dtype).min)
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]

    pooled_cache_mask = [None]

    dl.Indexer.__call__ = call
    return lambda: setattr(dl.Indexer, "__call__", orig)
