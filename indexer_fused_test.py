"""Offline: the fused indexer scoring kernel matches the reference scoring and its top-512."""
import mlx.core as mx
from indexer_fused import fused_scores, reference_scores

mx.random.seed(0)
ok = True
for n in (600, 4096, 11000):
    q = mx.random.normal((64, 128)).astype(mx.float32) * 0.1
    w = mx.random.normal((64,)).astype(mx.float32)
    pooled = mx.random.normal((n, 128)).astype(mx.bfloat16)
    a, b = fused_scores(q, w, pooled), reference_scores(q, w, pooled)
    err = mx.abs(a - b).max().item() / (mx.abs(b).max().item() + 1e-9)
    k = min(512, n)
    ta = set(mx.argpartition(-a, kth=k - 1)[:k].tolist())
    tb = set(mx.argpartition(-b, kth=k - 1)[:k].tolist())
    good = err < 1e-4 and len(ta & tb) >= k - 2
    ok &= good
    print(f"Np={n}: max relative error {err:.2e}, top-{k} overlap {len(ta & tb)}/{k} {'ok' if good else 'FAIL'}")
print("all passed" if ok else "FAILED")
