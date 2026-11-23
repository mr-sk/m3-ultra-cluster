"""Offline test of the server's copy-spec step (Engine._spec_step) with a fake model.

The fake model gives every position the same next-token distribution p. With a draft of
eight copies of token d, speculative sampling must still emit tokens distributed as p
(after top_p). Checks the first emitted token over many seeds, for a likely and an
unlikely draft token, and that greedy mode accepts exactly while d is the argmax.
usage: python spec_sampling_test.py vlm_cluster_server.py
"""
import importlib.util, sys
from types import SimpleNamespace

import mlx.core as mx

spec = importlib.util.spec_from_file_location("srv", sys.argv[1])
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
srv.HISA_MIN_POOLED = 12_000
P = [0.5, 0.3, 0.1, 0.06, 0.04]
LOGITS = mx.log(mx.array(P))


class FakeLM:
    def __call__(self, x, cache=None):
        n = x.shape[1]
        return SimpleNamespace(logits=mx.broadcast_to(LOGITS, (1, n, len(P))))


class FixedDraft:
    def __init__(self, d):
        self.d, self.updates = d, []

    def draft(self, ctx):
        return [self.d] * 8

    def update(self, drafted, accepted):
        self.updates.append((drafted, accepted))


def emitted(d, seed, temp=1.0, top_p=1.0):
    eng = SimpleNamespace(lm=FakeLM(), indexers=[])
    slot = SimpleNamespace(tokens=[0], prompt_ids=[0] * 20, cache=[], seed=seed,
                           look=FixedDraft(d), job={"temperature": temp, "top_p": top_p},
                           sampler=srv._make_sampler(temp=temp, top_p=top_p))
    return srv.Engine._spec_step(eng, slot), slot.look.updates


ok = True
N = 6000
for d in (1, 3):
    counts = [0] * len(P)
    for seed in range(N):
        new, _ = emitted(d, seed)
        counts[new[0]] += 1
    freq = [c / N for c in counts]
    err = max(abs(f - p) for f, p in zip(freq, P))
    good = err < 0.025
    ok &= good
    print(f"draft token {d}: first emitted token frequencies {[round(f, 3) for f in freq]} vs p {P}: "
          f"max error {err:.3f} {'ok' if good else 'FAIL'}")

# top_p 0.85 keeps tokens 0-2 (0.5 + 0.3 + 0.1 >= 0.85): the emitted distribution is p over them, renormalized
counts = [0] * len(P)
for seed in range(N):
    new, _ = emitted(1, seed, top_p=0.85)
    counts[new[0]] += 1
want = [0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0, 0]
err = max(abs(c / N - w) for c, w in zip(counts, want))
good = err < 0.025
ok &= good
print(f"top_p 0.85: frequencies {[round(c / N, 3) for c in counts]} vs {[round(w, 3) for w in want]}: "
      f"max error {err:.3f} {'ok' if good else 'FAIL'}")

# greedy: draft of the argmax is accepted in full; any other draft token is rejected at once
new, ups = emitted(0, 1, temp=0.0)
good = new == [0] * 9 and ups == [(8, 8)]
ok &= good
print(f"greedy, draft = argmax: {new} {ups} {'ok' if good else 'FAIL'}")
new, ups = emitted(2, 1, temp=0.0)
good = new == [0] and ups == [(8, 0)]
ok &= good
print(f"greedy, draft != argmax: {new} {ups} {'ok' if good else 'FAIL'}")
print("all passed" if ok else "FAILED")
