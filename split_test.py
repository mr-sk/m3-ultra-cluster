"""Offline check that every layer runs on exactly one pipeline rank.
usage: python split_test.py vlm_cluster_server.py   (needs the server's venv)

mlx-vlm 0.7.12's PipelineMixin.pipeline drops a layer when the layer count doesn't
divide evenly: DeepSeek-V4-Flash has 43 layers, and on 2 ranks layer 21 ran nowhere.
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("srv", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Group:
    def __init__(self, rank, size):
        self._r, self._s = rank, size

    def rank(self):
        return self._r

    def size(self):
        return self._s


class Stack:
    def __init__(self, n):
        self.layers = list(range(n))


def held(n, rank, size):
    s = Stack(n)
    m.fixed_pipeline(s, Group(rank, size))
    return [l for l in s.layers[s.start_idx:s.end_idx] if l is not None]


def check(n, size):
    parts = [held(n, r, size) for r in range(size)]
    flat = sorted(i for p in parts for i in p)
    assert flat == list(range(n)), f"{n} layers on {size} ranks: {parts}"
    assert parts[0][-1] == n - 1, "rank 0 must run the last layers"


bad = 0
for n in (42, 43, 61, 62):
    for size in (2, 3):
        try:
            check(n, size)
            print("ok  ", n, "layers on", size, "ranks")
        except Exception as e:
            bad += 1
            print("FAIL", n, "layers on", size, "ranks:", type(e).__name__, str(e)[:100])
print(f"{8 - bad}/8 passed")
sys.exit(1 if bad else 0)
