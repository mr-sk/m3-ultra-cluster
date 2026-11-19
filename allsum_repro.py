"""all_sum latency between two Macs (JACCL), alone vs between GPU kernels.

Run on both nodes, e.g.:
  mlx.launch --backend jaccl --hostfile hosts.json -- python allsum_repro.py
"""
import time

import mlx.core as mx

world = mx.distributed.init()
N = 86  # all_sums per decoded token for DeepSeek-V4-Flash with tensor parallelism (2 per layer x 43)
REPS = 5


def report(name, seconds, count):
    if world.rank() == 0:
        print(f"{name:58s} {seconds / count * 1e6:8.0f} us per all_sum", flush=True)


def best(fn):
    fn()  # warm up
    return min(fn() for _ in range(REPS))


for shape in ((1, 1, 4096), (1, 1, 4, 4096)):
    x = mx.ones(shape, dtype=mx.bfloat16)
    mx.eval(x)
    kib = x.nbytes // 1024
    if world.rank() == 0:
        print(f"\n{shape} bfloat16, {kib} KiB", flush=True)

    def a():  # A: all_sum of an evaluated array, eval each
        t = time.perf_counter()
        for _ in range(N):
            mx.eval(mx.distributed.all_sum(x))
        return time.perf_counter() - t

    def b():  # B: GPU op then all_sum, eval each
        t = time.perf_counter()
        y = x
        for _ in range(N):
            y = mx.distributed.all_sum(y * 0.5)
            mx.eval(y)
        return time.perf_counter() - t

    def c():  # C: the same chain as B, one eval at the end (as inside a model forward pass)
        t = time.perf_counter()
        y = x
        for _ in range(N):
            y = mx.distributed.all_sum(y * 0.5)
        mx.eval(y)
        return time.perf_counter() - t

    def d():  # D: chain as C with the all_sum on the CPU stream
        t = time.perf_counter()
        y = x
        for _ in range(N):
            y = mx.distributed.all_sum(y * 0.5, stream=mx.cpu)
        mx.eval(y)
        return time.perf_counter() - t

    def e():  # E: N GPU ops chained, no all_sum (the kernels alone)
        t = time.perf_counter()
        y = x
        for _ in range(N):
            y = y * 0.5
        mx.eval(y)
        return time.perf_counter() - t

    report("A  all_sum alone, eval each", best(a), N)
    report("B  GPU multiply + all_sum, eval each", best(b), N)
    report("C  GPU multiply + all_sum, 86 chained, one eval", best(c), N)
    report("D  as C, all_sum on stream=mx.cpu", best(d), N)
    report("E  86 GPU multiplies alone, one eval (per multiply)", best(e), N)
