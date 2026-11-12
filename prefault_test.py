"""Checks for paging the weights back in after an idle gap.
usage: python prefault_test.py vlm_cluster_server.py   (needs the server's venv, on a Mac)

macOS unwires idle GPU memory within seconds, even with mx.set_wired_limit
raised, and may then compress or swap it. The first job after a long idle used
to page the weights back in inside GPU work, while the other rank's GPU waited
on it, and the GPU watchdog killed rank 1. Before such a job each rank now
touches its weights one tensor per command buffer, then waits on the CPU.
"""
import importlib.util, sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

spec = importlib.util.spec_from_file_location("srv", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

MODEL = nn.Sequential(nn.Linear(8, 8), nn.QuantizedLinear(64, 64), nn.Linear(8, 2, bias=False))
N_PARAMS = len(tree_flatten(MODEL.parameters()))


def test_first_job_prefaults():
    assert m.needs_prefault(100.0, None, idle=60)

def test_recent_job_skips():
    assert not m.needs_prefault(100.0, 50.0, idle=60)

def test_idle_gap_prefaults():
    assert m.needs_prefault(200.0, 100.0, idle=60)

def test_zero_means_every_job():
    assert m.needs_prefault(100.0, 100.0, idle=0)

def test_negative_means_never():
    assert not m.needs_prefault(1e9, None, idle=-1)


def test_one_command_buffer_per_tensor():
    calls = []
    real = m.mx.eval
    m.mx.eval = lambda *a: (calls.append(a), real(*a))
    try:
        n, touch_s, wait_s = m.prefault(MODEL)
    finally:
        m.mx.eval = real
    assert n == N_PARAMS and touch_s >= 0 and wait_s >= 0
    assert len(calls) == N_PARAMS
    assert all(len(a) == 1 and a[0].size == 1 for a in calls)


def test_barrier_after_all_touches():
    order = []
    real = m.mx.eval
    m.mx.eval = lambda *a: (order.append("touch"), real(*a))
    try:
        m.prefault(MODEL, barrier=lambda: order.append("barrier"))
    finally:
        m.mx.eval = real
    assert order == ["touch"] * N_PARAMS + ["barrier"]


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as e:
            failed += 1
            print("FAIL", name, type(e).__name__, e)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
