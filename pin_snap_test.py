"""Offline test: session-start snapshots (the end of the shared system + tools block) are taken
and survive LRU eviction.
usage: python pin_snap_test.py vlm_cluster_server.py
"""
import importlib.util, sys
from types import SimpleNamespace

import mlx.core as mx

spec = importlib.util.spec_from_file_location("srv", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
U = 7
m.BOUNDARY_IDS.add(U)
results = []


def check(name, ok):
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name}")


def fake_cache():
    return [SimpleNamespace(state=[mx.zeros((4,))])]


def session(system_len, task_len, tag):
    """ids of a first request: system + tools (no user token), then a user turn."""
    return [1] * system_len + [U] + [tag] * (task_len - 1)


# 1. the session-start point is the first user token, when the prefix is long enough
ids = session(8000, 400, 2)
check("session start found at the first user token", m._session_start(ids) == 8000)
check("no session start for a short prefix", m._session_start(session(100, 50, 2)) is None)

# 2. the planner adds it to the checkpoints of a cold prefill, and not when it's already cached
pts, pin = m._checkpoints_with_start(ids, 0, len(ids))
check("cold prefill checkpoints include the session start", 8000 in pts and pin == 8000)
pts, pin = m._checkpoints_with_start(ids, 8000, len(ids) - 8000)
check("no session-start checkpoint when the prefix is reused", pin is None and all(p > 0 for p in pts))

# 3. pinned snapshots survive LRU eviction; other snapshots are evicted as before
m.SNAPS.clear()
m.CACHE_MAX_SNAPS, m.CACHE_BUDGET_TOKENS = 6, 10 ** 9
m._add_snapshot(ids[:8000], fake_cache(), pinned=True)
for i in range(20):
    m._add_snapshot([9, i] * 50, fake_cache())
check("pinned snapshot still there after 20 later snapshots",
      any(x["ids"] == ids[:8000] for x in m.SNAPS))
check("cache size limit still holds", len(m.SNAPS) <= 6)

# 4. at most CACHE_MAX_PINNED pinned snapshots: the least recently used pinned one goes
m.SNAPS.clear()
for k in range(m.CACHE_MAX_PINNED + 2):
    m._add_snapshot([k] * 3000, fake_cache(), pinned=True)
pinned = [x for x in m.SNAPS if x.get("pinned")]
check(f"at most {m.CACHE_MAX_PINNED} pinned snapshots", len(pinned) == m.CACHE_MAX_PINNED)
check("the oldest pinned ones were dropped", all(x["ids"][0] >= 2 for x in pinned))

print("all passed" if all(results) else "FAILED")
