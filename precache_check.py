"""Offline: would precaching each reply have matched the next request? (server code path)

For consecutive recorded requests of one session (N, N+1), take N's messages plus the
assistant reply exactly as N+1 sent it back, encode it followed by a dummy tool result
and by a dummy user message, and keep the common token prefix: that is what would be
prefilled and snapshotted after N's reply. A hit means N+1's tokens start with it.
usage (on purple, from the repo): ~/vlm-env/bin/python precache_check.py MODEL_DIR REQUEST.json...
"""
import importlib.util, json, os, sys

spec = importlib.util.spec_from_file_location("srv", "vlm_cluster_server.py")
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
files = sorted(sys.argv[2:])


def ids_for(messages, tools):
    return tok.encode(srv.dsv4_prompt(messages, tools), add_special_tokens=False)


class _Q:
    def put(self, item):
        pass


srv.ENGINE_QUEUE = _Q()


def precache_ids(messages, tools, reply):
    """What the server would precache: its own queue_precache, then the engine's common prefix."""
    srv.PRECACHE.clear()
    srv.queue_precache(messages, tools, reply)
    a, b = (tok.encode(x, add_special_tokens=False) for x in srv.PRECACHE.popleft())
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


reqs = [json.load(open(f)) for f in files]
hits = misses = 0
gain = 0
for cur, nxt in zip(reqs, reqs[1:]):
    m, m2 = cur["messages"], nxt["messages"]
    if len(m2) <= len(m) or m2[: len(m)] != m or m2[len(m)].get("role") != "assistant":
        continue  # not the same session's next turn
    reply = m2[len(m)]
    pre = precache_ids(m, cur.get("tools"), reply)
    nxt_ids = ids_for(m2, nxt.get("tools"))
    p_ids = ids_for(m, cur.get("tools"))
    if nxt_ids[: len(pre)] == pre:
        hits += 1
        gain += len(pre) - len(p_ids)
    else:
        misses += 1
        k = next((i for i, (x, y) in enumerate(zip(pre, nxt_ids)) if x != y), None)
        print(f"miss after {os.path.basename(files[reqs.index(cur)])}: diverges at token {k} of {len(pre)} "
              f"(prompt was {len(p_ids)}): ...{tok.decode(pre[max(0, (k or 0) - 8):(k or 0) + 8])!r} vs "
              f"...{tok.decode(nxt_ids[max(0, (k or 0) - 8):(k or 0) + 8])!r}")
print(f"{hits} hits, {misses} misses; on hits, {gain / max(hits, 1):.0f} extra tokens cached per turn on average")
