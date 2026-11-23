"""Offline: how much of the model's output copies text already in its prompt?

For consecutive recorded requests of one session (N, N+1), the model's output for N is the
assistant turn N+1 sent back, encoded exactly as the server encodes it. Each output is split
into verbatim runs found in N's prompt (greedy longest match, runs of at least MIN tokens).
A lookup drafter must see the first few tokens of a run before it can find it, so the
"draftable" count leaves out the first LEAD tokens of each run.
usage (on purple, from the repo): ~/vlm-env/bin/python copy_check.py MODEL_DIR REQUEST.json...
"""
import collections, importlib.util, json, sys

spec = importlib.util.spec_from_file_location("srv", "vlm_cluster_server.py")
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
MIN, LEAD = 8, 4
enc = lambda msgs, tools: tok.encode(srv.dsv4_prompt(msgs, tools), add_special_tokens=False)


def runs(out, ctx):
    index = collections.defaultdict(list)
    for p in range(len(ctx) - MIN + 1):
        index[tuple(ctx[p:p + MIN])].append(p)
    i, found = 0, []
    while i < len(out):
        best = 0
        for p in index.get(tuple(out[i:i + MIN]), [])[-200:]:
            L = MIN
            while i + L < len(out) and p + L < len(ctx) and out[i + L] == ctx[p + L]:
                L += 1
            best = max(best, L)
        if best:
            found.append(best); i += best
        else:
            i += 1
    return found


reqs = [json.load(open(f)) for f in sorted(sys.argv[2:])]
tot = collections.Counter()
by_kind = collections.defaultdict(collections.Counter)
for cur, nxt in zip(reqs, reqs[1:]):
    m, m2 = cur["messages"], nxt["messages"]
    if len(m2) <= len(m) or m2[:len(m)] != m or m2[len(m)].get("role") != "assistant":
        continue
    reply = m2[len(m)]
    tools = cur.get("tools")
    a = enc(m, tools)
    b = enc(m + [reply, {"role": "user", "content": "y"}], tools)
    if b[:len(a)] != a:
        tot["skipped"] += 1
        continue
    tail = enc([{"role": "user", "content": "y"}], None)  # strip the dummy user turn's tokens
    out = b[len(a):]
    k = 0
    while k < min(len(out), len(tail)) and out[-1 - k] == tail[-1 - k]:
        k += 1
    out = out[:len(out) - k] if k else out
    rs = runs(out, a)
    kind = "+".join(sorted({c["function"]["name"] for c in reply.get("tool_calls") or []})) or "text"
    for key, c in (("all", tot), (kind, by_kind[kind])):
        c["turns"] += 1; c["out"] += len(out)
        c["in8"] += sum(rs)
        c["in16"] += sum(L for L in rs if L >= 16)
        c["in32"] += sum(L for L in rs if L >= 32)
        c["draftable"] += sum(L - LEAD for L in rs)
        c["runs"] += len(rs)


def show(name, c):
    o = max(c["out"], 1)
    print(f"{name:>18}: {c['turns']:4d} turns, {c['out']:6d} output tokens ({c['out'] / max(c['turns'], 1):.0f}/turn); "
          f"in copied runs >=8: {c['in8'] / o:.0%}, >=16: {c['in16'] / o:.0%}, >=32: {c['in32'] / o:.0%}; "
          f"draftable {c['draftable'] / o:.0%}; mean run {c['in8'] / max(c['runs'], 1):.0f} tokens")


show("all", tot)
for k, c in sorted(by_kind.items(), key=lambda x: -x[1]["out"]):
    show(k, c)
print("skipped (encoding mismatch):", tot["skipped"])
