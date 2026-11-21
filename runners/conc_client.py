"""Send recorded OpenCode requests to the server greedily, one at a time or all at once.
usage: python3 conc_client.py URL OUT_PREFIX MODE REQUEST.json...   MODE = seq | par
Writes OUT_PREFIX-<i>.txt (content + tool calls) and prints timings."""
import json, sys, threading, time, urllib.request
url, prefix, mode, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]


def one(i, path, out):
    body = json.load(open(path))
    body.update(stream=False, temperature=0.0, top_p=1.0, max_tokens=96)
    t = time.perf_counter()
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    msg = r["choices"][0]["message"]
    text = (msg.get("content") or "") + "\n" + json.dumps(msg.get("tool_calls") or [], sort_keys=True)
    open(f"{prefix}-{i}.txt", "w").write(text)
    out[i] = (time.perf_counter() - t, r["usage"]["completion_tokens"])


out = {}
t0 = time.perf_counter()
if mode == "seq":
    for i, p in enumerate(files):
        one(i, p, out)
else:
    ts = [threading.Thread(target=one, args=(i, p, out)) for i, p in enumerate(files)]
    for t in ts: t.start()
    for t in ts: t.join()
wall = time.perf_counter() - t0
for i in sorted(out):
    print(f"request {i}: {out[i][1]} tokens in {out[i][0]:.1f}s")
print(f"{mode}: {len(files)} requests, wall {wall:.1f}s")
