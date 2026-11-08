"""Multi-snapshot prompt cache: side requests, rewritten middle turns, correctness."""
import json, os, time, urllib.request
BASE = os.environ.get("DSV4_URL", "http://localhost:8090")
def chat(messages, **kw):
    body = dict(model="x", messages=messages, max_tokens=40, temperature=0, **kw)
    t = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"], time.time() - t
def stats():
    return json.load(urllib.request.urlopen(BASE + "/health"))["prompt_cache"]
def doc(tag, n):
    return "\n".join(f"[{tag}] Record {i}: shipment {tag}-{i:04d} left depot {i % 13} carrying {i * 31 % 900} units." for i in range(n))
A = [{"role": "system", "content": "You are a precise assistant. Answer in one short sentence."},
     {"role": "user", "content": doc("ALPHA", 260) + "\n\nAcknowledge ALPHA."},
     {"role": "assistant", "content": "ALPHA received."},
     {"role": "user", "content": doc("BRAVO", 260) + "\n\nAcknowledge BRAVO."},
     {"role": "assistant", "content": "BRAVO received."},
     {"role": "user", "content": doc("CHARLIE", 260) + "\n\nHow many units did shipment CHARLIE-0042 carry?"}]
def step(name, msgs, **kw):
    out, n, t = chat(msgs, **kw)
    print(f"{name:34} {n:6} tok {t:6.1f}s  {out!r:60.60}  cache={stats()}")
    return out
a1 = step("1 conversation, cold", A)
step("2 side request", [{"role": "user", "content": "Name a colour."}])
A2 = A + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And shipment BRAVO-0100?"}]
step("3 follow-up (should hit)", A2)
A3 = [dict(m) for m in A2]; A3[3] = {"role": "user", "content": doc("BRAVO", 250) + "\n\nAcknowledge BRAVO (revised)."}
x = step("4 middle turn rewritten", A3)
y = step("5 same, cache off", A3, cache=False)
print("IDENTICAL" if x == y else "DIFFERENT")
