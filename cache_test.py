import json, os, time, urllib.request
URL = os.environ.get("DSV4_URL", "http://localhost:8090") + "/v1/chat/completions"
def chat(messages, **kw):
    body = dict(model="x", messages=messages, max_tokens=80, temperature=0, **kw)
    t = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"], time.time() - t
doc = "\n".join(f"Paragraph {i}: The warehouse in district {i % 17} logged {i * 37 % 1000} crates of item code W-{i:04d} on day {i % 30 + 1}." for i in range(400))
m1 = [{"role": "system", "content": "You are a precise assistant."},
      {"role": "user", "content": doc + "\n\nHow many crates did paragraph 123 log? Answer in one sentence."}]
a1, n1, t1 = chat(m1)
print(f"turn 1  {n1} tokens  {t1:.1f}s  -> {a1!r}")
m2 = m1 + [{"role": "assistant", "content": a1},
           {"role": "user", "content": "And which district was paragraph 250 in? One sentence."}]
a2, n2, t2 = chat(m2)
print(f"turn 2  {n2} tokens  {t2:.1f}s  (should be cached) -> {a2!r}")
a3, n3, t3 = chat(m2, cache=False)
print(f"turn 2' {n3} tokens  {t3:.1f}s  (cache off)          -> {a3!r}")
print("IDENTICAL" if a2 == a3 else "DIFFERENT")
