"""Live check: status lines in the streamed reasoning, chat and thinking modes."""
import json, os, urllib.request
URL = os.environ.get("DSV4_URL", "http://localhost:8090") + "/v1/chat/completions"
def run(extra):
    body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "Name three prime numbers."}], **extra}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    rs, ct = "", ""
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode()
        if line.startswith("data: {"):
            d = json.loads(line[6:])["choices"][0]["delta"]
            rs += d.get("reasoning_content") or ""; ct += d.get("content") or ""
    return rs, ct
for label, extra in (("chat", {}), ("thinking", {"reasoning_effort": "high"})):
    rs, ct = run(extra)
    print(f"--- {label}\nREASONING:\n{rs.strip()}\nANSWER: {ct.strip()!r}")
