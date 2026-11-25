"""What real use of the cluster looked like: a report from the server logs and the proxy's
recordings, for deciding what to improve next.

usage: python3 runners/session_report.py [--days 7] [--proxy-dir ~/dsv4-proxy]
Reads the server logs on the head node over ssh (~/dsv4-server.log and ~/dsv4-logs/server/*.log
changed in the window; settings from ~/.config/m3-cluster.env as for the other runners) and the
proxy's recordings (<proxy-dir>/<date>/*.request.json) in the window. Read-only.
"""
import argparse, collections, datetime as dt, glob, json, os, re, shlex, statistics, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=7)
ap.add_argument("--proxy-dir", default=os.path.expanduser("~/dsv4-proxy"))
args = ap.parse_args()


def conf():
    env = {}
    path = os.environ.get("CLUSTER_ENV", os.path.expanduser("~/.config/m3-cluster.env"))
    for line in open(path):
        m = re.match(r"\s*([A-Z_]+)=(.*)", line)
        if m:
            env[m[1]] = os.path.expandvars(m[2].strip().strip('"').strip("'"))
    return env


C = conf()
since = dt.datetime.now() - dt.timedelta(days=args.days)

# ---- server logs (one ssh call: every log changed in the window, with a file marker)
cmd = (f"find ~/dsv4-server.log ~/dsv4-logs/server -name '*.log' -newermt "
       f"{shlex.quote(since.strftime('%Y-%m-%d %H:%M:%S'))} -print0 2>/dev/null | "
       "xargs -0 -I{} sh -c 'echo \"@@FILE {}\"; cat \"{}\"'")
raw = subprocess.run(["ssh", "-o", "BatchMode=yes", C["HEAD"], cmd], capture_output=True, text=True).stdout
REQ = re.compile(r"\] chatcmpl-\w+: (?P<tags>(?:\[[^\]]*\] )*)(?P<p>\d+) prompt \((?P<c>\d+) cached\) \+ (?P<g>\d+) gen tokens in "
                 r"(?P<t>[\d.]+)s \(prefill (?P<pr>\d+) tok/s, decode (?P<dr>[\d.]+) tok/s\)(?P<rest>.*)")
reqs, spec, events, seen = [], [], collections.Counter(), set()
for block in raw.split("@@FILE ")[1:]:
    name, _, body = block.partition("\n")
    for line in body.splitlines():
        m = REQ.search(line)
        if m:
            key = line[:60]
            if key in seen:  # the live log and a rotated copy can overlap
                continue
            seen.add(key)
            d = {k: (float(v) if k in "tprdr" or k in ("t", "pr", "dr") else v) for k, v in m.groupdict().items()}
            for k in ("p", "c", "g"):
                d[k] = int(d[k])
            d["t"], d["pr"], d["dr"] = float(m["t"]), float(m["pr"]), float(m["dr"])
            reqs.append(d)
            continue
        if "copy-spec:" in line:
            m2 = re.search(r"(\d+) verify passes, (\d+)/(\d+) draft tokens accepted", line)
            if m2:
                spec.append(tuple(map(int, m2.groups())))
        for key, pat in (("tool-call parse failures (DSML)", "DSML tool-call parse failed"),
                         ("loop stops", "LOOP stopped"), ("tracebacks", "Traceback"),
                         ("GPU timeouts", "GPU Timeout"), ("server starts", "serving on"),
                         ("long outputs (2000+ tokens)", "long output after")):
            if pat in line:
                events[key] += 1
        if "[client gone]" in line:
            events["client went away mid-reply"] += 1

# ---- proxy recordings: sessions are chains of requests whose messages extend the previous one
recs = []
for day in sorted(glob.glob(os.path.join(args.proxy_dir, "20*-*-*"))):
    try:
        if dt.datetime.strptime(os.path.basename(day), "%Y-%m-%d") < since.replace(hour=0, minute=0, second=0):
            continue
    except ValueError:
        continue
    for f in sorted(glob.glob(os.path.join(day, "*.request.json"))):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        recs.append((f, r))
sessions = []  # [n_requests, max_messages]
open_chains = []
for f, r in recs:
    msgs = r.get("messages") or []
    for ch in open_chains:
        if len(msgs) >= len(ch["last"]) and msgs[:len(ch["last"])] == ch["last"]:
            ch["n"] += 1; ch["last"] = msgs; ch["max_msgs"] = len(msgs)
            break
    else:
        open_chains.append({"n": 1, "last": msgs, "max_msgs": len(msgs), "tools": bool(r.get("tools"))})
sessions = [ch for ch in open_chains if ch["tools"]]


# ---- report
def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


print(f"# Cluster use, last {args.days} days (since {since:%Y-%m-%d %H:%M})\n")
if not reqs:
    print("No requests in the server logs for this window.")
else:
    dec = [r["g"] / r["dr"] for r in reqs if r["dr"] > 0]
    pre = [r["p"] / r["pr"] for r in reqs if r["pr"] > 0]
    gen = sum(r["g"] for r in reqs)
    new = [r["p"] - r["c"] for r in reqs]
    print("## Volume and time")
    print(f"- {len(reqs)} requests, {gen:,} tokens generated, {sum(r['p'] for r in reqs):,} prompt tokens "
          f"({sum(new):,} of them read new)")
    print(f"- model time: decode {sum(dec) / 60:.0f} min, prefill {sum(pre) / 60:.0f} min "
          f"(prefill is {pct(sum(pre), sum(pre) + sum(dec))} of the model's time)")
    print(f"- per request: {q([r['t'] for r in reqs], .5):.1f} s median, {q([r['t'] for r in reqs], .9):.1f} s at p90, "
          f"{max(r['t'] for r in reqs):.0f} s max")
    print("\n## Speed")
    print(f"- decode, token-weighted: {gen / sum(dec):.1f} tok/s" if dec else "- decode: -")
    if spec:
        p_, a_, d_ = (sum(x[i] for x in spec) for i in range(3))
        print(f"- copy-spec: {p_:,} verify passes, {pct(a_, d_)} of {d_:,} draft tokens accepted "
              f"({len(spec)} requests used it)")
    print(f"- prompt cache: {pct(sum(r['c'] for r in reqs), sum(r['p'] for r in reqs))} of prompt tokens came from the cache; "
          f"median new tokens per request {q(new, .5):,}, p90 {q(new, .9):,}")
    cold = [r for r in reqs if r["c"] == 0 and r["p"] > 2000]
    if cold:
        print(f"- {len(cold)} cold requests (nothing cached, 2k+ tokens): {sum(r['p'] / r['pr'] for r in cold if r['pr']):.0f} s of prefill")
    big = [r for r in reqs if r["p"] - r["c"] > 10000]
    if big:
        print(f"- {len(big)} requests read 10k+ new tokens (large tool outputs or files): "
              f"{sum(r['p'] / r['pr'] for r in big if r['pr']):.0f} s of prefill")
    print(f"- context: median prompt {q([r['p'] for r in reqs], .5):,} tokens, p90 {q([r['p'] for r in reqs], .9):,}, "
          f"max {max(r['p'] for r in reqs):,}" + (" (copy-spec turns off above 64k)" if max(r['p'] for r in reqs) > 64000 else ""))
    tools = collections.Counter(t.strip() for r in reqs for grp in re.findall(r"\[tool ([^\]]+)\]", r["tags"])
                                for t in grp.split(","))
    multi = sum(1 for r in reqs for grp in re.findall(r"\[tool ([^\]]+)\]", r["tags"]) if "," in grp)
    if tools:
        print(f"- tool calls: " + ", ".join(f"{k} {v}" for k, v in tools.most_common(8))
              + f"; {multi} replies made more than one call")
print("\n## Problems")
if events:
    for k, v in sorted(events.items()):
        print(f"- {k}: {v}")
else:
    print("- none logged")
print("\n## Sessions (from the proxy recordings)")
if sessions:
    ns = [s["n"] for s in sessions]
    print(f"- {len(sessions)} agent sessions; turns per session: median {q(ns, .5)}, p90 {q(ns, .9)}, max {max(ns)}")
else:
    print("- no agent sessions recorded in this window")
print(f"\n(notes: {len(recs)} recorded requests read; logs include test batches and benchmarks run in the window)")
