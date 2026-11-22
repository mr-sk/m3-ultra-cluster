"""Where an OpenCode run's steps go: tools per step, repeated reads, test runs, time per step.
usage: python3 runners/step_stats.py RUN.jsonl...   (opencode_run.sh writes RUN.jsonl)"""
import json, sys, collections
tot = collections.Counter(); runs = 0
for f in sys.argv[1:]:
    ev = [json.loads(l) for l in open(f) if l.strip().startswith("{")]
    steps = collections.defaultdict(list); order = []
    for e in ev:
        if e["type"] == "tool_use":
            mid = e["part"]["messageID"]
            if mid not in steps: order.append(mid)
            st = e["part"]["state"]; inp = st.get("input", {})
            key = inp.get("path") or inp.get("filePath") or inp.get("command") or inp.get("pattern") or ""
            steps[mid].append((e["part"]["tool"], str(key)[:70]))
    nsteps = sum(1 for e in ev if e["type"] == "step_start")
    calls = [c for m in order for c in steps[m]]
    reads = collections.Counter(k for t, k in calls if t == "read")
    multi = sum(1 for m in order if len(steps[m]) > 1)
    tools = collections.Counter(t for t, k in calls)
    tests = sum(1 for t, k in calls if t == "bash" and "test" in k)
    builds = sum(1 for t, k in calls if t == "bash" and "build" in k)
    rereads = sum(n - 1 for n in reads.values() if n > 1)
    print(f"{f}: {nsteps} steps, {len(calls)} tool calls, {multi} steps with >1 call; tools {dict(tools)}; "
          f"re-reads {rereads} ({ {k.split('/')[-1]: n for k, n in reads.items() if n > 1} }); test runs {tests}, builds {builds}")
    tot.update(steps=nsteps, calls=len(calls), multi=multi, rereads=rereads, tests=tests, builds=builds, **{"t_" + k: v for k, v in tools.items()}); runs += 1
print("mean per run:", {k: round(v / runs, 1) for k, v in sorted(tot.items())})
