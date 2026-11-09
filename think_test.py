"""Offline checks for thinking mode: request mapping, prompt endings, output split."""
import importlib.util, sys
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location("srv", sys.argv[1] + "/vlm_cluster_server.py")
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
ok = True
def check(name, got, want):
    global ok
    good = got == want; ok &= good
    print(f"{'ok ' if good else 'BAD'} {name}: {got!r}" + ("" if good else f"  (want {want!r})"))
rm = srv.reasoning_mode
check("no field -> chat", rm({}), (False, None, None))
check("reasoning_effort none", rm({"reasoning_effort": "none"}), (False, None, None))
check("reasoning_effort low", rm({"reasoning_effort": "low"}), (True, None, 1024))
check("reasoning_effort high", rm({"reasoning_effort": "high"}), (True, None, 4096))
check("reasoning_effort max", rm({"reasoning_effort": "max"}), (True, "max", 12288))
check("reasoning.effort", rm({"reasoning": {"effort": "medium"}}), (True, None, 2048))
check("chat_template_kwargs thinking", rm({"chat_template_kwargs": {"thinking": True}}), (True, None, 4096))
try: rm({"reasoning_effort": "turbo"}); check("bad effort rejected", False, True)
except srv.BadRequest: check("bad effort rejected", True, True)
msgs = [{"role": "user", "content": "hi"}]
check("chat prompt ends", srv.dsv4_prompt(msgs, [])[-25:], "<｜User｜>hi<｜Assistant｜></think>"[-25:])
check("think prompt ends", srv.dsv4_prompt(msgs, [], True, None)[-24:], "<｜User｜>hi<｜Assistant｜><think>"[-24:])
p = srv.dsv4_prompt(msgs, [], True, "max")
check("max effort prompt prefix", p.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"), True)
hist = [{"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>old reasoning</think>a1", "reasoning_content": "old reasoning"},
        {"role": "user", "content": "q2"}]
p = srv.dsv4_prompt(hist, [], True, None)
check("old reasoning dropped (no tools)", "old reasoning" in p, False)
check("think tags stripped from history", p.count("<think>"), 1)
tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
p = srv.dsv4_prompt(hist, tools, True, None)
check("old reasoning kept (tools present)", "old reasoning</think>a1" in p, True)
st = srv.split_think
check("split mid-thought", st("let me think", True), ("let me think", "", False))
check("split partial close held", st("hmm </thi", True), ("hmm ", "", False))
check("split done", st("why\n</think>\n\nAnswer.", True), ("why\n", "Answer.", True))
check("double </think>: final split uses the last", st("r1</think>draft answer. Let me answer.</think>Final.", True, final=True), ("r1</think>draft answer. Let me answer.", "Final.", True))
check("double </think>: streaming split uses the first", st("r1</think>draft", True), ("r1", "draft", True))
check("split chat mode", st("Answer.", False), ("", "Answer.", True))
j = srv.finish_job({}, "P", tools, "dsml", think=True)
check("think defaults", (j["temperature"], j["top_p"], j["max_tokens"]), (1.0, 0.95, 32768))
j = srv.finish_job({"temperature": 0.2}, "P", [], "dsml", think=True)
check("think respects client temp", (j["temperature"], j["top_p"]), (0.2, 1.0))
j = srv.finish_job({}, "P", [], "dsml")
check("chat defaults unchanged", (j["temperature"], j["top_p"], j["max_tokens"], j["think"]), (0.0, 1.0, 8192, False))
check("thinking_budget override", rm({"reasoning_effort": "low", "thinking_budget": 300}), (True, None, 300))
loop_text = "Intro text. " + ("The app reads the label and then the label is read by the app again. " * 3) * 40
check("loop detected", srv.looping(loop_text), True)
code = "\n".join(f"func test_case_{i}() {{ XCTAssertEqual(AppLabel.format(names[{i}]), expected[{i}]) }}" for i in range(80))
check("similar-but-different code is not a loop", srv.looping(code), False)
check("short text is not a loop", srv.looping("abc" * 10), False)
jb = srv.build_job({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high", "max_tokens": 4096}, None, {"model_type": "deepseek_v4"})
check("budget leaves room under client max_tokens", (jb["think_budget"], jb["max_tokens"]), (3072, 4096))
jb = srv.build_job({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high"}, None, {"model_type": "deepseek_v4"})
check("budget unchanged with default max_tokens", jb["think_budget"], 4096)
D = "\uff5cDSML\uff5c"
bad = (f"<{D}tool_calls>\n<{D}invoke name=\"editor\">\n"
       f"<{D}parameter name=\"path\" string=\"true\">a.swift</{D}parameter>\n"
       f"<{D}parameter name=\"old_text\" string=\"true\">let x = 1</{D}parameter>\n"
       f"<{D}parameter name=\"new_text\" string=\"false\">let x = \"two\"\nprint(x)</{D}parameter>\n"
       f"</{D}invoke>\n</{D}tool_calls>")
_, calls = srv.parse_dsml("Editing.\n\n" + bad)
args = __import__("json").loads(calls[0]["function"]["arguments"]) if calls else {}
check("mislabelled code kept as string", args.get("new_text"), 'let x = "two"\nprint(x)')
check("other args intact", (args.get("path"), args.get("old_text")), ("a.swift", "let x = 1"))
etools = [{"type": "function", "function": {"name": "editor", "parameters": {"type": "object",
          "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}, "line": {"type": "integer"}},
          "required": ["path", "old_text", "new_text", "line"]}}}]
calls = [{"id": "c", "type": "function", "function": {"name": "editor", "arguments": '{"path": "a.swift", "old_text": "dup"}'}}]
srv.check_required(calls, etools, "raw")
a2 = __import__("json").loads(calls[0]["function"]["arguments"])
check("missing required args are NOT filled", "new_text" in a2, False)
P = lambda n, v: f'<{D}parameter name="{n}" string="true">{v}</{D}parameter>\n'
call = lambda params: f'\n\n<{D}tool_calls>\n<{D}invoke name="edit">\n' + "".join(params) + f'</{D}invoke>\n</{D}tool_calls>'
_, c1 = srv.parse_dsml("x" + call([P("path", "a.swift"), P("old", "1"), P("new", "2"), P("path", "a.swift")]))
a1 = __import__("json").loads(c1[0]["function"]["arguments"]) if c1 else {}
check("identical duplicate kept once", a1, {"path": "a.swift", "old": "1", "new": "2"})
_, c2 = srv.parse_dsml("x" + call([P("path", "a.swift"), P("old", "1"), P("path", "b.swift")]))
check("conflicting duplicate still rejected", c2, [])
two = f'\n\n<{D}tool_calls>\n<{D}invoke name="a">\n' + P("path", "x") + f'</{D}invoke>\n<{D}invoke name="b">\n' + P("path", "x") + f'</{D}invoke>\n</{D}tool_calls>'
_, c3 = srv.parse_dsml("x" + two)
check("same name in separate invokes untouched", [(x["function"]["name"], x["function"]["arguments"]) for x in c3], [("a", '{"path": "x"}'), ("b", '{"path": "x"}')])
M = srv.STATS_MARK
line1 = srv.stats_start_line(111441, 15085)
check("start line", line1, f"{M} reading 96,356 new tokens (15,085 cached) \u00b7 ~2.2 min to read\n\n")
check("start line, small read has no ETA", srv.stats_start_line(120, 100), f"{M} reading 20 new tokens (100 cached)\n\n")
end = srv.stats_end_line(114037, 111440, 2137, 24809.0, 31.5, 72.6, {"snapshots": 11, "tokens": 153019, "peak_gib": 71.8})
check("end line", end, f"\n\n{M} 2,137 tokens at 31.5 tok/s \u00b7 read 2,597 new + 111,440 cached in 5s \u00b7 73s total \u00b7 cache 11 saved (153k tok) \u00b7 peak 71.8 GiB")
check("strip stats", srv.strip_stats(line1 + "real reasoning" + end), "real reasoning")
check("strip leaves normal text", srv.strip_stats("no marks here"), "no marks here")
h2 = [{"role": "user", "content": "q1"},
      {"role": "assistant", "content": "a1", "reasoning_content": line1 + "think A" + end},
      {"role": "user", "content": "q2"}]
p2 = srv.dsv4_prompt(h2, tools, True, None)
check("stats never reach the model", M in p2, False)
check("real reasoning kept", "think A</think>a1" in p2, True)
h3 = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": line1 + end}, {"role": "user", "content": "q2"}]
check("stats-only reasoning dropped", srv.dsv4_prompt(h3, tools, True, None) == srv.dsv4_prompt(
      [h3[0], {"role": "assistant", "content": "a1"}, h3[2]], tools, True, None), True)
srv.PREFILL_EST[0] = 730.0; srv.update_prefill_estimate(111441, 15085, 814.0)
check("prefill estimate learns", round(srv.PREFILL_EST[0]), round(0.7 * 730 + 0.3 * (96356 / (111441 / 814))))
print("ALL OK" if ok else "FAILURES")
