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
check("no field -> chat", rm({}), (False, None))
check("reasoning_effort none", rm({"reasoning_effort": "none"}), (False, None))
check("reasoning_effort low", rm({"reasoning_effort": "low"}), (True, None))
check("reasoning_effort high", rm({"reasoning_effort": "high"}), (True, None))
check("reasoning_effort max", rm({"reasoning_effort": "max"}), (True, "max"))
check("reasoning.effort", rm({"reasoning": {"effort": "medium"}}), (True, None))
check("chat_template_kwargs thinking", rm({"chat_template_kwargs": {"thinking": True}}), (True, None))
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
print("ALL OK" if ok else "FAILURES")
