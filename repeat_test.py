"""Offline checks for sampling a turn when a tool keeps failing the same way.
usage: python repeat_test.py vlm_cluster_server.py   (needs the server's venv)
"""
import importlib.util, json, sys

spec = importlib.util.spec_from_file_location("srv", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SAME = "No changes to apply: oldString and newString are identical."


def err(message):
    return json.dumps({"error": {"type": "tool.execution", "message": message}, "content": []})


def turn(n, content, name="edit"):
    cid = f"call_{n}"
    return [{"role": "assistant", "content": "",
             "tool_calls": [{"id": cid, "function": {"name": name, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": cid, "content": content}]


def job(messages, **extra):
    body = {"messages": messages, **extra}
    return m.finish_job(body, "prompt", [{"type": "function"}], "dsml")


def fails(n):
    return [x for i in range(n) for x in turn(i, err(SAME))]


def test_no_tool_results():
    assert m.failure_streak([{"role": "user", "content": "hi"}]) == 0

def test_one_failure():
    assert m.failure_streak(fails(1)) == 1

def test_same_error_counts_up():
    assert m.failure_streak(fails(3)) == 3

def test_different_error_restarts():
    assert m.failure_streak(fails(2) + turn(9, err("Could not find oldString"))) == 1

def test_success_ends_streak():
    assert m.failure_streak(fails(2) + turn(9, "Edit applied")) == 0

def test_different_tool_restarts():
    assert m.failure_streak(fails(2) + turn(9, err(SAME), name="write")) == 1

def test_message_between_failures_keeps_streak():
    # OpenCode's own "hit the output limit" message landed inside the B1 streak.
    msgs = fails(1) + [{"role": "user", "content": "limit"}] + turn(9, err(SAME))
    assert m.failure_streak(msgs) == 2

def test_content_parts_are_read():
    parts = [{"type": "text", "text": err(SAME)}]
    assert m.failure_streak(turn(1, parts) + turn(2, parts)) == 2

def test_plain_text_mentioning_error_is_not_a_failure():
    text = json.dumps({"output": "error: tests failed"})
    assert m.failure_streak(turn(1, text) + turn(2, text)) == 0

def test_first_failure_stays_greedy():
    j = job(fails(1))
    assert (j["temperature"], j["top_p"]) == (0.0, 1.0)

def test_repeat_failure_samples():
    j = job(fails(2))
    assert (j["temperature"], j["top_p"]) == (1.0, 0.95)

def test_repeat_failure_is_labelled():
    j = job(fails(2))
    assert j["param_source"]["temperature"] == "repeat"
    assert j["repeat_streak"] == 2

def test_client_temperature_is_respected():
    j = job(fails(3), temperature=0)
    assert j["temperature"] == 0.0

def test_client_top_p_is_kept():
    assert job(fails(2), top_p=0.8)["top_p"] == 0.8

def test_recovery_returns_to_greedy():
    assert job(fails(2) + turn(9, "Edit applied"))["temperature"] == 0.0

def test_can_be_disabled():
    m.RESAMPLE_ON_REPEAT = False
    try:
        assert job(fails(3))["temperature"] == 0.0
    finally:
        m.RESAMPLE_ON_REPEAT = True


bad = 0
tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]
for name, fn in tests:
    try:
        fn()
        print("ok  ", name)
    except Exception as e:
        bad += 1
        print("FAIL", name, type(e).__name__, str(e)[:80])
print(f"{len(tests) - bad}/{len(tests)} passed")
sys.exit(1 if bad else 0)
