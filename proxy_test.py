"""Offline checks for record_proxy.py's --rules option.
usage: python3 proxy_test.py record_proxy.py
"""
import importlib.util, json, sys

spec = importlib.util.spec_from_file_location("proxy", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

RULES = "Working rules: act in small steps."
TOOLS = [{"type": "function", "function": {"name": "edit", "parameters": {}}}]


def req(messages, tools=TOOLS):
    body = {"model": "x", "messages": messages}
    if tools is not None:
        body["tools"] = tools
    return json.dumps(body).encode()


def first_user(raw):
    return next(x for x in json.loads(raw)["messages"] if x["role"] == "user")["content"]


def test_rules_go_before_the_first_user_message():
    raw, added = m.add_rules(req([{"role": "system", "content": "sys"},
                                  {"role": "user", "content": "Task: fix it"}]), RULES)
    assert added and first_user(raw) == RULES + "\n\nTask: fix it", first_user(raw)

def test_request_without_tools_is_unchanged():
    # OpenCode's title request has no tools; its prompt must stay as sent.
    body = req([{"role": "user", "content": "Here is the first message"}], tools=None)
    assert m.add_rules(body, RULES) == (body, False)

def test_rules_already_there_are_not_added_twice():
    body = req([{"role": "user", "content": RULES + "\n\nTask: fix it"}])
    assert m.add_rules(body, RULES) == (body, False)

def test_only_the_first_user_message_changes():
    raw, _ = m.add_rules(req([{"role": "user", "content": "Task"},
                              {"role": "assistant", "content": "ok"},
                              {"role": "user", "content": "later"}]), RULES)
    msgs = json.loads(raw)["messages"]
    assert msgs[0]["content"].startswith(RULES) and msgs[2]["content"] == "later"

def test_content_parts_get_rules_in_the_first_text_part():
    raw, added = m.add_rules(req([{"role": "user", "content": [
        {"type": "text", "text": "Task"}, {"type": "text", "text": "more"}]}]), RULES)
    parts = first_user(raw)
    assert added and parts[0]["text"] == RULES + "\n\nTask" and parts[1]["text"] == "more", parts

def test_no_rules_or_bad_json_pass_through():
    body = req([{"role": "user", "content": "Task"}])
    assert m.add_rules(body, None) == (body, False)
    assert m.add_rules(b"not json", RULES) == (b"not json", False)


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
