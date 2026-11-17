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


EDIT_TOOLS = [{"type": "function", "function": {"name": "edit", "parameters": {
    "properties": {"path": {}, "oldString": {}, "newString": {}}, "required": ["path", "oldString", "newString"]}}}]
OLD = "    private let key = \"virtualdesktopbadgeNotes\""


def read_turn(cid, path, text):
    body = "\n".join(f"{i}: {ln}" for i, ln in enumerate(text.split("\n"), 1))
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function",
             "function": {"name": "read", "arguments": json.dumps({"path": path})}}]},
            {"role": "tool", "tool_call_id": cid, "content": f"Read file {path}, lines 1-9\n{body}"}]


def sse(calls):
    delta = {"tool_calls": [{"index": i, "id": f"c{i}", "type": "function",
                             "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]}
    return b"data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}).encode() + b"\n"


def args_of(line, i=0):
    return json.loads(json.loads(line[6:])["choices"][0]["delta"]["tool_calls"][i]["function"]["arguments"])


HISTORY = read_turn("r1", "/repo/NoteStore.swift", "import Foundation\n" + OLD + "\n}") + \
          read_turn("r2", "/repo/SpaceIndex.swift", "func spaceIndex() {}\n")


def test_missing_path_filled_when_one_file_has_the_text():
    line, filled = m.fill_edit_path(sse([("edit", {"oldString": OLD, "newString": "x"})]), HISTORY, EDIT_TOOLS)
    assert filled == ["/repo/NoteStore.swift"] and args_of(line)["path"] == "/repo/NoteStore.swift", line

def test_ambiguous_text_is_left_alone():
    hist = HISTORY + read_turn("r3", "/repo/Copy.swift", OLD)
    line = sse([("edit", {"oldString": OLD, "newString": "x"})])
    assert m.fill_edit_path(line, hist, EDIT_TOOLS) == (line, [])

def test_short_old_string_is_left_alone():
    line = sse([("edit", {"oldString": "}", "newString": "x"})])
    assert m.fill_edit_path(line, HISTORY, EDIT_TOOLS) == (line, [])

def test_text_no_file_has_is_left_alone():
    line = sse([("edit", {"oldString": "something never read anywhere", "newString": "x"})])
    assert m.fill_edit_path(line, HISTORY, EDIT_TOOLS) == (line, [])

def test_call_with_path_and_other_tools_untouched():
    line = sse([("edit", {"path": "/repo/X.swift", "oldString": OLD, "newString": "x"}), ("read", {"path": "/a"})])
    assert m.fill_edit_path(line, HISTORY, EDIT_TOOLS) == (line, [])

def test_only_when_the_schema_requires_path():
    line = sse([("edit", {"oldString": OLD, "newString": "x"})])
    no_path_schema = [{"type": "function", "function": {"name": "edit", "parameters": {"required": ["oldString"]}}}]
    assert m.fill_edit_path(line, HISTORY, no_path_schema) == (line, [])

def test_non_tool_lines_pass_through():
    for line in (b"data: [DONE]\n", b"data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n", b": ping\n"):
        assert m.fill_edit_path(line, HISTORY, EDIT_TOOLS) == (line, [])

def test_failed_edits_do_not_count_as_known_text():
    hist = HISTORY[:2] + [{"role": "assistant", "content": "", "tool_calls": [{"id": "e1", "type": "function",
        "function": {"name": "edit", "arguments": json.dumps({"path": "/repo/Other.swift", "oldString": "q", "newString": OLD})}}]},
        {"role": "tool", "tool_call_id": "e1", "content": "{\"error\": {\"message\": \"Could not find oldString\"}}"}]
    line, filled = m.fill_edit_path(sse([("edit", {"oldString": OLD, "newString": "x"})]), hist, EDIT_TOOLS)
    assert filled == ["/repo/NoteStore.swift"], filled


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
