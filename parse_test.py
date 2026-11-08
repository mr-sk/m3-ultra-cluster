import importlib.util, sys
spec = importlib.util.spec_from_file_location("srv", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cases = {
 "plural close": 'Let me look.\n\n<tool_call> {"name": "search_codebase", "arguments": {"queries": ["BadgeRenderer", "render"]}} </tool_calls>',
 "proper": '<tool_call>\n{"name": "read_files", "arguments": {"paths": ["a.swift"]}}\n</tool_call>',
 "cut at stop stem": '<tool_call>{"name": "x", "arguments": {}}\n</tool_call',
 "fenced": '<tool_call>\n```json\n{"name": "x", "arguments": {"a": 1}}\n```\n</tool_call>',
 "no close": '<tool_call>{"name": "x", "arguments": {"a": "}"}}',
 "malformed": '<tool_call>{name: x}</tool_call>',
 "plain": 'Just an answer.',
}
for k, t in cases.items():
    c, calls = m.parse_tool_call(t)
    print(f"{k:18} content={c!r:24.24} calls={[(x['function']['name'], x['function']['arguments']) for x in calls]}")
