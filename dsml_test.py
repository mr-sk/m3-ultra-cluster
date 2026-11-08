import importlib.util, json, sys, glob
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location("srv", sys.argv[1] + "/vlm_cluster_server.py")
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
from mlx_vlm.utils import load_processor, load_config
from mlx_vlm.prompt_utils import apply_chat_template
path = glob.glob(str(__import__("pathlib").Path.home()) + "/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/*")[0]
config = load_config(path)
print("model_type:", config.get("model_type"), "-> dsv4 path:", srv.is_dsv4(config))
from pathlib import Path; proc = load_processor(Path(path))

# 1. no tools: new encoder vs the chat template the server used until now
convs = [
  [{"role": "user", "content": "hello"}],
  [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"},
   {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "2+2?"}],
]
for c in convs:
    old = apply_chat_template(proc, config, c)
    new = srv.dsv4_prompt(c, [])
    print("no-tools identical:", old == new)
    if old != new:
        print("  OLD:", repr(old)); print("  NEW:", repr(new))

# 2. tools + a full tool round-trip, OpenAI format in
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}}]
msgs = [{"role": "system", "content": "You are Cline."},
        {"role": "user", "content": "read a.swift"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a.swift\", \"limit\": 50}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "print(1)"}]
p = srv.dsv4_prompt(msgs, tools)
print("\n--- prompt tail ---"); print(p[-600:])

# 3. parse what the encoder itself renders for that assistant turn
rendered = p[p.rindex("<｜DSML｜tool_calls"):p.rindex("</｜DSML｜tool_calls>") + len("</｜DSML｜tool_calls>")]
content, calls = srv.parse_dsml("Reading it.\n\n" + rendered)
print("\nround trip:", repr(content), [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls])
# cut at the stop stem, two calls, malformed
two = rendered.replace("</｜DSML｜tool_calls>", "").rstrip() + "\n" + rendered[rendered.index("<｜DSML｜invoke"):]
print("stop-stem cut:", [c["function"]["name"] for c in srv.parse_dsml(rendered[:-1])[1]])
print("two calls:   ", [json.loads(c["function"]["arguments"]) for c in srv.parse_dsml(two)[1]])
print("malformed:   ", srv.parse_dsml("<｜DSML｜tool_calls>\nnonsense</｜DSML｜tool_calls>")[1])
print("plain:       ", srv.parse_dsml("Just text."))
