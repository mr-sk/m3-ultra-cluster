"""Make replay prompts for copy_spec.py: N random recorded agent turns, encoded as the server does.
usage (on purple, from the repo): ~/vlm-env/bin/python spec_prompts.py MODEL_DIR OUT.json N REQUEST.json..."""
import importlib.util, json, random, sys
spec = importlib.util.spec_from_file_location("srv", "vlm_cluster_server.py")
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
out_path, n = sys.argv[2], int(sys.argv[3])
reqs = [json.load(open(f)) for f in sorted(sys.argv[4:])]
turns = []
for cur, nxt in zip(reqs, reqs[1:]):
    m, m2 = cur["messages"], nxt["messages"]
    if len(m2) > len(m) and m2[:len(m)] == m and m2[len(m)].get("role") == "assistant":
        kind = "+".join(sorted({c["function"]["name"] for c in m2[len(m)].get("tool_calls") or []})) or "text"
        turns.append((kind, cur))
random.Random(7).shuffle(turns)
chosen = turns[:n]
data = [{"kind": k, "ids": tok.encode(srv.dsv4_prompt(r["messages"], r.get("tools")), add_special_tokens=False)}
        for k, r in chosen]
json.dump({"eos": tok.eos_token_id, "prompts": data}, open(out_path, "w"))
print([(d["kind"], len(d["ids"])) for d in data], "eos", tok.eos_token_id)
