"""Bytes of weights read per decoded token, per pipeline rank (headers only).
usage: python3 active_bytes.py MODEL_DIR"""
import glob, json, os, re, struct, sys
BYTES = {"F32": 4, "F16": 2, "BF16": 2, "U32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "U16": 2, "I32": 4, "F8_E8M0": 1}
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json"))); tc = cfg.get("text_config") or cfg
k, E, L = tc["num_experts_per_tok"], tc["n_routed_experts"], tc["num_hidden_layers"]
per_layer = {}
other = 0
for f in glob.glob(os.path.join(d, "model-*.safetensors")):
    with open(f, "rb") as fh:
        hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    for name, v in hdr.items():
        if name == "__metadata__":
            continue
        b = BYTES.get(v["dtype"], 2)
        for x in v["shape"]:
            b *= x
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            other += b
            continue
        li = int(m.group(1))
        slot = per_layer.setdefault(li, {"experts": 0, "rest": 0})
        slot["experts" if "switch_mlp" in name else "rest"] += b
def active(layers):
    return sum(per_layer[i]["rest"] + per_layer[i]["experts"] * k / E for i in layers)
G = 1e9
for rank, layers in ((1, range(0, 21)), (0, range(21, L))):
    a = active(layers)
    extra = other if rank == 0 else 0   # embedding read on rank 1 is one row; lm_head on rank 0 is read fully
    print(f"rank {rank} (layers {layers.start}-{layers.stop - 1}): {a / G:.2f} GB per token in layers"
          + (f" + lm_head/norms {other / G:.2f} GB" if rank == 0 else ""))
    for bw in (800, 600):
        print(f"   at {bw} GB/s: {(a + extra) / (bw * 1e9) * 1e3:.1f} ms")
r = per_layer[30]
print(f"one layer: non-expert {r['rest'] / 1e6:.0f} MB, all experts {r['experts'] / 1e9:.2f} GB, 6 experts {r['experts'] * k / E / 1e6:.0f} MB")
