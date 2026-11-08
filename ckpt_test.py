import importlib.util, sys
spec = importlib.util.spec_from_file_location("srv", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
U = 7; m.BOUNDARY_IDS.add(U)
def conv(turn_lens):  # build ids with a user token at the start of each turn
    ids = []
    for n in turn_lens: ids += [U] + [1] * (n - 1)
    return ids
full = conv([3000, 5000, 200, 9000, 6000, 300])   # 23.5k tokens
print("total", len(full), "user tokens at", [i for i, t in enumerate(full) if t == U])
print("cold   :", m._checkpoint_lengths(full, 0, len(full)))
print("reused 8000 (abs pts):", [p + 8000 for p in m._checkpoint_lengths(full, 8000, len(full) - 8000)])
print("short  :", m._checkpoint_lengths(conv([500, 300]), 0, 800))
print("1 token:", m._checkpoint_lengths(conv([5]), 4, 1))
