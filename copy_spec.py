"""Prototype: copy-based speculative decoding (prompt lookup) on the two-rank pipeline.

For each replay prompt (recorded agent turns, spec_prompts.py), decode greedily twice from the
same cache: one token per pass (reference), and with drafts copied from the context. A draft
is the text that followed the most recent earlier occurrence of the last KEY tokens; it is
verified in one pass only when it is at least MIN_DRAFT tokens (shorter passes take mlx-vlm's
slow bit-exact path, see verify_cost.py). mlx-vlm's speculative cache transaction keeps only
the accepted tokens. Both ranks make every decision from the same tokens, so they stay in step.

Run on both ranks with the server stopped:
  cluster_launch.py ... -- python copy_spec.py --model M --prompts spec_prompts.json
"""
import argparse, copy, json, os, time

os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "1000")
import mlx.core as mx
from mlx.utils import tree_flatten
import mlx_vlm.models.pipeline as vpipe
from mlx_vlm.models import cache as vcache
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.utils import sharded_load
from proto_interleave import fixed_pipeline
from mlx_vlm.sample_utils import apply_top_p

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--max-new", type=int, default=400)
ap.add_argument("--key", type=int, default=8, help="suffix length used to find a copy")
ap.add_argument("--min-match", type=int, default=12, help="backward match needed to trust a copy")
ap.add_argument("--min-draft", type=int, default=8, help="verify only drafts of at least this many tokens")
ap.add_argument("--start-draft", type=int, default=16, help="first draft length; doubles while fully accepted")
ap.add_argument("--max-draft", type=int, default=64)
ap.add_argument("--cooldown", type=int, default=6, help="single steps after a draft that got <3 tokens accepted")
ap.add_argument("--show-diff", action="store_true", help="print both texts where a run diverges")
ap.add_argument("--temp", type=float, default=0.0, help="sample at this temperature (0 = greedy), as the server does")
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--configs", default="", help="'key,min_match,start,max,cooldown/...' to sweep (overrides the single settings)")
ap.add_argument("--only", type=int, default=-1, help="run one prompt index")
ap.add_argument("--debug", action="store_true", help="compare a verify pass with one-token decoding, then stop")
ap.add_argument("--exact-indexer", action="store_true", help="exact top-k in every indexer (decode and multi-token), not HISA")
args = ap.parse_args()


def dist(logits):
    """Sampling distribution for rows of logits: log-softmax, top_p, temperature (as the server)."""
    lp = logits.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    if 0 < args.top_p < 1:
        lp = apply_top_p(lp, args.top_p)
    return mx.softmax(lp / max(args.temp, 0.01), axis=-1)


def key(index, salt=0):
    return mx.random.key((args.seed * 1_000_003 + index + salt) % (2 ** 31))


def pick(row, index):
    """The token at position `index` from one row of logits: greedy, or sampled with a per-token seed."""
    if args.temp <= 0:
        return int(mx.argmax(row).item())
    return int(mx.random.categorical(mx.log(dist(row) + 1e-30), key=key(index)).item())


def verify(logits, draft, index):
    """Accepted draft length and the next token, as the server's Engine._verify decides."""
    if args.temp <= 0:
        preds = mx.argmax(logits, axis=-1).tolist()
        a = 0
        while a < len(draft) and preds[a] == draft[a]:
            a += 1
        return a, preds[a]
    probs = dist(logits)
    p_d = mx.take_along_axis(probs[:-1], mx.array(draft)[:, None], axis=-1)[:, 0]
    ok = (mx.random.uniform(shape=(len(draft),), key=key(index)) < p_d).tolist()
    a = 0
    while a < len(draft) and ok[a]:
        a += 1
    row = probs[a]
    if a < len(draft):
        row = mx.where(mx.arange(row.shape[0]) == draft[a], 0.0, row)
    return a, int(mx.random.categorical(mx.log(row + 1e-30), key=key(index, 7919)).item())


class Lookup:
    """Earlier occurrences of the last `key` tokens; picks the one whose match with the
    current text reaches furthest back (ties: most recent) and returns what followed it,
    if that backward match is at least `min_match` tokens."""

    def __init__(self, key, min_match, by_match=0):
        self.key, self.min_match, self.pos, self.done = key, min_match, {}, key - 1
        self.by_match = by_match  # >0: draft at least the backward match length, up to this

    def draft(self, ctx, limit):
        k = self.key
        while self.done < len(ctx) - 2:  # index every position except the current end
            self.done += 1
            self.pos.setdefault(tuple(ctx[self.done - k + 1:self.done + 1]), []).append(self.done)
        best, best_len = None, 0
        end = len(ctx) - 1
        for p in reversed(self.pos.get(tuple(ctx[-k:]), [])[-16:]):
            L = k
            while L < 256 and p - L >= 0 and ctx[p - L] == ctx[end - L]:
                L += 1
            if L > best_len:
                best, best_len = p, L
        if best is None or best_len < self.min_match:
            return []
        if self.by_match:
            limit = max(limit, min(self.by_match, best_len))
        return ctx[best + 1:best + 1 + limit]


def main():
    world = mx.distributed.init(); rank = world.rank()
    say = lambda *a: print(f"[rank {rank}]", *a, flush=True)
    vpipe.PipelineMixin.pipeline = fixed_pipeline
    model, processor = sharded_load(args.model, pipeline_group=world)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    for _, p in tree_flatten(model.parameters()):
        mx.eval(p.sum())
    barrier = lambda: mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    barrier()
    lm = model.language_model
    if args.exact_indexer:
        import mlx_vlm.models.deepseek_v4.language as dl
        idx = [m for _, m in lm.named_modules() if isinstance(m, dl.Indexer)]
        for m in idx:
            m.index_block = 0
        say(f"exact indexer selection in {len(idx)} indexers")
    data = json.load(open(args.prompts))
    eos = data["eos"]
    CONFIGS = ([tuple(int(v) for v in c.split(",")) for c in args.configs.replace(";", "/").split("/") if c.strip()]
               or [(args.key, args.min_match, args.start_draft, args.max_draft, args.cooldown)])
    totals = {cfg: {"plain_s": 0.0, "spec_s": 0.0, "tokens": 0, "same": 0} for cfg in CONFIGS}

    for i, pr in enumerate(data["prompts"]):
        if args.only >= 0 and i != args.only:
            continue
        ids = pr["ids"]
        base = vcache.make_prompt_cache(lm)
        for s in range(0, len(ids) - 1, 512):
            lm(mx.array(ids[s:min(s + 512, len(ids) - 1)])[None], cache=base, skip_logits=True)
            mx.eval([c.state for c in base])
        first = int(mx.argmax(lm(mx.array(ids[-1:])[None], cache=base).logits[0, -1]).item())
        mx.eval([c.state for c in base])

        if args.debug:
            draft = ids[2000:2016]  # 16 tokens of real text as the draft
            inp = [first] + draft
            # one token at a time, teacher-forced through the same tokens
            c = copy.deepcopy(base); mx.eval([x.state for x in c])
            single = []
            for t_ in inp:
                single.append(lm(mx.array([[t_]]), cache=c).logits[0, -1].astype(mx.float32))
            mx.eval(single)
            after_single = lm(mx.array([[123]]), cache=c).logits[0, -1].astype(mx.float32)
            # one pass, no transaction
            c = copy.deepcopy(base); mx.eval([x.state for x in c])
            block = lm(mx.array(inp)[None], cache=c).logits[0].astype(mx.float32)
            for j in range(len(inp)):
                d = mx.abs(block[j] - single[j]).max().item()
                say(f"pos {j}: max |logit diff| {d:.4f}; argmax single {int(mx.argmax(single[j]).item())} "
                    f"block {int(mx.argmax(block[j]).item())}")
            # transaction, commit 6 of 17, then one more token: compare with one-token decoding of the same 6
            for keep in (6, 17):
                c = copy.deepcopy(base); mx.eval([x.state for x in c])
                tx = start_speculative_cache(c, len(inp))
                mx.eval(lm(mx.array(inp)[None], cache=c).logits)
                tx.commit(keep)
                nxt = lm(mx.array([[123]]), cache=c).logits[0, -1].astype(mx.float32)
                c2 = copy.deepcopy(base); mx.eval([x.state for x in c2])
                for t_ in inp[:keep]:
                    lm(mx.array([[t_]]), cache=c2)
                ref_ = lm(mx.array([[123]]), cache=c2).logits[0, -1].astype(mx.float32)
                say(f"commit {keep}: next-token max |logit diff| vs one-token path {mx.abs(nxt - ref_).max().item():.4f}, "
                    f"argmax {int(mx.argmax(nxt).item())} vs {int(mx.argmax(ref_).item())}")
            return

        # Reference: one token per pass.
        c = copy.deepcopy(base); mx.eval([x.state for x in c])
        toks = [first]
        barrier(); t = time.perf_counter()
        while len(toks) < args.max_new and toks[-1] != eos:
            y = lm(mx.array([[toks[-1]]]), cache=c).logits[0, -1]
            toks.append(pick(y, len(toks)))
        barrier(); plain_s = time.perf_counter() - t
        ref = toks
        del c

        for cfg in CONFIGS:
          key, min_match, start_draft, max_draft, cooldown = cfg[:5]
          mode = cfg[5] if len(cfg) > 5 else 0  # 1: first draft as long as the match; 2: keep length on partial
          # Copy drafts, verified in one pass.
          c = copy.deepcopy(base); mx.eval([x.state for x in c])
          toks = [first]; ctx = list(ids) + toks
          look = Lookup(key, min_match, max_draft if mode & 1 else 0)
          passes = singles = drafted = accepted = 0
          dlen, cool = start_draft, 0
          barrier(); t = time.perf_counter()
          while len(toks) < args.max_new and toks[-1] != eos:
              room = args.max_new - len(toks)
              draft = []
              if cool > 0:
                  cool -= 1
              elif room > args.min_draft:
                  draft = look.draft(ctx, min(dlen, room - 1))
              if len(draft) >= args.min_draft:
                  inp = [toks[-1]] + draft
                  tx = start_speculative_cache(c, len(inp))
                  a, bonus = verify(lm(mx.array(inp)[None], cache=c).logits[0], draft, len(toks))
                  tx.commit(a + 1)  # keep inp[0..a] in the cache
                  new = draft[:a] + [bonus]
                  passes += 1; drafted += len(draft); accepted += a
                  if a == len(draft):
                      dlen = min(dlen * 2, max_draft)
                  else:
                      dlen = max(start_draft, a) if mode & 2 else start_draft
                      if a < 3:
                          cool = cooldown
              else:
                  y = lm(mx.array([[toks[-1]]]), cache=c).logits[0, -1]
                  new = [pick(y, len(toks))]
                  singles += 1
              if eos in new:
                  new = new[:new.index(eos) + 1]
              toks += new; ctx += new
          barrier(); spec_s = time.perf_counter() - t
          del c
          n_spec = len(toks)
          toks = toks[:len(ref)]
          same = next((k for k, (x, y) in enumerate(zip(ref, toks)) if x != y), min(len(ref), len(toks)))
          n = len(ref)
          if same < n and args.show_diff:
              tk = getattr(processor, "tokenizer", processor)
              say(f"  diverges at {same}: reference ...{tk.decode(ref[max(0, same - 12):same + 25])!r}")
              say(f"  diverges at {same}: copy-spec ...{tk.decode(toks[max(0, same - 12):same + 25])!r}")
          T = totals[cfg]
          T["plain_s"] += plain_s; T["spec_s"] += spec_s; T["tokens"] += n
          T["spec_tokens"] = T.get("spec_tokens", 0) + n_spec
          T["same"] += int(same == n and len(toks) == n)
          say(f"cfg {cfg} prompt {i} ({pr['kind']}, {len(ids)} tokens): {n} tokens; plain {n / plain_s:.1f} tok/s, "
              f"copy-spec {n_spec / spec_s:.1f} tok/s ({(n_spec / spec_s) / (n / plain_s):.2f}x); {passes} verify passes "
              f"({accepted}/{drafted} draft tokens accepted), {singles} single steps; "
              f"identical to reference for {same}/{n} tokens")
    for cfg, T in totals.items():
        say(f"TOTAL cfg (key,min_match,start,max,cooldown[,mode])={cfg}: {T['tokens']} tokens; plain {T['tokens'] / T['plain_s']:.1f} tok/s, "
            f"copy-spec {T['spec_tokens'] / T['spec_s']:.1f} tok/s "
            f"({(T['spec_tokens'] / T['spec_s']) / (T['tokens'] / T['plain_s']):.2f}x); "
            f"{T['same']} of the runs fully identical")


main()
