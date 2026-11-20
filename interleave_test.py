"""Offline checks for interleave(): two fake ranks as threads, a fake exchange, a fake model.
usage: python interleave_test.py vlm_cluster_server.py   (or with no argument: interleaver.py)"""
import random, sys, threading, os
import importlib.util
if len(sys.argv) > 1:  # the server's own copy: python interleave_test.py vlm_cluster_server.py
    _spec = importlib.util.spec_from_file_location("srv", sys.argv[1])
    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
    Slot, interleave = _m.Slot, _m.interleave
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from interleaver import Slot, interleave


def nxt(sid, tok):  # the fake model: next token from (request, previous token)
    return (tok * 31 + sid * 7 + 3) % 10007


def reference(sid, first, n):
    toks = [first]
    for _ in range(n):
        toks.append(nxt(sid, toks[-1]))
    return toks


class Exchange:
    def __init__(self):
        self.b = threading.Barrier(2, timeout=5)
        self.box = {}

    def __call__(self, rank, hidden, token, control):
        self.box[rank] = (hidden, token, control)
        self.b.wait()
        h, _, _ = self.box[1]
        _, t, c = self.box[0]
        out = (t, c, h)
        self.b.wait()  # both read before the next step overwrites
        return out


def run_case(firsts, limits, event_at=None, seed=0):
    """Decode len(firsts) requests, each until it has limits[i] new tokens. Returns tokens seen by each rank."""
    ex = Exchange()
    results, errors = {}, []
    rng = random.Random(seed)
    event_steps = set(event_at or [])

    def rank_main(rank):
        try:
            slots = [Slot(i, f) for i, f in enumerate(firsts)]
            lanes = [slots[0], slots[1] if len(slots) > 1 else None]
            state = {"step": 0}

            def first_half(slot):
                return ("h", slot.sid, slot.tokens[-1])

            def second_half(slot, hidden):
                assert hidden == ("h", slot.sid, slot.tokens[-1]), f"rank 0 got {hidden} for slot {slot.sid} at {slot.tokens[-1]}"
                return nxt(slot.sid, slot.tokens[-1])

            def on_token(slot, tok):
                slot.tokens.append(tok)
                if len(slot.tokens) - 1 >= limits[slot.sid]:
                    slot.finished = True
                    return True
                return False

            def want_event():
                state["step"] += 1
                return state["step"] in event_steps

            # The engine loop: decode until an event, apply it, carry on until every slot is done.
            while not all(s.finished for s in slots):
                interleave(rank, lanes, first_half, second_half,
                           lambda h, t, c: ex(rank, h, t, c),
                           on_token if rank == 0 else None, want_event if rank == 0 else None)
                # In the server, rank 0 broadcasts which slots finished; here both ranks know the limits.
                for s in slots:
                    if len(s.tokens) - 1 >= limits[s.sid]:
                        s.finished = True
            results[rank] = [s.tokens for s in slots]
        except Exception as e:
            errors.append(f"rank {rank}: {type(e).__name__}: {e}")

    ts = [threading.Thread(target=rank_main, args=(r,)) for r in (0, 1)]
    for t in ts: t.start()
    for t in ts: t.join(timeout=20)
    if any(t.is_alive() for t in ts):
        raise AssertionError("deadlock: a rank is still running")
    if errors:
        raise AssertionError("; ".join(errors))
    return results


def check(firsts, limits, **kw):
    res = run_case(firsts, limits, **kw)
    for i, f in enumerate(firsts):
        want = reference(i, f, limits[i])
        assert res[0][i] == want, f"rank 0 slot {i}: {res[0][i][:6]}... vs {want[:6]}..."
        assert res[1][i] == want, f"rank 1 slot {i} differs from rank 0"


def test_one_request():
    check([5], [12])

def test_two_requests_same_length():
    check([5, 9], [10, 10])

def test_two_requests_one_finishes_first():
    check([5, 9], [3, 15])
    check([5, 9], [15, 3])

def test_one_token_each():
    check([1, 2], [1, 1])

def test_events_requested_midway():
    check([5, 9], [12, 12], event_at=[2, 3, 7])

def test_random_event_timings():
    rng = random.Random(42)
    for i in range(40):
        limits = [rng.randint(1, 15), rng.randint(1, 15)]
        events = rng.sample(range(1, 30), rng.randint(0, 5))
        check([rng.randint(0, 999), rng.randint(0, 999)], limits, event_at=events)


bad = 0
tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]
for name, fn in tests:
    try:
        fn(); print("ok  ", name)
    except Exception as e:
        bad += 1; print("FAIL", name, type(e).__name__, str(e)[:120])
print(f"{len(tests) - bad}/{len(tests)} passed")
sys.exit(1 if bad else 0)
