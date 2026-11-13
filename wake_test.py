"""Checks the wake socket that lets rank 1 wait for jobs without spinning.
usage: python wake_test.py vlm_cluster_server.py   (needs the server's venv)

Rank 1 waits for the next job inside a CPU all_sum, and JACCL waits for that by
polling the Thunderbolt RDMA completion queue: one core at 100% for as long as
the server is idle (seen 2026-11-13). Rank 0 now sends one byte over a TCP
socket before each broadcast, and rank 1 blocks on it, at no CPU, before
entering the all_sum.
"""
import importlib.util, json, socket, sys, threading, time

import mlx.core as mx
import numpy as np

spec = importlib.util.spec_from_file_location("srv", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

failures = 0


def check(name, ok, detail=""):
    global failures
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    failures += not ok


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def pair():
    """A connected (rank 0, rank 1) pair of Wake ends on localhost."""
    port = free_port()
    out = {}
    t = threading.Thread(target=lambda: out.setdefault(
        "server", m.Wake.listen(port, 1, host="127.0.0.1", timeout=10)))
    t.start()
    worker = m.Wake.connect("127.0.0.1", port, rank=1, timeout=10)
    t.join()
    return out["server"], worker


# 1. Address: rank 0's host from MLX_JACCL_COORDINATOR, our own port.
check("wake_address uses the coordinator host",
      m.wake_address({"MLX_JACCL_COORDINATOR": "10.1.2.3:32323"}, 8095) == ("10.1.2.3", 8095))
check("wake_address is None without a coordinator", m.wake_address({}, 8095) is None)

# 2. wait() blocks until rank 0 signals, then returns True.
server, worker = pair()
done = []
t = threading.Thread(target=lambda: done.append(worker.wait()))
t.start()
time.sleep(0.3)
check("wait blocks before a signal", not done)
server.signal()
t.join(2)
check("wait returns True after a signal", done == [True], str(done))

# 3. Waiting costs (almost) no CPU. A spinning wait would burn ~1 s here.
cpu0 = time.process_time()
t = threading.Thread(target=lambda: done.append(worker.wait()))
t.start()
time.sleep(1.0)
cpu = time.process_time() - cpu0
server.signal()
t.join(2)
check("waiting 1 s uses under 0.1 s of CPU", cpu < 0.1, f"{cpu:.3f} s")

# 4. A signal sent before the worker waits is not lost.
server.signal()
server.signal()
check("early signals are kept", worker.wait() and worker.wait())

# 5. If rank 0 goes away, wait() returns False instead of blocking forever.
server.close()
check("wait returns False when rank 0 closes", worker.wait() is False)
worker.close()

# 6. A stranger connecting first does not take the worker's place.
port = free_port()
out = {}
t = threading.Thread(target=lambda: out.setdefault(
    "server", m.Wake.listen(port, 1, host="127.0.0.1", timeout=10)))
t.start()
time.sleep(0.2)
stranger = socket.create_connection(("127.0.0.1", port))
stranger.sendall(b"GET / HTTP/1.0\r\n\r\n")
worker = m.Wake.connect("127.0.0.1", port, rank=1, timeout=10)
t.join(5)
out["server"].signal()
check("listen ignores connections without the hello", worker.wait() is True)
stranger.close(); out["server"].close(); worker.close()

# 7. broadcast() uses the wake socket around its all_sum.
class FakeWorld:
    def __init__(self, rank): self._rank = rank
    def size(self): return 2
    def rank(self): return self._rank

events = []
payload = json.dumps({"prompt": "hi"}).encode()

def fake_all_sum(x, group=None, stream=None):
    events.append("all_sum")
    if x.size == 1:
        return mx.array([len(payload)], dtype=mx.int32)
    return mx.array(np.frombuffer(payload, dtype=np.uint8).astype(np.int32))

class FakeWake:
    def __init__(self, alive=True): self.alive = alive
    def signal(self): events.append("signal")
    def wait(self):
        events.append("wait")
        return self.alive

real_all_sum = mx.distributed.all_sum
mx.distributed.all_sum = fake_all_sum
try:
    m.WAKE = FakeWake()
    events.clear()
    got = m.broadcast(FakeWorld(1))
    check("rank 1 waits on the socket before the all_sum",
          events[:2] == ["wait", "all_sum"] and got == {"prompt": "hi"}, str(events))

    events.clear()
    m.broadcast(FakeWorld(0), {"prompt": "hi"})
    check("rank 0 signals before the all_sum", events[:2] == ["signal", "all_sum"], str(events))

    m.WAKE = FakeWake(alive=False)
    events.clear()
    got = m.broadcast(FakeWorld(1))
    check("rank 1 shuts down when rank 0 is gone",
          got == {"cmd": "shutdown"} and "all_sum" not in events, f"{got} {events}")
finally:
    mx.distributed.all_sum = real_all_sum
    m.WAKE = None

print("all passed" if not failures else f"{failures} failed")
sys.exit(1 if failures else 0)
