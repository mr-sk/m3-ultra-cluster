"""Checks that cluster_launch.py forwards output without spinning the CPU.
usage: python launch_test.py cluster_launch.py   (needs the mlx venv that has mlx.launch)

mlx.launch's IO loop (mlx 0.32.4, mlx/_distributed_utils/launch.py) never
sleeps: each rank's thread asks select() whether the child's stdin is
writable, which it nearly always is, and the main loop selects on our own
stdin, which at EOF (/dev/null under nohup) is always readable. Measured
2026-11-13: ~1.7 cores for the server's whole life. cluster_launch.py runs
mlx.launch with a fixed copy of that loop.
"""
import os, subprocess, sys, textwrap

LAUNCH = os.path.abspath(sys.argv[1])
failures = 0


def check(name, ok, detail=""):
    global failures
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    failures += not ok


# Runs in a child process so that its stdin can be /dev/null or a pipe, as
# under nohup or an interactive shell.
CHILD = textwrap.dedent(r"""
    import importlib.util, sys, time
    from subprocess import PIPE, Popen
    import mlx._distributed_utils.launch as L

    spec = importlib.util.spec_from_file_location("cluster_launch", sys.argv[1])
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    which, script = sys.argv[2], sys.argv[3]
    fn = C._launch_with_io if which == "fixed" else L._launch_with_io

    class Local(L.CommandProcess):
        def __init__(self, rank):
            self._p = Popen(script.replace("RANK", str(rank)), shell=True,
                            executable="/bin/bash", stdin=PIPE, stdout=PIPE, stderr=PIPE)
        @property
        def process(self): return self._p
        @property
        def exit_status(self): return self._p.poll(), False
        def preprocess_output(self, data, is_stdout=False): return data
        def terminate(self): self._p.terminate(); self._p.wait()

    cpu0 = time.process_time()
    fn(Local, [((0,), {}), ((1,), {})], False)
    print(f"CPU {time.process_time() - cpu0:.3f}", file=sys.stderr)
""")


def run(which, script, stdin=subprocess.DEVNULL, data=None):
    p = subprocess.run([sys.executable, "-c", CHILD, LAUNCH, which, script],
                       stdin=stdin, input=data, capture_output=True, text=True, timeout=60)
    cpu = [float(l.split()[1]) for l in p.stderr.splitlines() if l.startswith("CPU ")]
    return p.stdout, (cpu[0] if cpu else None), p.stderr


IDLE = "echo hello-from-RANK; sleep 2; echo bye-from-RANK"

out, cpu, err = run("stock", IDLE)
print(f"info: stock loop used {cpu} s of CPU over ~2 s idle")

out, cpu, err = run("fixed", IDLE)
check("fixed loop uses under 0.3 s of CPU over ~2 s idle (stdin /dev/null)",
      cpu is not None and cpu < 0.3, f"{cpu} s; {err.strip()[-300:]}")
check("output from both ranks is still forwarded",
      all(f"{w}-from-{r}" in out for w in ("hello", "bye") for r in (0, 1)), out.strip())

out, cpu, err = run("fixed", "read x; echo got-RANK:$x; sleep 1",
                    stdin=None, data="ping\n")
check("stdin is still forwarded to every rank",
      "got-0:ping" in out and "got-1:ping" in out, out.strip() + err.strip()[-300:])

# install() swaps the loop into mlx's launcher, but only on the version it was copied from.
import importlib.util
import mlx.core as mx
import mlx._distributed_utils.launch as L
spec = importlib.util.spec_from_file_location("cluster_launch", LAUNCH)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
stock = L._launch_with_io
check("install() patches mlx's launcher on mlx " + mx.__version__,
      C.install() is True and L._launch_with_io is C._launch_with_io)
L._launch_with_io = stock
check("install() leaves other mlx versions alone",
      C.install(version="9.9.9") is False and L._launch_with_io is stock)
L._launch_with_io = stock

print("all passed" if not failures else f"{failures} failed")
sys.exit(1 if failures else 0)
