"""mlx.launch without the busy IO loop.

usage: same arguments as mlx.launch, e.g.
  ~/cluster-env/bin/python cluster_launch.py --backend jaccl --hostfile ~/m3-ultra-jaccl.json -- ...

mlx.launch's IO loop (mlx 0.32.4, mlx/_distributed_utils/launch.py,
_launch_with_io) never sleeps while the job runs:
  - each rank's thread passes the child's stdin to select() as a write target
    even when there is nothing to write. A pipe is almost always writable, so
    select() returns at once, every time;
  - the main loop selects on our own stdin, and at EOF (/dev/null under nohup)
    that is always readable, so it does not wait either.
Measured 2026-11-13 on the cluster's launcher: ~1.7 cores for the server's whole
life. This file runs mlx.launch with a copy of that loop with both waits fixed
and nothing else changed. It only patches the mlx version it was copied from;
on any other version it warns and runs the stock launcher.
"""
import os
import sys
import threading
import time
from itertools import chain
from queue import Empty as QueueEmpty
from queue import Queue
from select import select

import mlx.core as mx
import mlx._distributed_utils.launch as L
from mlx._distributed_utils.common import log, log_warning

COPIED_FROM = "0.32.4"


def _launch_with_io(command_class, arguments, verbose):
    stop = False
    exit_codes = [(None, None)] * len(arguments)

    def _thread_fn(rank, *args, **kwargs):
        stdin_queue = kwargs.pop("stdin_queue")
        stdout_queue = kwargs.pop("stdout_queue")
        stderr_queue = kwargs.pop("stderr_queue")

        command = command_class(rank, *args, **kwargs)
        p = command.process
        os.set_blocking(p.stdout.fileno(), False)
        os.set_blocking(p.stderr.fileno(), False)
        os.set_blocking(p.stdin.fileno(), False)

        to_read = [p.stdout.fileno(), p.stderr.fileno()]
        stdin_open = True

        stdin_buffer = b""
        while True:
            exited = p.poll() is not None
            try:
                stdin_buffer += stdin_queue.get_nowait()
            except QueueEmpty:
                pass
            # Fix 1: only ask for the child's stdin when there is something to
            # write. Stock passed it every time, so select() never waited.
            to_write = [p.stdin.fileno()] if stdin_open and stdin_buffer else []
            # The wait is shorter while input is queued but not yet writable.
            timeout = 0 if exited else (0.05 if stdin_buffer else 1.0)
            rlist, wlist, _ = select(to_read, to_write, [], timeout)
            for fd in rlist:
                is_stdout = fd == p.stdout.fileno()
                try:
                    data = os.read(fd, 8192)
                except BlockingIOError:
                    continue
                if not data:
                    to_read.remove(fd)
                    continue
                msg = command.preprocess_output(data.decode(errors="ignore"), is_stdout)
                if is_stdout:
                    stdout_queue.put(msg.encode())
                else:
                    stderr_queue.put(msg.encode())
            for fd in wlist:
                if len(stdin_buffer) > 0:
                    try:
                        n = os.write(fd, stdin_buffer)
                    except BrokenPipeError:
                        stdin_open = False
                        continue
                    stdin_buffer = stdin_buffer[n:]
            if exited and not rlist:
                break
            if stop and not exited:
                command.terminate()

        exit_codes[rank] = command.exit_status

        if exit_codes[rank][1]:
            log_warning(f"Node with rank {rank} was killed")
        elif exit_codes[rank][0] != 0:
            log_warning(f"Node with rank {rank} exited with code {exit_codes[rank][0]}")
        else:
            log(verbose, f"Node with rank {rank} completed")

    stdin_queues = []
    stdout_queues = []
    stderr_queues = []
    threads = []
    for i, (args, kwargs) in enumerate(arguments):
        stdin_queues.append(Queue())
        stdout_queues.append(Queue())
        stderr_queues.append(Queue())
        t = threading.Thread(
            target=_thread_fn,
            args=args,
            kwargs=kwargs
            | {
                "stdin_queue": stdin_queues[-1],
                "stdout_queue": stdout_queues[-1],
                "stderr_queue": stderr_queues[-1],
            },
        )
        t.start()
        threads.append(t)

    os.set_blocking(sys.stdin.fileno(), False)
    os.set_blocking(sys.stdout.fileno(), True)
    os.set_blocking(sys.stderr.fileno(), True)
    stdin_open = True
    while not stop or any(not q.empty() for q in chain(stdout_queues, stderr_queues)):
        # Broadcast user input to the jobs
        if stdin_open:
            rlist, _, _ = select([sys.stdin.fileno()], [], [], 0.1)
        else:
            # Fix 2: at EOF stdin is always "readable", so stock never waited.
            rlist = []
            time.sleep(0.1)
        for fd in rlist:
            try:
                stdin_buffer = os.read(fd, 8192)
            except BlockingIOError:
                continue
            if not stdin_buffer:
                stdin_open = False
                continue
            for q in stdin_queues:
                q.put(stdin_buffer)

        # Gather job output
        for q in stdout_queues:
            try:
                while not q.empty():
                    sys.stdout.buffer.write(q.get_nowait())
            except QueueEmpty:
                pass
        for q in stderr_queues:
            try:
                while not q.empty():
                    sys.stderr.buffer.write(q.get_nowait())
            except QueueEmpty:
                pass
        sys.stdout.buffer.flush()
        sys.stderr.buffer.flush()

        # Check if all are running and terminate otherwise
        if any(t.is_alive() for t in threads):
            for i, t in enumerate(threads):
                if not t.is_alive():
                    if exit_codes[i][0] != 0:
                        stop = True
                        break
        else:
            break

    # Wait for the jobs to finish
    for t in threads:
        t.join()


def install(version=None):
    """Swap the fixed loop into mlx's launcher if mlx is the version it was copied from."""
    version = mx.__version__ if version is None else version
    if version != COPIED_FROM:
        log_warning(f"cluster_launch: written for mlx {COPIED_FROM}, found {version}; "
                    "running the stock mlx.launch (its IO loop may spin a core or two)")
        return False
    L._launch_with_io = _launch_with_io
    return True


if __name__ == "__main__":
    install()
    sys.exit(L.main())
