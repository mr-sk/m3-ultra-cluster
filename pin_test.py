"""Checks that model weights stay wired (pinned in RAM) between requests.
usage: python pin_test.py vlm_cluster_server.py   (needs the server's venv, on a Mac)

macOS drops GPU residency after ~2 s idle (a macOS bug, see ml-explore/mlx#4633).
MLX's MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS re-requests it, but only for memory
inside the wired limit, and mlx-vlm's stream_generate sets that limit back to 0
after every call. The server sets the env var and holds the limit.
"""
import importlib, importlib.util, os, subprocess, sys

import mlx.core as mx
import mlx.nn as nn

spec = importlib.util.spec_from_file_location("srv", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# importlib: the attribute mlx_vlm.generate is the generate() function, not the package.
dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
ar = importlib.import_module("mlx_vlm.generate.ar")

MODEL = nn.Linear(4, 4)
MAX = mx.device_info()["max_recommended_working_set_size"]


def current_limit():
    # MLX has no getter: set_wired_limit returns the previous value.
    old = mx.set_wired_limit(0)
    mx.set_wired_limit(old)
    return old


def test_stock_context_unpins_after_generate():
    # Documents the upstream behaviour the fix works around.
    mx.set_wired_limit(0)
    with m.STOCK_WIRED_LIMIT(MODEL):
        assert current_limit() == MAX
    assert current_limit() == 0


def test_pin_wires_working_set():
    mx.set_wired_limit(0)
    assert m.pin_weights() == MAX
    assert current_limit() == MAX


def test_generate_keeps_weights_pinned():
    mx.set_wired_limit(0)
    m.pin_weights()
    for mod in (dispatch, ar):
        with mod.wired_limit(MODEL, [mx.default_stream(mx.gpu)]):
            assert current_limit() == MAX
        assert current_limit() == MAX, mod.__name__


def _env_after_import(preset=None):
    env = {k: v for k, v in os.environ.items() if k != REFRESH}
    if preset is not None:
        env[REFRESH] = preset
    code = ("import importlib.util, os, sys; "
            "s = importlib.util.spec_from_file_location('srv', sys.argv[1]); "
            "s.loader.exec_module(importlib.util.module_from_spec(s)); "
            f"print(os.environ.get({REFRESH!r}))")
    out = subprocess.run([sys.executable, "-c", code, sys.argv[1]], env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


REFRESH = "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS"


def test_server_turns_on_residency_refresh():
    assert _env_after_import() == "1000"


def test_residency_refresh_can_be_overridden():
    assert _env_after_import("0") == "0"


def test_pin_is_idempotent():
    mx.set_wired_limit(0)
    m.pin_weights()
    m.pin_weights()
    with dispatch.wired_limit(MODEL):
        pass
    assert current_limit() == MAX


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as e:
            failed += 1
            print("FAIL", name, type(e).__name__, e)
    mx.set_wired_limit(0)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
