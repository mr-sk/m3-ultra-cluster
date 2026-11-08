"""Convert DeepSeek-V4-Flash-0731 (official FP8 + FP4 experts) to MLX, option B.

mlx-vlm 0.7.12's generic FP8 transform plans MXFP8 for every layer, which
doesn't fit 0731's FP4-packed experts ("expected (256, 2048, 1024), received
(..., 512)"). Skipping it for deepseek_v4 lets mlx-vlm's DeepSeek branch load
the checkpoint in its native mixed layout (experts MXFP4, FP8 layers MXFP8,
both lossless repacks). The usual -q 4 then quantizes only the layers still
in full precision (affine 4-bit, group 64).

usage: python dsv4_0731_convert.py <mlx_path> [--check]
  --check: lazy-load only and report the layout, no conversion.
"""
import sys
from collections import Counter

import mlx_vlm.fp8 as fp8
import mlx_vlm.utils as vu

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REV = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"

_orig = fp8.transform_fp8_weights


def _skip_for_dsv4(weights, config, target_quantization=None):
    if config.get("model_type") == "deepseek_v4":
        return weights, None  # let the DeepSeek-V4 branch build the mixed layout
    return _orig(weights, config, target_quantization)


fp8.transform_fp8_weights = _skip_for_dsv4

out = sys.argv[1]
if "--check" in sys.argv:
    path = vu.get_model_path(REPO, revision=REV)
    model = vu.load_model(path, lazy=True)
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    kinds = Counter()
    for name, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        q = getattr(m, "mode", None) if hasattr(m, "scales") else None
        bits = getattr(m, "bits", None)
        kinds[(type(m).__name__, q, bits)] += 1
    for k, n in kinds.most_common(12):
        print(n, k)
    print("lazy load OK")
else:
    import importlib
    conv = importlib.import_module("mlx_vlm.convert")  # mlx_vlm rebinds the name to a function
    conv.convert(hf_path=REPO, revision=REV, mlx_path=out, quantize=True, q_bits=4, q_group_size=64)
    print("conversion finished")
