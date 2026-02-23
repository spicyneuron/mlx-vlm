import re

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.convert import (
    apply_float_overrides,
    build_override_predicate,
    parse_overrides,
)


# -- parse_overrides ----------------------------------------------------------


def test_parse_overrides_int():
    result = parse_overrides(["lm_head=8"])
    assert len(result) == 1
    assert isinstance(result[0][0], re.Pattern)
    assert result[0][1] == 8


def test_parse_overrides_float_dtypes():
    for dtype in ("float16", "bfloat16", "float32"):
        result = parse_overrides([f"embed_tokens={dtype}"])
        assert result[0][1] == dtype


def test_parse_overrides_multiple():
    result = parse_overrides(["lm_head=8", "embed_tokens=float16"])
    assert len(result) == 2
    assert result[0][1] == 8
    assert result[1][1] == "float16"


def test_parse_overrides_regex():
    result = parse_overrides([r"layers\.0\..*=6"])
    assert result[0][0].search("model.layers.0.mlp.down_proj")
    assert result[0][0].search("model.layers.1.mlp.down_proj") is None


def test_parse_overrides_missing_equals():
    with pytest.raises(ValueError, match="Expected PATTERN=VALUE"):
        parse_overrides(["lm_head"])


def test_parse_overrides_invalid_regex():
    with pytest.raises(ValueError, match="Invalid regex"):
        parse_overrides(["[invalid=8"])


def test_parse_overrides_quant_mode():
    for mode in ("mxfp4", "nvfp4", "mxfp8"):
        result = parse_overrides([f"down_proj={mode}"])
        assert result[0][1] == mode


def test_parse_overrides_invalid_value():
    with pytest.raises(ValueError, match="Invalid override value"):
        parse_overrides(["lm_head=garbage"])


# -- build_override_predicate -------------------------------------------------


def test_override_int_match():
    overrides = [(re.compile("lm_head"), 8)]
    pred = build_override_predicate(overrides, lambda p, m: True, 64)
    result = pred("model.lm_head", None)
    assert result == {"group_size": 64, "bits": 8, "mode": "affine"}


def test_override_float_returns_false():
    overrides = [(re.compile("embed_tokens"), "float16")]
    pred = build_override_predicate(overrides, lambda p, m: True, 64)
    assert pred("model.embed_tokens", None) is False


def test_override_no_match_delegates():
    overrides = [(re.compile("lm_head"), 8)]
    base = lambda p, m: {"group_size": 64, "bits": 4, "mode": "affine"}
    pred = build_override_predicate(overrides, base, 64)
    result = pred("model.layers.0.mlp.down_proj", None)
    assert result == {"group_size": 64, "bits": 4, "mode": "affine"}


def test_override_preserves_multimodal_skip():
    """Override predicate delegates to base, which should still skip vision modules."""
    overrides = [(re.compile("lm_head"), 8)]

    def base_with_skip(path, module):
        if "vision" in path:
            return False
        return True

    pred = build_override_predicate(overrides, base_with_skip, 64)
    assert pred("vision_model.encoder", None) is False
    assert pred("language_model.layers.0", None) is True


def test_override_quant_mode():
    overrides = [(re.compile("down_proj"), "mxfp4")]
    pred = build_override_predicate(overrides, None, 64)
    result = pred("model.layers.0.mlp.down_proj", None)
    assert result == {"group_size": 32, "bits": 4, "mode": "mxfp4"}


def test_override_first_match_wins():
    overrides = [
        (re.compile("lm_head"), 8),
        (re.compile("lm_head"), 6),
    ]
    pred = build_override_predicate(overrides, lambda p, m: True, 64)
    assert pred("model.lm_head", None)["bits"] == 8


def test_override_group_size_passthrough():
    overrides = [(re.compile("lm_head"), 4)]
    pred = build_override_predicate(overrides, lambda p, m: True, 32)
    assert pred("model.lm_head", None)["group_size"] == 32


# -- apply_float_overrides ----------------------------------------------------


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)


def test_apply_float_overrides_casts():
    model = _Wrapper()
    overrides = [(re.compile("linear"), "float16")]
    apply_float_overrides(model, overrides)
    assert model.linear.weight.dtype == mx.float16


def test_apply_float_overrides_no_match():
    model = _Wrapper()
    original_dtype = model.linear.weight.dtype
    overrides = [(re.compile("lm_head"), "float16")]
    apply_float_overrides(model, overrides)
    assert model.linear.weight.dtype == original_dtype


def test_apply_float_overrides_skips_int():
    model = _Wrapper()
    original_dtype = model.linear.weight.dtype
    overrides = [(re.compile("linear"), 8)]
    apply_float_overrides(model, overrides)
    assert model.linear.weight.dtype == original_dtype
