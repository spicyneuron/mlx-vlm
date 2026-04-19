import mlx.core as mx
import pytest

from mlx_vlm.utils import (
    ensure_matching_weights,
    format_weight_compatibility_error,
    inspect_weight_compatibility,
)


def test_inspect_weight_compatibility_finds_missing_unexpected_and_shape_mismatches():
    expected_weights = {
        "vision_tower.weight": mx.zeros((4, 4)),
        "language_model.lm_head.weight": mx.zeros((4, 4)),
        "language_model.model.layers.0.weight": mx.zeros((4, 4)),
    }
    weights = {
        "vision_tower.weight": mx.zeros((8, 4)),
        "lm_head.weight": mx.zeros((4, 4)),
        "extra.weight": mx.zeros((2, 2)),
    }

    issues = inspect_weight_compatibility(expected_weights, weights)

    assert issues["missing"] == [
        "language_model.lm_head.weight",
        "language_model.model.layers.0.weight",
    ]
    assert issues["unexpected"] == ["extra.weight", "lm_head.weight"]
    assert issues["shape_mismatches"] == [
        ("vision_tower.weight", (4, 4), (8, 4))
    ]
    assert issues["suggestions"] == [
        ("lm_head.weight", "language_model.lm_head.weight")
    ]


def test_format_weight_compatibility_error_includes_actionable_sections():
    issues = {
        "expected_count": 3,
        "provided_count": 2,
        "missing": ["language_model.lm_head.weight"],
        "unexpected": ["lm_head.weight"],
        "shape_mismatches": [("vision_tower.weight", (4, 4), (8, 4))],
        "suggestions": [("lm_head.weight", "language_model.lm_head.weight")],
    }

    message = format_weight_compatibility_error(issues)

    assert "Checkpoint weights do not match the instantiated model" in message
    assert "Missing keys:" in message
    assert "Unexpected keys:" in message
    assert "Shape mismatches:" in message
    assert "Possible remaps:" in message
    assert "lm_head.weight -> language_model.lm_head.weight" in message


def test_ensure_matching_weights_raises_helpful_error():
    expected_weights = {"language_model.lm_head.weight": mx.zeros((4, 4))}
    weights = {"lm_head.weight": mx.zeros((4, 4))}

    with pytest.raises(ValueError) as exc_info:
        ensure_matching_weights(expected_weights, weights)

    message = str(exc_info.value)
    assert "Missing: 1, unexpected: 1, shape mismatches: 0." in message
    assert "lm_head.weight -> language_model.lm_head.weight" in message
