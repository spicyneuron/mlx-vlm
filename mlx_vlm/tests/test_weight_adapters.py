import mlx.core as mx

from mlx_vlm.utils import adapt_weight_structure


def test_adapt_weight_structure_splits_in_proj_weights_and_biases():
    weights = {
        "resampler.attn.in_proj_weight": mx.zeros((12, 4)),
        "resampler.attn.in_proj_bias": mx.zeros((12,)),
    }
    expected_keys = {
        "resampler.attn.q_proj.weight",
        "resampler.attn.k_proj.weight",
        "resampler.attn.v_proj.weight",
        "resampler.attn.q_proj.bias",
        "resampler.attn.k_proj.bias",
        "resampler.attn.v_proj.bias",
    }

    adapted = adapt_weight_structure(weights, expected_keys)

    assert "resampler.attn.in_proj_weight" not in adapted
    assert "resampler.attn.in_proj_bias" not in adapted
    assert adapted["resampler.attn.q_proj.weight"].shape == (4, 4)
    assert adapted["resampler.attn.k_proj.weight"].shape == (4, 4)
    assert adapted["resampler.attn.v_proj.weight"].shape == (4, 4)
    assert adapted["resampler.attn.q_proj.bias"].shape == (4,)
    assert adapted["resampler.attn.k_proj.bias"].shape == (4,)
    assert adapted["resampler.attn.v_proj.bias"].shape == (4,)


def test_adapt_weight_structure_splits_switch_glu_gate_up_and_down_proj():
    weights = {
        "language_model.model.layers.0.feed_forward.experts.gate_up_proj": mx.zeros(
            (8, 4)
        ),
        "language_model.model.layers.0.feed_forward.experts.down_proj": mx.zeros(
            (4, 8)
        ),
    }
    expected_keys = {
        "language_model.model.layers.0.feed_forward.experts.switch_glu.gate_proj.weight",
        "language_model.model.layers.0.feed_forward.experts.switch_glu.up_proj.weight",
        "language_model.model.layers.0.feed_forward.experts.switch_glu.down_proj.weight",
    }

    adapted = adapt_weight_structure(weights, expected_keys)

    assert (
        "language_model.model.layers.0.feed_forward.experts.gate_up_proj"
        not in adapted
    )
    assert "language_model.model.layers.0.feed_forward.experts.down_proj" not in adapted
    assert (
        adapted[
            "language_model.model.layers.0.feed_forward.experts.switch_glu.gate_proj.weight"
        ].shape
        == (4, 4)
    )
    assert (
        adapted[
            "language_model.model.layers.0.feed_forward.experts.switch_glu.up_proj.weight"
        ].shape
        == (4, 4)
    )
    assert (
        adapted[
            "language_model.model.layers.0.feed_forward.experts.switch_glu.down_proj.weight"
        ].shape
        == (4, 8)
    )


def test_adapt_weight_structure_splits_switch_mlp_gate_up_and_down_proj():
    weights = {
        "language_model.model.layers.0.mlp.experts.gate_up_proj": mx.zeros((2, 8, 4)),
        "language_model.model.layers.0.mlp.experts.down_proj": mx.zeros((2, 4, 8)),
    }
    expected_keys = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
    }

    adapted = adapt_weight_structure(weights, expected_keys)

    assert "language_model.model.layers.0.mlp.experts.gate_up_proj" not in adapted
    assert "language_model.model.layers.0.mlp.experts.down_proj" not in adapted
    assert (
        adapted["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"].shape
        == (2, 4, 4)
    )
    assert (
        adapted["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"].shape
        == (2, 4, 4)
    )
    assert (
        adapted["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"].shape
        == (2, 4, 8)
    )


def test_adapt_weight_structure_keeps_existing_targets():
    weights = {
        "resampler.attn.in_proj_weight": mx.zeros((12, 4)),
        "resampler.attn.q_proj.weight": mx.ones((4, 4)),
    }
    expected_keys = {
        "resampler.attn.q_proj.weight",
        "resampler.attn.k_proj.weight",
        "resampler.attn.v_proj.weight",
    }

    adapted = adapt_weight_structure(weights, expected_keys)

    assert "resampler.attn.in_proj_weight" in adapted
    assert adapted["resampler.attn.q_proj.weight"].shape == (4, 4)
