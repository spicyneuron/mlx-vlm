import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import sanitize_key
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def _pop_first(weights, keys):
    for key in keys:
        if key in weights:
            return weights.pop(key)
    return None


def _pop_pair(weights, left_keys, right_keys):
    left_key = next((key for key in left_keys if key in weights), None)
    right_key = next((key for key in right_keys if key in weights), None)
    if left_key is None or right_key is None:
        return None, None
    return weights.pop(left_key), weights.pop(right_key)


class Model(Qwen3_5Model):

    def __init__(self, config: ModelConfig):
        # only initialize nn.Module, skip the initialization of vision_tower and language_model in the parent class
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        # ignore mtp weights
        weights = {key: value for key, value in weights.items() if "mtp." not in key}

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.config.text_config.num_hidden_layers):
            prefix = f"model.language_model.layers.{l}.mlp"
            switch_prefix = f"{prefix}.switch_mlp"
            gate_key = f"{switch_prefix}.gate_proj.weight"
            up_key = f"{switch_prefix}.up_proj.weight"
            down_key = f"{switch_prefix}.down_proj.weight"

            gate_up_weight = weights.pop(f"{prefix}.experts.gate_up_proj", None)
            if gate_up_weight is not None:
                # gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
                gate_weight, up_weight = mx.split(gate_up_weight, 2, axis=-2)
                weights[gate_key] = gate_weight
                weights[up_key] = up_weight
            elif gate_key not in weights and up_key not in weights:
                gate_weight, up_weight = _pop_pair(
                    weights,
                    (
                        f"{prefix}.experts.gate_proj",
                        f"{prefix}.experts.gate_proj.weight",
                    ),
                    (
                        f"{prefix}.experts.up_proj",
                        f"{prefix}.experts.up_proj.weight",
                    ),
                )

                if gate_weight is None or up_weight is None:
                    gate_weights = []
                    up_weights = []
                    for expert_idx in range(self.config.text_config.num_experts):
                        expert_prefix = f"{prefix}.experts.{expert_idx}"
                        gate_weight, up_weight = _pop_pair(
                            weights,
                            (
                                f"{expert_prefix}.gate_proj",
                                f"{expert_prefix}.gate_proj.weight",
                            ),
                            (
                                f"{expert_prefix}.up_proj",
                                f"{expert_prefix}.up_proj.weight",
                            ),
                        )
                        if gate_weight is None or up_weight is None:
                            gate_weights = []
                            up_weights = []
                            break
                        gate_weights.append(gate_weight)
                        up_weights.append(up_weight)

                    if gate_weights:
                        gate_weight = mx.stack(gate_weights)
                        up_weight = mx.stack(up_weights)

                if gate_weight is not None and up_weight is not None:
                    weights[gate_key] = gate_weight
                    weights[up_key] = up_weight

            down_weight = _pop_first(
                weights,
                (
                    f"{prefix}.experts.down_proj",
                    f"{prefix}.experts.down_proj.weight",
                ),
            )
            if down_weight is None and down_key not in weights:
                down_weights = []
                for expert_idx in range(self.config.text_config.num_experts):
                    down_weight = _pop_first(
                        weights,
                        (
                            f"{prefix}.experts.{expert_idx}.down_proj",
                            f"{prefix}.experts.{expert_idx}.down_proj.weight",
                        ),
                    )
                    if down_weight is None:
                        down_weights = []
                        break
                    down_weights.append(down_weight)
                if down_weights:
                    down_weight = mx.stack(down_weights)
            if down_weight is not None:
                weights[down_key] = down_weight

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )

        sanitized_weights = {}
        for key, value in weights.items():
            key = sanitize_key(key)

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in norm_keys):
                if value.ndim == 1:
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights
