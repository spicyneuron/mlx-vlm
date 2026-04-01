from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_server_config(monkeypatch):
    server.app.state.server_config = server.empty_server_config()
    monkeypatch.delenv("MLX_TRUST_REMOTE_CODE", raising=False)
    yield
    server.app.state.server_config = server.empty_server_config()


@pytest.mark.parametrize("value", [224, "22", [1.0], [1.5], [True], [1, 2, 3]])
def test_chat_completions_endpoint_rejects_invalid_resize_shape(client, value):
    response = client.post(
        "/chat/completions",
        json={
            "model": "demo",
            "messages": [{"role": "user", "content": "Hello"}],
            "resize_shape": value,
        },
    )

    assert response.status_code == 422


def test_chat_request_schema_allows_one_or_two_resize_shape_values():
    resize_shape = server.ChatRequest.model_json_schema()["properties"]["resize_shape"]
    lengths = {
        (item["minItems"], item["maxItems"])
        for item in resize_shape["anyOf"]
        if item.get("type") == "array"
    }

    assert lengths == {(1, 1), (2, 2)}


def test_build_server_config_uses_env_trust_remote_code(monkeypatch):
    monkeypatch.setenv("MLX_TRUST_REMOTE_CODE", "true")

    args = server.build_parser().parse_args([])

    config = server.build_server_config(args)

    assert config["trust_remote_code"] is True
    assert config["generation"] == {}


def test_server_defaults_and_request_overrides_merge_cleanly():
    server_config = {
        "model": "server-model",
        "adapter_path": "server-adapter",
        "trust_remote_code": False,
        "generation": {
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.92,
            "top_k": 5,
            "min_p": 0.2,
        },
    }
    request = server.OpenAIRequest(
        input="Hello",
        max_output_tokens=12,
        top_k=40,
        min_p=0.08,
        repetition_penalty=1.15,
    )
    request_overrides = request.model_dump(
        exclude_none=True,
        exclude_unset=True,
    )
    request_overrides.pop("model", None)
    request_overrides.pop("stream", None)
    request_overrides.pop("input", None)
    request_overrides["max_tokens"] = request_overrides.pop("max_output_tokens")

    model_name, adapter_path = server.resolve_load_config(
        request.model,
        None,
        server_config,
    )
    generation = server.resolve_generation_config(
        {
            **server_config["generation"],
            **request_overrides,
        },
        model_name,
    )

    assert model_name == "server-model"
    assert adapter_path == "server-adapter"
    assert generation["max_tokens"] == 12
    assert generation["temperature"] == 0.7
    assert generation["top_p"] == 0.92
    assert generation["top_k"] == 40
    assert generation["min_p"] == 0.08
    assert generation["repetition_penalty"] == 1.15


def test_responses_endpoint_uses_server_model_and_resolves_generation_args(
    client
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
    )

    server.app.state.server_config = {
        "model": "server-model",
        "adapter_path": "server-adapter",
        "trust_remote_code": False,
        "generation": {
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.92,
            "top_k": 5,
            "min_p": 0.2,
        },
    }

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ) as mock_get_cached_model,
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/responses",
            json={
                "input": "Hello",
                "max_output_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
                "enable_thinking": False,
                "thinking_budget": 24,
                "thinking_start_token": "<think>",
            },
        )

    assert response.status_code == 200
    assert mock_get_cached_model.call_args.args == ("server-model", "server-adapter")
    assert mock_get_cached_model.call_args.kwargs == {"trust_remote_code": False}
    assert mock_template.call_args.kwargs["enable_thinking"] is False
    assert mock_template.call_args.kwargs["thinking_budget"] == 24
    assert mock_template.call_args.kwargs["thinking_start_token"] == "<think>"
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["temperature"] == 0.7
    assert mock_generate.call_args.kwargs["top_p"] == 0.92
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["enable_thinking"] is False
    assert mock_generate.call_args.kwargs["thinking_budget"] == 24
    assert mock_generate.call_args.kwargs["thinking_start_token"] == "<think>"
    assert response.json()["model"] == "server-model"
    assert response.json()["max_output_tokens"] == 12
    assert response.json()["temperature"] == 0.7
    assert response.json()["top_p"] == 0.92


def test_responses_endpoint_requires_model_without_request_or_server_default(client):
    response = client.post(
        "/responses",
        json={
            "input": "Hello",
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "No model specified. Pass a model in the request or start the server with --model."
    )


def test_chat_completions_endpoint_request_overrides_server_defaults(
    client
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    server.app.state.server_config = {
        "model": "server-model",
        "adapter_path": "server-adapter",
        "trust_remote_code": False,
        "generation": {
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.92,
            "top_k": 5,
            "min_p": 0.2,
        },
    }

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ) as mock_get_cached_model,
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "request-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
                "resize_shape": [512],
            },
        )

    assert response.status_code == 200
    assert mock_get_cached_model.call_args.args == ("request-model", None)
    assert mock_get_cached_model.call_args.kwargs == {"trust_remote_code": False}
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["temperature"] == 0.7
    assert mock_generate.call_args.kwargs["top_p"] == 0.92
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["resize_shape"] == (512, 512)
    assert response.json()["model"] == "request-model"


def test_chat_completions_endpoint_passes_tools_to_template(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    server.app.state.server_config = {
        "model": "server-model",
        "adapter_path": None,
        "trust_remote_code": False,
        "generation": {},
    }

    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt") as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": tools,
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["tools"] == tools


def test_chat_completions_endpoint_keeps_server_adapter_for_matching_model(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    server.app.state.server_config = {
        "model": "server-model",
        "adapter_path": "server-adapter",
        "trust_remote_code": False,
        "generation": {},
    }

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ) as mock_get_cached_model,
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "server-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    assert mock_get_cached_model.call_args.args == ("server-model", "server-adapter")
