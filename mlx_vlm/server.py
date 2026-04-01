import argparse
import gc
import importlib
import json
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List, Literal, Optional, Union

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import scan_cache_dir
from mlx_lm.tokenizer_utils import _infer_tool_parser
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Required, TypeAlias, TypedDict

from .generate import (
    DEFAULT_SEED,
    filter_generation_config,
    generate,
    resolve_generation_config,
    stream_generate,
)
from .prompt_utils import apply_chat_template, filter_chat_template_kwargs
from .utils import load
from .version import __version__

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080
ResizeShapeInput: TypeAlias = tuple[int] | tuple[int, int]



def resolve_load_config(
    request_model: Optional[str],
    request_adapter_path: Optional[str],
    server_config: dict[str, Any],
) -> tuple[str, Optional[str]]:
    model_name = request_model or server_config["model"]
    if model_name is None:
        raise HTTPException(
            status_code=400,
            detail="No model specified. Pass a model in the request or start the server with --model.",
        )

    adapter_path = request_adapter_path
    if adapter_path is None and (
        request_model is None or model_name == server_config["model"]
    ):
        adapter_path = server_config["adapter_path"]

    return model_name, adapter_path



def get_model(model_path: str, adapter_path: Optional[str] = None):
    global model_cache
    trust_remote_code = app.state.server_config["trust_remote_code"]
    cache_key = (model_path, adapter_path, trust_remote_code)

    if model_cache.get("cache_key") == cache_key:
        return model_cache["model"], model_cache["processor"], model_cache["config"]

    if model_cache:
        unload_model_sync()

    try:
        print(f"Loading model: {model_path}" + (f", adapter: {adapter_path}" if adapter_path else ""))
        model, processor = load(model_path, adapter_path, trust_remote_code=trust_remote_code)
        config = model.config
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
    }
    return model, processor, config


@asynccontextmanager
async def lifespan(server_app):
    # Startup
    server_config = server_app.state.server_config
    model_path = server_config["model"]
    adapter_path = server_config["adapter_path"]
    if model_path:
        try:
            print(f"Preloading model: {model_path}")
            get_model(model_path, adapter_path)
        except Exception as e:
            print(f"Failed to preload model: {e}")
            print("Server will continue without a preloaded model.")
    yield
    unload_model_sync()


app = FastAPI(
    title="MLX-VLM Inference API",
    description="API for using Vision Language Models (VLMs) and Omni Models (Vision, Audio and Video support) with MLX.",
    version=__version__,
    lifespan=lifespan,
)
app.state.server_config = {"model": None, "adapter_path": None, "trust_remote_code": False, "generation": {}}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES = 10  # Maximum number of images to process at once

# Loading/unloading utilities

model_cache = {}


class FlexibleBaseModel(BaseModel):
    """Base model that ignores unknown OpenAI SDK fields."""

    model_config = ConfigDict(extra="ignore")


class CommonRequest(FlexibleBaseModel):
    model: Optional[str] = Field(
        None,
        description="The model to use for generation.",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response chunk by chunk.",
    )
    temperature: Optional[float] = Field(
        None,
        description="Temperature for sampling.",
    )
    top_p: Optional[float] = Field(
        None,
        description="Top-p sampling.",
    )
    top_k: Optional[int] = Field(
        None,
        description="Top-k sampling cutoff.",
    )
    min_p: Optional[float] = Field(
        None,
        description="Min-p sampling threshold.",
    )
    repetition_penalty: Optional[float] = Field(
        None,
        description="Penalty applied to repeated tokens.",
    )
    logit_bias: Optional[dict[int, float]] = Field(
        None,
        description="Additive logit bias keyed by token id.",
    )
    resize_shape: Optional[ResizeShapeInput] = Field(
        None,
        description="Resize shape for the image. Provide one integer for a square resize or two integers for (height, width).",
    )
    prefill_step_size: Optional[int] = Field(
        None,
        description="Number of tokens to process per prefill step.",
    )
    kv_bits: Optional[int] = Field(
        None,
        description="Number of bits for KV cache quantization.",
    )
    kv_group_size: Optional[int] = Field(
        None,
        description="Group size for KV cache quantization.",
    )
    max_kv_size: Optional[int] = Field(
        None,
        description="Maximum KV size for the prompt cache (tokens).",
    )
    quantized_kv_start: Optional[int] = Field(
        None,
        description="Start index for the quantized KV cache.",
    )
    enable_thinking: Optional[bool] = Field(
        None,
        description="Enable thinking mode in the chat template.",
    )
    thinking_budget: Optional[int] = Field(
        None,
        description="Maximum number of thinking tokens before forcing the end token.",
    )
    thinking_start_token: Optional[str] = Field(
        None,
        description="Token that marks the start of a thinking block.",
    )
    thinking_end_token: Optional[str] = Field(
        None,
        description="Token that marks the end of a thinking block.",
    )



# Synchronous unload function for internal use
def unload_model_sync():
    global model_cache
    if not model_cache:
        return False

    print(
        f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}"
    )
    # Clear references
    model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache()
    print("Model unloaded and cache cleared.")
    return True


# OpenAI API Models

# Models for /responses endpoint


class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["input_text", "text"]
    ]  # The type of the input item. Always `input_text`.


class ResponseInputImageParam(TypedDict, total=False):
    detail: Literal["high", "low", "auto"] = Field(
        "auto", description="The detail level of the image to be sent to the model."
    )
    """The detail level of the image to be sent to the model.

    One of `high`, `low`, or `auto`.
    """
    type: Required[
        Literal["input_image"]
    ]  # The type of the input item. Always `input_image`.
    image_url: Required[str]
    file_id: Optional[str]
    """The ID of the file to be sent to the model.
     NOTE : wouldn't this help the model if we passed the file_id as well to the vlm models
    """


class InputAudio(TypedDict, total=False):
    data: Required[str]
    format: Required[str]


class ResponseInputAudioParam(TypedDict, total=False):
    type: Required[
        Literal["input_audio"]
    ]  # The type of the input item. Always `input_audio`.
    input_audio: Required[InputAudio]


class ImageUrl(TypedDict, total=False):
    url: Required[str]


class ResponseImageUrlParam(TypedDict, total=False):
    type: Required[
        Literal["image_url"]
    ]  # The type of the input item. Always`image_url`.
    image_url: Required[ImageUrl]


ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
    ResponseInputAudioParam,
]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]


class ResponseOutputText(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["output_text"]
    ]  # The type of the output item. Always `output_text`


ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]


class ChatMessage(FlexibleBaseModel):
    role: Literal["user", "assistant", "system", "developer", "tool"] = Field(
        ...,
        description="Role of the message sender (e.g., 'system', 'user', 'assistant').",
    )
    content: Optional[
        Union[
            str,
            ResponseInputMessageContentListParam,
            ResponseOutputMessageContentList,
        ]
    ] = Field(None, description="Content of the message.")
    tool_calls: List = []


class OpenAIRequest(CommonRequest):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """

    input: Union[str, List[ChatMessage]] = Field(
        ..., description="Input text or list of chat messages."
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens to generate.",
    )


class OpenAIUsage(BaseModel):
    """Token usage details including input tokens, output tokens, breakdown, and total tokens used."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class OpenAIErrorObject(BaseModel):
    """Error object returned when the model fails to generate a Response."""

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None


class OpenAIResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for this Response")
    object: Literal["response"] = Field(
        ..., description="The object type of this resource - always set to response"
    )
    created_at: int = Field(
        ..., description="Unix timestamp (in seconds) of when this Response was created"
    )
    status: Literal["completed", "failed", "in_progress", "incomplete"] = Field(
        ..., description="The status of the response generation"
    )
    error: Optional[OpenAIErrorObject] = Field(
        None,
        description="An error object returned when the model fails to generate a Response",
    )
    instructions: Optional[str] = Field(
        None,
        description="Inserts a system (or developer) message as the first item in the model's context",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description="An upper bound for the number of tokens that can be generated for a response",
    )
    model: str = Field(..., description="Model ID used to generate the response")
    output: List[Union[ChatMessage, Any]] = Field(
        ..., description="An array of content items generated by the model"
    )
    output_text: Optional[str] = Field(
        None,
        description="SDK-only convenience property containing aggregated text output",
    )
    temperature: Optional[float] = Field(
        None, ge=0, le=2, description="Sampling temperature between 0 and 2"
    )
    top_p: Optional[float] = Field(
        None, ge=0, le=1, description="Nucleus sampling probability mass"
    )
    truncation: Union[Literal["auto", "disabled"], str] = Field(
        "disabled", description="The truncation strategy to use"
    )
    usage: OpenAIUsage = Field(
        ..., description="Token usage details"
    )  # we need the model to return stats
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )


class BaseStreamEvent(BaseModel):
    type: str


class ContentPartOutputText(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: List[str] = []


class MessageItem(BaseModel):
    id: str
    type: Literal["message"]
    status: Literal["in_progress", "completed"]
    role: str
    content: List[ContentPartOutputText] = []


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"]
    response: OpenAIResponse


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"]
    response: OpenAIResponse


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"]
    output_index: int
    item: MessageItem


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"]
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"]
    output_index: int
    item: MessageItem


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"]
    response: OpenAIResponse


StreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
]

# Models for /chat/completion endpoint


class UsageStats(OpenAIUsage):
    """
    Inherits from OpenAIUsage and adds additional fields for usage statistics.
    """

    prompt_tps: float = Field(..., description="Tokens per second for the prompt.")
    generation_tps: float = Field(
        ..., description="Tokens per second for the generation."
    )
    peak_memory: float = Field(
        ..., description="Peak memory usage during the generation."
    )


class ChatRequest(CommonRequest):
    messages: List[ChatMessage]
    adapter_path: Optional[str] = Field(
        None,
        description="The path to the adapter weights.",
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens to generate.",
    )
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    tools: Optional[List[Any]] = Field(
        None,
        description="Available tools for the model to call.",
    )


class ChatChoice(BaseModel):
    finish_reason: str
    message: ChatMessage


class ChatResponse(BaseModel):
    model: str
    choices: List[ChatChoice]
    usage: Optional[UsageStats]


class ChatStreamChoice(BaseModel):
    index: int = 0
    finish_reason: Optional[str] = None
    delta: ChatMessage


class ChatStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatStreamChoice]
    usage: Optional[UsageStats]


def parse_responses_input(
    input_data: Union[str, List[ChatMessage]],
) -> tuple[list[dict[str, Any]], list[str], Optional[str]]:
    chat_messages = []
    images = []
    instructions = None

    if input_data in ("", []):
        print("no input")
        raise HTTPException(status_code=400, detail="Missing input.")

    if isinstance(input_data, str):
        chat_messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for message in input_data:
            if not isinstance(message, ChatMessage):
                print("not a ChatMessage")
                raise HTTPException(status_code=400, detail="Invalid input format.")

            if message.content is None:
                chat_messages.append({"role": message.role, "content": ""})
                continue

            if isinstance(message.content, str):
                chat_messages.append(
                    {"role": message.role, "content": message.content}
                )
                if message.role == "system":
                    instructions = message.content
                continue

            if not isinstance(message.content, list):
                print("Invalid message content format.")
                raise HTTPException(status_code=400, detail="Invalid input format.")

            for item in message.content:
                if not isinstance(item, dict):
                    print(f"Invalid message content item format: {item}")
                    raise HTTPException(
                        status_code=400,
                        detail="Missing type in input item.",
                    )

                if item["type"] == "input_text":
                    chat_messages.append(
                        {
                            "role": message.role,
                            "content": item["text"],
                        }
                    )
                    if message.role == "system":
                        instructions = item["text"]
                elif item["type"] == "input_image":
                    images.append(item["image_url"])
                else:
                    print(f"invalid input item type: {item['type']}")
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid input item type.",
                    )
    else:
        print("neither string not list")
        raise HTTPException(status_code=400, detail="Invalid input format.")

    if not chat_messages and not images:
        print("no input")
        raise HTTPException(status_code=400, detail="Missing input.")

    return chat_messages, images, instructions


def parse_chat_messages(
    messages: List[ChatMessage],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    images = []
    audio = []
    processed_messages = []

    for message in messages:
        if message.content is None:
            processed_messages.append({"role": message.role, "content": ""})
        elif isinstance(message.content, str):
            processed_messages.append(
                {"role": message.role, "content": message.content}
            )
        elif isinstance(message.content, list):
            text_content = ""
            for item in message.content:
                if isinstance(item, dict):
                    if message.role == "user":
                        if item["type"] == "input_image":
                            images.append(item["image_url"])
                        elif item["type"] == "image_url":
                            images.append(item["image_url"]["url"])
                        elif item["type"] == "input_audio":
                            audio.append(item["input_audio"]["data"])
                    if item["type"] in ("text", "input_text"):
                        text_content = item.get("text", "")
            processed_messages.append(
                {"role": message.role, "content": text_content}
            )

    return processed_messages, images, audio


def process_tool_calls(model_output: str, tool_module, tools):
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )

        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            tool_call_index = 0
            for match in matches:
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    tool_call = tool_module.parse_tool_call(call, tools)
                    called_tool = {}
                    called_tool["type"] = "function"
                    called_tool["index"] = tool_call_index
                    called_tool["id"] = str(uuid.uuid4())
                    called_tool["function"] = {}
                    called_tool["function"]["name"] = tool_call["name"].strip()
                    called_tool["function"]["arguments"] = json.dumps(
                        tool_call["arguments"], ensure_ascii=False
                    )
                    called_tools.append(called_tool)
                    tool_call_index += 1
                except Exception:
                    print(f"Invalid tool call: {call}")
    return dict(calls=called_tools, remaining_text=remaining)


# Models for /models endpoint


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int


class ModelsResponse(BaseModel):
    object: Literal["list"]
    data: List[ModelInfo]


# OpenAI compatile endpoints


def sse_response(gen):
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _stream_responses(
    model, processor, prompt, images, generation_config,
    model_name, response_id, message_id, generated_at, instructions,
):
    try:
        base_response = OpenAIResponse(
            id=response_id,
            object="response",
            created_at=int(generated_at),
            status="in_progress",
            instructions=instructions,
            max_output_tokens=generation_config["max_tokens"],
            model=model_name,
            output=[],
            output_text="",
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )

        yield f"event: response.created\ndata: {ResponseCreatedEvent(type='response.created', response=base_response).model_dump_json()}\n\n"
        yield f"event: response.in_progress\ndata: {ResponseInProgressEvent(type='response.in_progress', response=base_response).model_dump_json()}\n\n"

        message_item = MessageItem(
            id=message_id, type="message", status="in_progress", role="assistant", content=[],
        )
        yield f"event: response.output_item.added\ndata: {ResponseOutputItemAddedEvent(type='response.output_item.added', output_index=0, item=message_item).model_dump_json()}\n\n"

        content_part = ContentPartOutputText(type="output_text", text="", annotations=[])
        yield f"event: response.content_part.added\ndata: {ResponseContentPartAddedEvent(type='response.content_part.added', item_id=message_id, output_index=0, content_index=0, part=content_part).model_dump_json()}\n\n"

        full_text = ""
        for chunk in stream_generate(model=model, processor=processor, prompt=prompt, image=images, **generation_config):
            if chunk is None or not hasattr(chunk, "text"):
                continue
            full_text += chunk.text
            usage_stats = {"input_tokens": chunk.prompt_tokens, "output_tokens": chunk.generation_tokens}
            yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=chunk.text).model_dump_json()}\n\n"

        yield f"event: response.output_text.done\ndata: {ResponseOutputTextDoneEvent(type='response.output_text.done', item_id=message_id, output_index=0, content_index=0, text=full_text).model_dump_json()}\n\n"

        final_content_part = ContentPartOutputText(type="output_text", text=full_text, annotations=[])
        yield f"event: response.content_part.done\ndata: {ResponseContentPartDoneEvent(type='response.content_part.done', item_id=message_id, output_index=0, content_index=0, part=final_content_part).model_dump_json()}\n\n"

        final_message_item = MessageItem(
            id=message_id, type="message", status="completed", role="assistant", content=[final_content_part],
        )
        yield f"event: response.output_item.done\ndata: {ResponseOutputItemDoneEvent(type='response.output_item.done', output_index=0, item=final_message_item).model_dump_json()}\n\n"

        completed_response = base_response.model_copy(
            update={
                "status": "completed",
                "output": [final_message_item],
                "usage": {
                    "input_tokens": usage_stats["input_tokens"],
                    "output_tokens": usage_stats["output_tokens"],
                    "total_tokens": usage_stats["input_tokens"] + usage_stats["output_tokens"],
                },
            }
        )
        yield f"event: response.completed\ndata: {ResponseCompletedEvent(type='response.completed', response=completed_response).model_dump_json()}\n\n"

    except Exception as e:
        print(f"Error during stream generation: {e}")
        traceback.print_exc()
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    finally:
        mx.clear_cache()
        gc.collect()
        print("Stream finished, cleared cache.")


async def _stream_chat(
    model, processor, prompt, images, audio, generation_config,
    model_name, tool_parser_type, tool_module, tools,
):
    try:
        token_gen = stream_generate(
            model=model, processor=processor, prompt=prompt,
            image=images, audio=audio, **generation_config,
        )

        output_text = ""
        request_id = f"chatcmpl-{uuid.uuid4()}"
        for chunk in token_gen:
            if chunk is None or not hasattr(chunk, "text"):
                print("Warning: Received unexpected chunk format:", chunk)
                continue

            output_text += chunk.text
            usage_stats = {
                "input_tokens": chunk.prompt_tokens,
                "output_tokens": chunk.generation_tokens,
                "total_tokens": chunk.prompt_tokens + chunk.generation_tokens,
                "prompt_tps": chunk.prompt_tps,
                "generation_tps": chunk.generation_tps,
                "peak_memory": chunk.peak_memory,
            }
            choices = [ChatStreamChoice(delta=ChatMessage(role="assistant", content=chunk.text))]
            chunk_data = ChatStreamChunk(
                id=request_id, created=int(time.time()), model=model_name,
                usage=usage_stats, choices=choices,
            )
            yield f"data: {chunk_data.model_dump_json()}\n\n"

        if tool_parser_type is not None:
            tool_calls = process_tool_calls(model_output=output_text, tool_module=tool_module, tools=tools)
        else:
            tool_calls = {"calls": []}

        choices = [
            ChatStreamChoice(
                finish_reason="stop",
                delta=ChatMessage(role="assistant", content="", tool_calls=tool_calls["calls"]),
            )
        ]
        chunk_data = ChatStreamChunk(
            id=request_id, created=int(time.time()), model=model_name,
            usage=usage_stats, choices=choices,
        )
        yield f"data: {chunk_data.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"Error during stream generation: {e}")
        traceback.print_exc()
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    finally:
        mx.clear_cache()
        gc.collect()
        print("Stream finished, cleared cache.")


@app.post("/responses")
@app.post("/v1/responses", include_in_schema=False)
async def responses_endpoint(openai_request: OpenAIRequest):
    """
    OpenAI-compatible endpoint for generating text based on a prompt and optional images.

    using client.responses.create method.

    example:

    from openai import OpenAI

    API_URL = "http://0.0.0.0:8000"
    API_KEY = 'any'

    def run_openai(prompt, img_url,system, stream=False, max_output_tokens=512, model="mlx-community/Qwen2.5-VL-3B-Instruct-8bit"):
        ''' Calls the OpenAI API
        '''

        client = OpenAI(base_url=f"{API_URL}", api_key=API_KEY)

        try :
            response = client.responses.create(
                model=model,
                input=[
                    {"role":"system",
                    "content": f"{system}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"{img_url}"},
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
                stream=stream
            )
            if not stream:
                print(response.output[0].content[0].text)
                print(response.usage)
            else:
                for event in response:
                    # Process different event types if needed
                    if hasattr(event, 'delta') and event.delta:
                        print(event.delta, end="", flush=True)
                    elif event.type == 'response.completed':
                        print("\n--- Usage ---")
                        print(event.response.usage)

        except Exception as e:
            # building a response object to match the one returned when request is successful so that it can be processed in the same way
            return {"model - error":str(e),"content":{}, "model":model}

    """

    try:
        chat_messages, images, instructions = parse_responses_input(openai_request.input)

        server_config = app.state.server_config
        model_name, adapter_path = resolve_load_config(openai_request.model, None, server_config)

        raw = openai_request.model_dump(exclude_none=True, exclude_unset=True)
        request_gen = filter_generation_config(raw)
        if openai_request.max_output_tokens is not None:
            request_gen["max_tokens"] = openai_request.max_output_tokens
        merged = {**server_config["generation"], **request_gen}
        try:
            generation_config = resolve_generation_config(merged)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if "qat" in model_name and "kv_bits" in generation_config:
            print(f"Model {model_name} is QAT; kv_bits ignored. Use --max-kv-size instead.")
            generation_config.pop("kv_bits")
            generation_config.pop("max_kv_size", None)
        template_kwargs = filter_chat_template_kwargs({**merged, **raw})

        if len(images) > MAX_IMAGES:
            raise HTTPException(status_code=400, detail=f"Too many images. Maximum supported is {MAX_IMAGES}.")

        model, processor, config = get_model(model_name, adapter_path)
        try:
            formatted_prompt = apply_chat_template(
                processor, config, chat_messages,
                num_images=len(images), num_audios=0,
                **template_kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        generated_at = datetime.now().timestamp()
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if openai_request.stream:
            return sse_response(_stream_responses(
                model, processor, formatted_prompt, images, generation_config,
                model_name, response_id, message_id, generated_at, instructions,
            ))

        else:
            # Non-streaming response
            try:
                # Use generate from generate.py
                result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=images,
                    verbose=False,  # stats are passed in the response
                    **generation_config,
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                response = OpenAIResponse(
                    id=response_id,
                    object="response",
                    created_at=int(generated_at),
                    status="completed",
                    instructions=instructions,
                    max_output_tokens=generation_config["max_tokens"],
                    model=model_name,
                    output=[
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": result.text,
                                }
                            ],
                        }
                    ],
                    output_text=result.text,
                    temperature=generation_config["temperature"],
                    top_p=generation_config["top_p"],
                    usage={
                        "input_tokens": result.prompt_tokens,
                        "output_tokens": result.generation_tokens,
                        "total_tokens": result.total_tokens,
                    },
                )
                return response

            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /responses endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post(
    "/chat/completions", response_model=None
)  # Response model handled dynamically based on stream flag
@app.post("/v1/chat/completions", response_model=None, include_in_schema=False)
async def chat_completions_endpoint(
    request: ChatRequest,
):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored if not already in the prompt.
    Can operate in streaming or non-streaming mode.
    """

    try:
        processed_messages, images, audio = parse_chat_messages(request.messages)

        server_config = app.state.server_config
        model_name, adapter_path = resolve_load_config(request.model, request.adapter_path, server_config)

        raw = request.model_dump(exclude_none=True, exclude_unset=True)
        request_gen = filter_generation_config(raw)
        merged = {**server_config["generation"], **request_gen}
        try:
            generation_config = resolve_generation_config(merged)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if "qat" in model_name and "kv_bits" in generation_config:
            print(f"Model {model_name} is QAT; kv_bits ignored. Use --max-kv-size instead.")
            generation_config.pop("kv_bits")
            generation_config.pop("max_kv_size", None)
        template_kwargs = filter_chat_template_kwargs({**merged, **raw})

        if len(images) > MAX_IMAGES:
            raise HTTPException(status_code=400, detail=f"Too many images. Maximum supported is {MAX_IMAGES}.")

        model, processor, config = get_model(model_name, adapter_path)
        try:
            formatted_prompt = apply_chat_template(
                processor, config, processed_messages,
                num_images=len(images), num_audios=len(audio),
                **template_kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        tool_parser_type = None
        tool_module = None
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        if hasattr(tokenizer, "chat_template"):
            tool_parser_type = _infer_tool_parser(tokenizer.chat_template)
            if tool_parser_type is not None:
                tool_module = importlib.import_module(
                    f"mlx_lm.tool_parsers.{tool_parser_type}"
                )

        if request.stream:
            return sse_response(_stream_chat(
                model, processor, formatted_prompt, images, audio, generation_config,
                model_name, tool_parser_type, tool_module, request.tools,
            ))

        else:
            # Non-streaming response
            try:
                # Use generate from generate.py
                gen_result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=images,
                    audio=audio,
                    verbose=False,  # Keep API output clean
                    **generation_config,
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                usage_stats = UsageStats(
                    input_tokens=gen_result.prompt_tokens,
                    output_tokens=gen_result.generation_tokens,
                    total_tokens=gen_result.total_tokens,
                    prompt_tps=gen_result.prompt_tps,
                    generation_tps=gen_result.generation_tps,
                    peak_memory=gen_result.peak_memory,
                )

                if tool_parser_type is not None:
                    tool_calls = process_tool_calls(
                        model_output=gen_result.text,
                        tool_module=tool_module,
                        tools=request.tools,
                    )
                else:
                    tool_calls = {}
                    tool_calls["calls"] = []
                    tool_calls["remaining_text"] = gen_result.text

                choices = [
                    ChatChoice(
                        finish_reason="stop",
                        message=ChatMessage(
                            role="assistant",
                            content=tool_calls["remaining_text"],
                            tool_calls=tool_calls["calls"],
                        ),
                    )
                ]

                result = ChatResponse(
                    model=model_name,
                    usage=usage_stats,
                    choices=choices,
                )

                return result

            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.get("/models", response_model=ModelsResponse)
@app.get("/v1/models", response_model=ModelsResponse, include_in_schema=False)
def models_endpoint():
    """
    Return list of locally downloaded MLX models.
    """

    files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        return all(f in file_names for f in files)

    # Scan the cache directory for downloaded mlx models
    hf_cache_info = scan_cache_dir()
    downloaded_models = [repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)]

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]

    response = {"object": "list", "data": models}

    return response


# MLX_VLM API endpoints


@app.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    return {
        "status": "healthy",
        "loaded_model": model_cache.get("model_path", None),
        "loaded_adapter": model_cache.get("adapter_path", None),
    }


@app.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": model_cache.get("model_path", None),
        "adapter_name": model_cache.get("adapter_path", None),
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": "Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="Optional Hugging Face repo ID or local path to the MLX model weights, tokenizer, and config.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_SERVER_HOST,
        help="Host for the HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Port for the HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Default maximum number of tokens to generate per request.",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        dest="temperature",
        type=float,
        default=None,
        help="Default sampling temperature per request.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Default top-p sampling value per request.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Default top-k sampling cutoff per request.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Default min-p sampling threshold per request.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="Number of tokens to process per prefill step. "
        "Lower values reduce peak memory usage but may be slower. "
        "Try 512 or 256 if you hit GPU memory errors during prefill.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Number of bits for KV cache quantization.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=None,
        help="Group size for KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV size for the prompt cache (tokens).",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=None,
        help="Start index (of token) for the quantized KV cache.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    env_trust_remote_code = os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
    app.state.server_config = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "trust_remote_code": args.trust_remote_code or env_trust_remote_code,
        "generation": filter_generation_config(vars(args)),
    }
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
