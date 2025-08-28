#!/usr/bin/env python3

import argparse
import json
import logging
import uuid
from typing import List

from mlx_lm.server import APIHandler, ModelProvider, run
from mlx_vlm.generate import generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load


class VLMModelProvider(ModelProvider):
    """Extended ModelProvider that can load both text-only and vision-language models."""

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        """
        Load model using mlx-vlm's loader first, fallback to mlx-lm for text-only models.
        """
        model_path, adapter_path, draft_model_path = map(
            lambda s: s.lower() if s else None,
            (model_path, adapter_path, draft_model_path),
        )

        model_path = self.default_model_map.get(model_path, model_path)

        if self.model_key == (model_path, adapter_path, draft_model_path):
            return self.model, self.tokenizer

        # Remove the old model if it exists
        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        # Try loading as VLM model first
        if model_path == "default_model":
            if self.cli_args.model is None:
                raise ValueError(
                    "A model path has to be given as a CLI "
                    "argument or in the HTTP request"
                )
            model_path_actual = self.cli_args.model
        else:
            try:
                self._validate_model_path(model_path)
                model_path_actual = model_path
            except Exception:
                raise

        try:
            # Try VLM loading first
            model, processor = load(
                model_path_actual, adapter_path=adapter_path, trust_remote_code=True
            )
            self.model_key = (model_path, adapter_path, draft_model_path)
            self.model = model
            self.tokenizer = processor  # For VLM models, processor acts as tokenizer
            return self.model, self.tokenizer

        except Exception:
            # Fallback to mlx-lm loading for text-only models
            try:
                result = super().load(model_path, adapter_path, draft_model_path)
                return result
            except Exception:
                raise


class VLMAPIHandler(APIHandler):
    """Extended APIHandler with multi-modal support for chat completions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_images_from_messages(self, messages: List[dict]) -> tuple:
        """
        Extract images from chat messages and return processed messages + images.

        Args:
            messages: List of chat message dictionaries

        Returns:
            Tuple of (processed_messages, images_list)
        """
        processed_messages = []
        images = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if isinstance(content, list):
                # Handle structured content (text + images)
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                images.append(image_url.get("url", ""))
                            else:
                                images.append(image_url)

                # Combine text parts into single content string
                if text_parts:
                    combined_text = " ".join(text_parts)
                    processed_messages.append({"role": role, "content": combined_text})
            else:
                # Plain text content
                processed_messages.append({"role": role, "content": content})

        return processed_messages, images

    def handle_chat_completions(self) -> List[int]:
        """
        Handle chat completion requests with multi-modal support.
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Extract images from messages
        messages, images = self.extract_images_from_messages(body["messages"])

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"

        # Check if this is a VLM model (has processor instead of just tokenizer)
        has_vision_support = hasattr(self.model, "vision_model") or hasattr(
            self.tokenizer, "image_processor"
        )

        if has_vision_support and images:
            # Use VLM-specific chat template and tokenization
            try:
                formatted_prompt = apply_chat_template(
                    self.tokenizer,
                    self.model.config if hasattr(self.model, "config") else self.model,
                    messages,
                    num_images=len(images),
                )
                return self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            except Exception:
                # Fallback to standard processing if VLM chat template fails
                pass

        # Fallback to original mlx-lm processing for text-only
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            # Use original message processing from parent class
            from mlx_lm.server import process_message_content

            process_message_content(body["messages"])
            prompt = self.tokenizer.apply_chat_template(
                body["messages"],
                body.get("tools") or None,
                add_generation_prompt=True,
                **self.model_provider.cli_args.chat_template_args,
            )
        else:
            from mlx_lm.server import convert_chat

            prompt = convert_chat(body["messages"], body.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)

        return prompt

    def handle_completion(self, prompt: List[int], stop_id_sequences: List[List[int]]):
        """
        Extended completion handling with VLM support.
        """
        # Extract images from the current request body if available
        images = []
        if hasattr(self, "body") and "messages" in self.body:
            _, images = self.extract_images_from_messages(self.body["messages"])

        # Check if this is a VLM model with images
        has_vision_support = hasattr(self.model, "vision_model") or hasattr(
            self.tokenizer, "image_processor"
        )

        if has_vision_support and images:
            # Use VLM generation
            if self.stream:
                self.end_headers()

            # Convert prompt back to text for VLM processing
            if hasattr(self.tokenizer, "decode"):
                prompt_text = self.tokenizer.decode(prompt)
            else:
                # Fallback if decode not available
                prompt_text = ""

            try:
                if self.stream:
                    # VLM streaming generation
                    for chunk in stream_generate(
                        model=self.model,
                        processor=self.tokenizer,
                        prompt=prompt_text,
                        image=images,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                    ):
                        if chunk is None or not hasattr(chunk, "text"):
                            continue

                        response = self.generate_response(chunk.text, None)
                        self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                        self.wfile.flush()

                    # Send final response
                    response = self.generate_response("", "stop")
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.write("data: [DONE]\n\n".encode())
                    self.wfile.flush()
                else:
                    # VLM non-streaming generation
                    result = generate(
                        model=self.model,
                        processor=self.tokenizer,
                        prompt=prompt_text,
                        image=images,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        verbose=False,
                    )

                    response = self.generate_response(
                        result.text,
                        "stop",
                        prompt_token_count=result.prompt_tokens,
                        completion_token_count=result.generation_tokens,
                    )

                    response_json = json.dumps(response).encode()
                    self.send_header("Content-Length", str(len(response_json)))
                    self.end_headers()
                    self.wfile.write(response_json)
                    self.wfile.flush()

            except Exception as e:
                # Fallback to text-only generation on error
                print(f"VLM generation failed, falling back to text-only: {e}")
                super().handle_completion(prompt, stop_id_sequences)
        else:
            # Use original mlx-lm generation for text-only
            super().handle_completion(prompt, stop_id_sequences)


def main():
    """Main entry point for the extended VLM server."""
    parser = argparse.ArgumentParser(
        description="MLX VLM Extended Server (based on mlx-lm)"
    )

    # Inherit all arguments from mlx-lm server
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=str,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template""",
        default="{}",
    )

    args = parser.parse_args()

    # Parse chat template args
    args.chat_template_args = json.loads(args.chat_template_args)

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("MLX-VLM Extended Server (multimodal support enabled)")

    # Use the extended classes with mlx-lm's run function
    run(args.host, args.port, VLMModelProvider(args), handler_class=VLMAPIHandler)


if __name__ == "__main__":
    main()
