# -*- coding: utf-8 -*-
"""
@file: core/generator/llm.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.10.0
"""

from threading import Thread
from typing import Any, Dict, Iterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from core.config.devices import resolve_devices
from core.config.settings import Settings


class LLMGenerator:
    """
    A wrapper for the Qwen3-1.7B language model that handles model loading, quantization,
    and both streaming and non-streaming text generation.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = "Qwen/Qwen3-1.7B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Ensure safe padding & mask
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        resolved_devices = resolve_devices()
        self.device = resolved_devices.get("llm", torch.device("cpu"))

        quantization_config = None
        if self.device.type == "cuda":
            if settings.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            elif settings.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

        # If quantized, prefer device_map="auto";
        # otherwise place the whole model on the chosen device
        device_map = None
        self._uses_device_map = False
        if self.device.type == "cuda" and quantization_config is not None:
            device_map = "auto"
            self._uses_device_map = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        if not self._uses_device_map:
            # Explicitly place the full model on the target device in non-sharded mode
            self.model.to(self.device)
        self.model.eval()

    def generate(
        self, prompt: str, max_new_tokens: int = 4069, stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generates text from a prompt, either as a single string or a stream of tokens.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, concise assistant. "
                    "Do NOT include chain-of-thought, hidden analysis, "
                    "or any <think> tags in your replies. "
                    "If reasoning is needed, keep it internal and return ONLY the final answer. "
                    "Base answers strictly on the user’s query and provided context; "
                    "if unsure, ask a brief clarifying question."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        if not self._uses_device_map:
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        if stream:
            return self._stream_generate(model_inputs, max_new_tokens)
        else:
            generated_ids = self.model.generate(
                model_inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic
                repetition_penalty=1.05,  # light guard against loops
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
            ]
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _stream_generate(self, inputs: Dict[str, Any], max_new_tokens: int) -> Iterator[str]:
        """
        Handles the streaming generation of text.
        """
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Respect device placement rules
        generation_inputs = inputs
        if not self._uses_device_map:
            generation_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_kwargs = dict(
            **generation_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer
