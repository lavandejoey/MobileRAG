# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.10.0
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import TextIteratorStreamer

from core.config.settings import Settings
from core.generator.llm import LLMGenerator


@pytest.fixture
def mock_settings_cpu():
    return Settings(device="cpu", quantization="none")


@pytest.fixture
def mock_settings_cuda_4bit():
    return Settings(device="cuda:0", quantization="4bit")


@pytest.fixture
def mock_settings_cuda_8bit():
    return Settings(device="cuda:0", quantization="8bit")


@pytest.fixture
def mock_auto_tokenizer():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_patch:
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.batch_decode.return_value = ["decoded text"]
        mock_tokenizer_patch.return_value = mock_tokenizer
        yield mock_tokenizer_patch


@pytest.fixture
def mock_auto_model_for_causal_lm():
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.return_value.eval.return_value = None
        yield mock_model


@pytest.fixture
def mock_resolve_devices():
    with patch("core.config.devices.resolve_devices") as mock_rd:
        with patch("torch.cuda.is_available", return_value=False):
            mock_rd.return_value = {
                "llm": torch.device("cpu"),
                "reranker": torch.device("cpu"),
                "embed": torch.device("cpu"),
            }
            yield mock_rd


def test_llm_generator_init_cpu(
    mock_settings_cpu, mock_auto_tokenizer, mock_auto_model_for_causal_lm, mock_resolve_devices
):
    generator = LLMGenerator(mock_settings_cpu)
    assert generator.device == torch.device("cpu")
    mock_auto_tokenizer.assert_called_once_with(generator.model_name)
    mock_auto_model_for_causal_lm.assert_called_once_with(
        generator.model_name,
        quantization_config=None,
        device_map=None,
        torch_dtype=torch.float32,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llm_generator_init_cuda_4bit(
    mock_settings_cuda_4bit, mock_auto_tokenizer, mock_auto_model_for_causal_lm
):
    with patch("core.config.devices.resolve_devices") as mock_rd:
        mock_rd.return_value = {"llm": torch.device("cuda:0")}
        generator = LLMGenerator(mock_settings_cuda_4bit)
        assert generator.device == torch.device("cuda:0")
        mock_auto_model_for_causal_lm.assert_called_once()
        args, kwargs = mock_auto_model_for_causal_lm.call_args
        assert kwargs["quantization_config"].load_in_4bit is True
        assert kwargs["device_map"] == "auto"
        assert kwargs["torch_dtype"] == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llm_generator_init_cuda_8bit(
    mock_settings_cuda_8bit, mock_auto_tokenizer, mock_auto_model_for_causal_lm
):
    with patch("core.config.devices.resolve_devices") as mock_rd:
        mock_rd.return_value = {"llm": torch.device("cuda:0")}
        generator = LLMGenerator(mock_settings_cuda_8bit)
        assert generator.device == torch.device("cuda:0")
        mock_auto_model_for_causal_lm.assert_called_once()
        args, kwargs = mock_auto_model_for_causal_lm.call_args
        assert kwargs["quantization_config"].load_in_8bit is True
        assert kwargs["device_map"] == "auto"
        assert kwargs["torch_dtype"] == torch.bfloat16


def test_llm_generator_generate_non_streaming(llm_generator_cpu):
    prompt = "Hello, world!"
    result = llm_generator_cpu.generate(prompt)
    assert result == "decoded text"
    llm_generator_cpu.tokenizer.apply_chat_template.assert_called_once()
    llm_generator_cpu.model.generate.assert_called_once()


@patch("core.generator.llm.TextIteratorStreamer", spec=TextIteratorStreamer)
def test_llm_generator_generate_streaming(mock_streamer, llm_generator_cpu):
    prompt = "Hello, world!"
    mock_streamer_instance = mock_streamer.return_value
    mock_streamer_instance.__iter__.return_value = iter(["token1", "token2"])

    result_iterator = llm_generator_cpu.generate(prompt, stream=True)

    results = list(result_iterator)
    assert results == ["token1", "token2"]

    llm_generator_cpu.model.generate.assert_called_once()


@pytest.fixture
def llm_generator_cpu(
    mock_settings_cpu, mock_auto_tokenizer, mock_auto_model_for_causal_lm, mock_resolve_devices
):
    return LLMGenerator(mock_settings_cpu)
