import torch

from core.config.devices import resolve_devices


def test_resolve_devices_no_gpu():
    devices = resolve_devices(vram_gb=0)
    assert devices["llm"] == torch.device("cpu")
    assert devices["reranker"] == torch.device("cpu")
    assert devices["embed"] == torch.device("cpu")


def test_resolve_devices_low_vram():
    devices = resolve_devices(vram_gb=4)
    assert devices["llm"] == torch.device("cuda:0")
    assert devices["reranker"] == torch.device("cuda:0")
    assert devices["embed"] == torch.device("cpu")


def test_resolve_devices_high_vram():
    devices = resolve_devices(vram_gb=8)
    assert devices["llm"] == torch.device("cuda:0")
    assert devices["reranker"] == torch.device("cuda:0")
    assert devices["embed"] == torch.device("cuda:0")
