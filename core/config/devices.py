from typing import Dict

import torch


def resolve_devices(vram_gb: float = -1.0) -> Dict[str, torch.device]:
    """
    Resolves the devices for the different models based on available VRAM.

    Args:
        vram_gb: The amount of VRAM in GB. If -1.0, it will be detected automatically.

    Returns:
        A dictionary mapping model types to torch devices.
    """
    if vram_gb == -1.0 and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if vram_gb > 6:
        return {
            "llm": torch.device("cuda:0"),
            "reranker": torch.device("cuda:0"),
            "embed": torch.device("cuda:0"),
        }
    elif vram_gb > 2:
        return {
            "llm": torch.device("cuda:0"),
            "reranker": torch.device("cuda:0"),
            "embed": torch.device("cpu"),
        }
    else:
        return {
            "llm": torch.device("cpu"),
            "reranker": torch.device("cpu"),
            "embed": torch.device("cpu"),
        }
