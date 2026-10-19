"""训练和推理共用的运行配置。"""

from __future__ import annotations

import torch


def select_device(requested: str) -> torch.device:
    """选择设备，并在用户明确要求 CUDA 但不可用时指出原因。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 未检测到可用 CUDA，请使用 --device cpu 或检查显卡环境")
    return torch.device(requested)
