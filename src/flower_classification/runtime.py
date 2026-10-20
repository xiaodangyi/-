"""训练和推理共用的运行配置。"""

from __future__ import annotations

import json
from pathlib import Path
import torch


def resolve_checkpoint(requested: str | None = None) -> Path:
    """优先使用显式权重路径，否则读取按验证集固定的正式模型记录。"""
    if requested:
        path = Path(requested).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2]
        registry = root / "configs/selected-model.json"
        if not registry.is_file():
            raise FileNotFoundError("尚无正式模型选择记录，请用 --checkpoint 指定已训练的权重")
        configuration = json.loads(registry.read_text(encoding="utf-8"))
        path = (root / configuration["checkpoint"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"模型权重不存在：{path}。权重不随 Git 下载，需要先完成对应训练。")
    return path


def select_device(requested: str) -> torch.device:
    """选择设备，并在用户明确要求 CUDA 但不可用时指出原因。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 未检测到可用 CUDA，请使用 --device cpu 或检查显卡环境")
    return torch.device(requested)
