"""使用训练好的模型预测单张花卉图片。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flower_classification.data import image_to_tensor
from flower_classification.model import build_model
from flower_classification.runtime import select_device, resolve_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="预测单张花卉图片")
    parser.add_argument("--checkpoint", default=None, help="不指定时使用 configs/selected-model.json 中的正式模型")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    checkpoint = torch.load(resolve_checkpoint(args.checkpoint), map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    device = select_device(args.device)
    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_path = Path(args.image).resolve()
    tensor = image_to_tensor(image_path, checkpoint["image_size"])
    with torch.no_grad():
        probabilities = model(tensor.unsqueeze(0).to(device)).softmax(dim=1)[0]
    values, indices = probabilities.topk(min(5, len(classes)))
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        print(f"{classes[index]}: {value:.4f}")


if __name__ == "__main__":
    main()
