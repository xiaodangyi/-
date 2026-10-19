"""在测试集上评估模型并生成混淆矩阵。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flower_classification.data import FlowerDataset, load_manifest
from flower_classification.model import build_model
from flower_classification.plotting import plt
from flower_classification.runtime import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="评估花卉分类模型")
    parser.add_argument("--manifest-dir", default="data/processed")
    parser.add_argument("--data-dir", default=None, help="数据清单对应的数据根目录，可用于更换机器后的路径覆盖")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    records, test_classes = load_manifest(Path(args.manifest_dir) / "test.csv", args.data_dir)
    if test_classes != classes:
        raise ValueError("测试集和模型保存的类别不一致，请确认使用了同一份数据划分")
    dataset = FlowerDataset(records, classes, checkpoint["image_size"], training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = select_device(args.device)
    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
            for actual, predicted in zip(labels.numpy(), predictions):
                matrix[actual, predicted] += 1
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    rows = []
    for index, name in enumerate(classes):
        total = matrix[index].sum()
        rows.append({"class_name": name, "images": int(total), "correct": int(matrix[index, index]), "accuracy": float(matrix[index, index] / max(total, 1))})
    pd.DataFrame(rows).to_csv(output_dir / "class_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    figure, axis = plt.subplots(figsize=(max(6, len(classes) * 0.6), max(5, len(classes) * 0.5)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
    axis.set_title(f"混淆矩阵，准确率 {accuracy:.4f}")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    print(f"测试集准确率：{accuracy:.4f}，样本数：{matrix.sum()}，设备：{device}")


if __name__ == "__main__":
    main()
