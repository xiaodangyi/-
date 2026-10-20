"""训练手写 ResNet-18 并保存最佳模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flower_classification.data import FlowerDataset, load_manifest
from flower_classification.model import build_model
from flower_classification.metrics import classification_metrics
from flower_classification.plotting import plt
from flower_classification.runtime import select_device


def set_seed(seed: int) -> None:
    """固定随机种子，降低实验结果的随机波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, training: bool, epoch_label: str, log_every: int) -> tuple[float, float, float]:
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    started = time.monotonic()
    phase = "训练" if training else "验证"
    matrix = None
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        if matrix is None:
            matrix = np.zeros((logits.shape[1], logits.shape[1]), dtype=np.int64)
        np.add.at(matrix, (labels.detach().cpu().numpy(), logits.detach().argmax(dim=1).cpu().numpy()), 1)
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(loader):
            print(
                f"轮次 {epoch_label} | {phase}批次 {batch_index}/{len(loader)} | "
                f"损失 {total_loss / total:.4f} | 准确率 {correct / total:.2%} | "
                f"本阶段耗时 {time.monotonic() - started:.1f} 秒",
                flush=True,
            )
    return total_loss / max(total, 1), correct / max(total, 1), classification_metrics(matrix)["macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description="训练手写 ResNet-18")
    parser.add_argument("--manifest-dir", default="data/processed")
    parser.add_argument("--data-dir", default=None, help="数据清单对应的数据根目录，可用于更换机器后的路径覆盖")
    parser.add_argument("--output-dir", default="outputs/experiment-001")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=10, help="每隔多少个批次打印一次实时进度")
    parser.add_argument("--augmentation", choices=["basic", "crop-color"], default="basic")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=0, help="验证准确率连续未提高的轮数，0 表示不启用早停")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1 or args.log_every < 1:
        raise SystemExit("epochs、batch-size、image-size 和 log-every 必须大于 0")
    if not 0 <= args.label_smoothing < 1 or args.weight_decay < 0 or args.learning_rate <= 0 or args.patience < 0:
        raise SystemExit("标签平滑必须在 [0,1)，权重衰减和耐心轮数不能为负，学习率必须大于 0")
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"输出目录非空，为保留已有实验已停止；请指定新的 --output-dir：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records, classes = load_manifest(Path(args.manifest_dir) / "train.csv", args.data_dir)
    val_records, val_classes = load_manifest(Path(args.manifest_dir) / "val.csv", args.data_dir)
    if val_classes != classes:
        raise ValueError("训练集和验证集类别不一致，请重新生成数据划分")
    if "sha256" in train_records and "sha256" in val_records:
        shared_hashes = set(train_records["sha256"]) & set(val_records["sha256"])
        if shared_hashes:
            raise ValueError("训练集和验证集存在相同内容，已停止训练")
    train_set = FlowerDataset(train_records, classes, args.image_size, training=True, augmentation=args.augmentation)
    val_set = FlowerDataset(val_records, classes, args.image_size, training=False)
    device = select_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    validation_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_accuracy = -1.0
    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_macro_f1 = 0.0
    stale_epochs = 0
    manifest_hashes = {
        name: hashlib.sha256((Path(args.manifest_dir) / name).read_bytes()).hexdigest()
        for name in ("train.csv", "val.csv")
    }
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"设备：{device_name} | 训练 {len(train_set)} 张 | 验证 {len(val_set)} 张", flush=True)
    print(f"共 {args.epochs} 轮，批大小 {args.batch_size}，结果目录：{output_dir}", flush=True)
    print(f"增强：{args.augmentation} | 标签平滑：{args.label_smoothing} | 权重衰减：{args.weight_decay}", flush=True)
    print("本窗口正在实际训练。按 Ctrl+C 可中断；已保存的最佳模型和完整轮次日志会保留。", flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_label = f"{epoch:03d}/{args.epochs:03d}"
        train_loss, train_accuracy, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, True, epoch_label, args.log_every)
        val_loss, val_accuracy, val_f1 = run_epoch(model, val_loader, validation_criterion, optimizer, device, False, epoch_label, args.log_every)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "train_macro_f1": train_f1, "val_loss": val_loss, "val_accuracy": val_accuracy, "val_macro_f1": val_f1, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
        print(f"轮次 {epoch_label} 完成 | 训练损失 {train_loss:.4f} | 训练准确率 {train_accuracy:.2%} | 验证损失 {val_loss:.4f} | 验证准确率 {val_accuracy:.2%} | 验证宏平均 F1 {val_f1:.4f}", flush=True)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                    "best_epoch": epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "seed": args.seed,
                    "data_manifest_dir": str(Path(args.manifest_dir).resolve()),
                    "training_args": vars(args),
                    "best_val_macro_f1": val_f1,
                    "manifest_sha256": manifest_hashes,
                },
                output_dir / "best.pt",
            )
            print(f"已更新最佳模型：验证准确率 {best_val_accuracy:.2%}", flush=True)
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(f"验证准确率连续 {stale_epochs} 轮未提高，执行早停。", flush=True)
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    history_df.plot(x="epoch", y=["train_loss", "val_loss"], ax=axes[0], title="损失曲线")
    history_df.plot(x="epoch", y=["train_accuracy", "val_accuracy"], ax=axes[1], title="准确率曲线")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)
    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_val_macro_f1,
        "completed_epochs": len(history),
        "manifest_sha256": manifest_hashes,
        "training_args": vars(args),
        "device_name": device_name,
        "torch_version": str(torch.__version__),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练完成，最佳验证准确率：{best_val_accuracy:.4f}，设备：{device}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练已由 Ctrl+C 中断。已保存结果保留；当前脚本不支持断点续训，重新运行请使用新的输出目录。", flush=True)
        raise SystemExit(130)
