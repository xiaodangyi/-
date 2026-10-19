"""生成训练集、验证集和测试集清单。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flower_classification.data import scan_images, split_records


def main() -> None:
    parser = argparse.ArgumentParser(description="分层划分花卉数据集")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = scan_images(data_dir)
    conflict_count = int(records["label_conflict"].sum())
    splits = split_records(records, args.train_ratio, args.val_ratio, args.seed)
    if conflict_count:
        print(f"为避免错误标签和数据泄漏，清单排除了 {conflict_count} 张跨类别重复图片；原始图片未修改")
    for name, split in splits.items():
        split.insert(0, "root_dir", str(data_dir))
        split.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: {len(split)} 张图片")


if __name__ == "__main__":
    main()
