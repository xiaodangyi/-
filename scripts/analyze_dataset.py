"""生成数据集统计表和可视化图表。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flower_classification.data import scan_images
from flower_classification.plotting import plt


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描花卉数据集并生成分析报告")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = scan_images(data_dir)
    if records.empty:
        raise SystemExit(f"{data_dir} 中没有找到图片，请先按类别放入数据")

    records.insert(0, "root_dir", str(data_dir))
    records.to_csv(output_dir / "image_inventory.csv", index=False, encoding="utf-8-sig")
    valid = records[records["valid"]].copy()
    duplicate_count = int(valid.duplicated("sha256", keep=False).sum()) if not valid.empty else 0
    summary = (
        records.groupby("class_name", as_index=False)
        .agg(
            images=("path", "count"),
            valid_images=("valid", "sum"),
            average_width=("width", "mean"),
            average_height=("height", "mean"),
        )
    )
    summary["duplicate_valid_images"] = summary["class_name"].map(
        valid[valid.duplicated("sha256", keep=False)].groupby("class_name").size()
    ).fillna(0).astype(int)
    summary.to_csv(output_dir / "class_summary.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary.sort_values("images", ascending=False).plot.bar(x="class_name", y="images", ax=axes[0], legend=False, color="#3d76b9")
    axes[0].set_title("各类别图片数量")
    axes[0].set_xlabel("类别")
    axes[0].set_ylabel("图片数量")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].scatter(valid["width"], valid["height"], alpha=0.45, color="#d97961")
    axes[1].set_title("图片尺寸分布")
    axes[1].set_xlabel("宽度")
    axes[1].set_ylabel("高度")
    figure.tight_layout()
    figure.savefig(output_dir / "dataset_overview.png", dpi=150)
    plt.close(figure)
    print(f"扫描完成：{len(records)} 张图片，{summary['class_name'].nunique()} 个类别，重复内容图片 {duplicate_count} 张")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
