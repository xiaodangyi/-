"""展示已有主要结果，并交互识别用户选择的图片。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def preview(path: Path) -> None:
    """使用 Windows 默认程序展示图片，预览失败时仍可继续识别。"""
    try:
        os.startfile(str(path))
    except OSError as error:
        print(f"图片预览未打开：{error}")


def predict(path_text: str, show_preview: bool) -> int:
    image = Path(path_text.strip().strip("\"'")).expanduser()
    if not image.is_absolute():
        image = ROOT / image
    if not image.is_file():
        print(f"没有找到图片：{image}，请粘贴实际文件的完整路径。")
        return 1
    if show_preview:
        preview(image)
    print(f"\n正在识别：{image.name}", flush=True)
    result = subprocess.run(
        [sys.executable, "-s", str(ROOT / "scripts/predict.py"), "--image", str(image), "--device", "auto"],
        cwd=str(ROOT),
    )
    if result.returncode:
        print("识别失败，请检查图片是否为有效的 JPG、PNG 或 WebP 等支持格式。")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="花卉分类验收演示")
    parser.add_argument("--image", help="直接识别指定图片")
    parser.add_argument("--results-only", action="store_true", help="只展示已完成实验的结果")
    parser.add_argument("--no-preview", action="store_true", help="不打开图片窗口")
    args = parser.parse_args()

    print("花卉分类演示：雏菊、蒲公英、玫瑰、向日葵、郁金香", flush=True)
    print("已完成实验的主要结果（2026-10-20）：")
    with (ROOT / "docs/final-evaluation-20261020.csv").open(encoding="utf-8-sig", newline="") as source:
        labels = {"original_test": "原测试集", "external_test": "独立网络检验"}
        for row in csv.DictReader(source):
            if row["model"] == "selected" and row["cohort"] in labels:
                print(f"{labels[row['cohort']]}：{float(row['accuracy']):.2%}（{row['correct']}/{row['images']}）")
    print("原模型网络检验为 85%，本轮为 81%；这次网络泛化未改善。", flush=True)
    if not args.no_preview:
        preview(ROOT / "docs/assets/final-evaluation-20261020.png")
    if args.results_only:
        return 0
    if args.image:
        return predict(args.image, not args.no_preview)

    print(f"\n网上图片可另存到：{ROOT / 'data/demo_images'}")
    print("粘贴图片完整路径后按回车即可识别；输入 q 或按 Ctrl+C 结束。")
    while True:
        try:
            path_text = input("\n图片路径：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n演示结束。")
            return 0
        if path_text.lower() == "q":
            return 0
        if path_text:
            predict(path_text, not args.no_preview)


if __name__ == "__main__":
    raise SystemExit(main())
