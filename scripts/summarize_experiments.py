"""只依据验证指标选择正式模型，保存实验表格与对照图。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flower_classification.plotting import plt

NAMES = {"baseline": "原基线", "augmentation": "增强", "label-smoothing": "标签平滑", "weight-decay": "权重衰减", "augmentation-smoothing": "增强＋平滑", "augmentation-smoothing-decay": "增强＋平滑＋衰减", "augmentation-smoothing-enriched": "增强＋平滑＋补图"}


def main():
    parser = argparse.ArgumentParser(description="汇总已完成实验，正式选择仅使用验证集")
    parser.add_argument("--finalize", action="store_true", help="全部实验完成后固定正式模型配置")
    args = parser.parse_args()
    output = ROOT / "outputs/improvement-20261020"
    baseline_dir = ROOT / "outputs/experiment-002"
    baseline_config = json.loads((baseline_dir / "run_config.json").read_text(encoding="utf-8"))
    baseline_metrics = json.loads((output / "baseline-validation/metrics.json").read_text(encoding="utf-8"))
    baseline_history = pd.read_csv(baseline_dir / "history.csv")
    baseline_epoch = int(baseline_history.loc[baseline_history.val_accuracy.idxmax(), "epoch"])
    experiments = [("baseline", baseline_dir, baseline_epoch, baseline_metrics["accuracy"], baseline_metrics["macro_f1"], baseline_config)]
    plans = {}
    incomplete = []
    for plan_name in ("ablation", "combinations", "enriched"):
        path = ROOT / f"configs/{plan_name}-20261020.json"
        plans[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        plan = json.loads(path.read_text(encoding="utf-8"))
        for experiment in plan["experiments"]:
            directory = ROOT / plan["output_root"] / experiment["name"]
            summary_path = directory / "summary.json"
            if not summary_path.exists():
                incomplete.append(experiment["name"])
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            experiments.append((experiment["name"], directory, summary["best_epoch"], summary["best_val_accuracy"], summary["best_val_macro_f1"], summary["training_args"]))
    if incomplete and args.finalize:
        raise SystemExit("仍有未完成实验，禁止提前选择：" + ", ".join(incomplete))
    rows = []
    for name, directory, epoch, accuracy, macro_f1, config in experiments:
        history = pd.read_csv(directory / "history.csv")
        best_row = history[history.epoch == epoch].iloc[0]
        rows.append({"name": name, "best_epoch": epoch, "completed_epochs": len(history), "train_accuracy_at_best": float(best_row.train_accuracy), "val_accuracy": accuracy, "val_macro_f1": macro_f1, "train_val_gap": float(best_row.train_accuracy) - accuracy, "checkpoint": (directory / "best.pt").relative_to(ROOT).as_posix(), "augmentation": config.get("augmentation", "basic"), "label_smoothing": config.get("label_smoothing", 0.0), "weight_decay": config.get("weight_decay", 0.0001), "training_images": len(pd.read_csv(ROOT / config["manifest_dir"] / "train.csv"))})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "validation_comparison.csv", index=False, encoding="utf-8-sig")
    assets = ROOT / "docs/assets"
    assets.mkdir(exist_ok=True)
    selected_index = max(range(len(rows)), key=lambda index: (rows[index]["val_accuracy"], rows[index]["val_macro_f1"]))
    selected = rows[selected_index]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = ["#ed8b23" if index == selected_index else "#4c78a8" for index in range(len(rows))]
    for axis, metric, title in zip(axes, ("val_accuracy", "val_macro_f1"), ("验证准确率", "验证宏平均 F1")):
        bars = axis.bar(range(len(rows)), frame[metric], color=colors)
        axis.set_xticks(range(len(rows)), [NAMES[row["name"]] for row in rows], rotation=28, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(title + "（种子 42，30 轮）")
        axis.bar_label(bars, labels=[f"{value:.2%}" for value in frame[metric]], padding=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(assets / "improvement-20261020-comparison.png", dpi=150)
    plt.close(figure)
    if args.finalize:
        config = experiments[selected_index][5]
        checkpoint = ROOT / selected["checkpoint"]
        selected_config = {"model_name": selected["name"], "checkpoint": selected["checkpoint"], "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "selection_basis": "仅按固定原验证集准确率排序，平分时比较宏平均 F1；不读取历史测试或外部检验预测。", "val_accuracy": selected["val_accuracy"], "val_macro_f1": selected["val_macro_f1"], "best_epoch": selected["best_epoch"], "seed": 42, "training_images": selected["training_images"], "manifest_dir": config["manifest_dir"], "data_dir": config.get("data_dir", "data/raw"), "augmentation": selected["augmentation"], "label_smoothing": selected["label_smoothing"], "weight_decay": selected["weight_decay"], "external_test_manifest": "data/processed-enriched-20261020/external_test.csv", "external_test_manifest_sha256": hashlib.sha256((ROOT / "data/processed-enriched-20261020/external_test.csv").read_bytes()).hexdigest(), "plan_sha256": plans}
        destination = ROOT / "configs/selected-model.json"
        if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != selected_config:
            raise RuntimeError("正式选择记录已经存在且内容不同，请检查，禁止根据检验结果静默换模型")
        destination.write_text(json.dumps(selected_config, ensure_ascii=False, indent=2), encoding="utf-8")
        frame.to_csv(ROOT / "docs/improvement-20261020-results.csv", index=False, encoding="utf-8-sig")
        print("已按验证集固定正式模型：" + selected["checkpoint"], flush=True)
        print(json.dumps(selected_config, ensure_ascii=False, indent=2), flush=True)
    print(frame[["name", "val_accuracy", "val_macro_f1", "training_images"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
