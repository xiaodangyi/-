"""核对已有评测输出并生成最终对照报告，不训练或重新选择模型。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flower_classification.plotting import plt


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    registry_path = ROOT / "configs/selected-model.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    checkpoint = ROOT / registry["checkpoint"]
    assert digest(checkpoint) == registry["checkpoint_sha256"], "选定权重发生变化"
    manifest = ROOT / registry["external_test_manifest"]
    assert digest(manifest) == registry["external_test_manifest_sha256"], "独立检验清单发生变化"
    validation_results = pd.read_csv(ROOT / "docs/improvement-20261020-results.csv")
    classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    output_root = ROOT / "outputs/improvement-20261020"
    directories = {
        ("validation", "baseline"): output_root / "baseline-validation",
        ("original_test", "baseline"): ROOT / "outputs/experiment-002/evaluation",
        ("original_test", "selected"): output_root / "final-evaluation/selected-original-test",
        ("external_test", "baseline"): output_root / "final-evaluation/baseline-external",
        ("external_test", "selected"): output_root / "final-evaluation/selected-external",
    }
    aggregate, per_class, sources = [], [], {}
    for (cohort, model), directory in directories.items():
        metrics_path = directory / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        matrix_path = directory / "confusion_matrix.csv"
        matrix = pd.read_csv(matrix_path, index_col=0).loc[classes, classes]
        count = int(matrix.to_numpy().sum())
        correct = int(np.trace(matrix.to_numpy()))
        assert count == metrics["test_images"], "评测样本数不一致"
        assert abs(correct / count - metrics["accuracy"]) < 1e-12, "准确率与混淆矩阵不一致"
        if model == "selected":
            assert Path(metrics["checkpoint"]).resolve() == checkpoint.resolve(), "评测权重与登记不一致"
        if cohort == "external_test":
            assert Path(metrics["manifest"]).resolve() == manifest.resolve(), "外部检验清单不一致"
        aggregate.append({"cohort": cohort, "model": model, "images": count, "correct": correct,
                          "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
        paths = [metrics_path, matrix_path]
        predictions_path = directory / "predictions.csv"
        if predictions_path.is_file():
            predictions = pd.read_csv(predictions_path)
            actual = predictions.actual_class.eq(predictions.predicted_class)
            recorded = predictions.correct.astype(str).str.lower().eq("true")
            assert len(predictions) == count and int(actual.sum()) == correct, "逐图预测统计不一致"
            assert actual.equals(recorded), "逐图正确标记与类别不一致"
            reconstructed = pd.crosstab(predictions.actual_class, predictions.predicted_class)
            reconstructed = reconstructed.reindex(index=classes, columns=classes, fill_value=0)
            assert np.array_equal(reconstructed.to_numpy(), matrix.to_numpy()), "逐图预测与混淆矩阵不一致"
            if cohort == "external_test":
                assert set(predictions.path) == set(pd.read_csv(manifest).path), "外部预测样本路径不一致"
            paths.append(predictions_path)
        sources.update({path.relative_to(ROOT).as_posix(): digest(path) for path in paths})
        if cohort != "validation":
            class_metrics = pd.read_csv(directory / "class_metrics.csv").set_index("class_name")
            for name in classes:
                row = class_metrics.loc[name]
                images = int(matrix.loc[name].sum())
                hits = int(matrix.loc[name, name])
                assert images == int(row.images) and hits == int(row.correct), "分组统计不一致"
                per_class.append({"cohort": cohort, "model": model, "class_name": name,
                                  "images": images, "correct": hits, "recall": hits / images,
                                  "precision": float(row.precision), "f1": float(row.f1)})
    selected_val = validation_results.set_index("name").loc[registry["model_name"]]
    assert abs(selected_val.val_accuracy - registry["val_accuracy"]) < 1e-12, "登记验证分数不一致"
    val_count = len(pd.read_csv(ROOT / "data/processed/val.csv"))
    aggregate.append({"cohort": "validation", "model": "selected", "images": val_count,
                      "correct": round(registry["val_accuracy"] * val_count),
                      "accuracy": registry["val_accuracy"], "macro_f1": registry["val_macro_f1"]})
    frame = pd.DataFrame(aggregate)
    class_frame = pd.DataFrame(per_class)
    docs = ROOT / "docs"
    frame.to_csv(docs / "final-evaluation-20261020.csv", index=False, encoding="utf-8-sig")
    class_frame.to_csv(docs / "final-evaluation-20261020-classes.csv", index=False, encoding="utf-8-sig")
    evidence = {"selection_unchanged": True, "checkpoint_sha256": digest(checkpoint),
                "registry_sha256": digest(registry_path), "external_manifest_sha256": digest(manifest),
                "source_sha256": sources,
                "limitation": "单种子、小规模作者隔离网络检验；已查看的检验集不能继续用于选型。"}
    (docs / "final-evaluation-20261020-evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    colors = ["#4c78a8", "#e08c24"]
    groups = ["validation", "original_test", "external_test"]
    for index, (model, label) in enumerate((("baseline", "原基线"), ("selected", "按验证集选定"))):
        values = frame[frame.model.eq(model)].set_index("cohort").loc[groups].accuracy * 100
        bars = axes[0].bar(np.arange(3) + (index - 0.5) * 0.34, values, width=0.34,
                           label=label, color=colors[index])
        axes[0].bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=3, fontsize=9)
        external = class_frame[class_frame.cohort.eq("external_test") & class_frame.model.eq(model)]
        recalls = external.set_index("class_name").loc[classes].recall * 100
        bars = axes[1].bar(np.arange(5) + (index - 0.5) * 0.34, recalls, width=0.34, color=colors[index])
        axes[1].bar_label(bars, labels=[f"{value:.0f}" for value in recalls], padding=3, fontsize=9)
    axes[0].set_xticks(range(3), ["验证集\n552 张", "原测试集\n554 张", "独立网络检验\n100 张"])
    axes[0].set_title("总体准确率：网络检验回退 4 个百分点")
    axes[1].set_xticks(range(5), ["雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香"])
    axes[1].set_title("网络图片各类召回率（每类 20 张）")
    for axis in axes:
        axis.set_ylim(0, 115)
        axis.set_yticks(range(0, 101, 20))
        axis.set_ylabel("百分比（%）")
    axes[0].legend(loc="lower left")
    figure.suptitle("2026-10-20 最终评测：数据集内提升，网络泛化未改善")
    figure.text(0.5, 0.02, "单种子结果；两套模型均保留。未依据独立检验成绩重新选型。", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(docs / "assets/final-evaluation-20261020.png", dpi=150)
    plt.close(figure)
    print(frame.to_string(index=False))
    print("权重、清单、混淆矩阵和已有逐图预测核对通过；本次只生成报告。")


if __name__ == "__main__":
    main()
