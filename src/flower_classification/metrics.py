"""根据混淆矩阵统一计算分类指标，供训练和独立评估使用。"""

from __future__ import annotations

import numpy as np


def classification_metrics(matrix: np.ndarray) -> dict[str, float]:
    """无预测或无真实样本的类别按零处理，避免除零和虚高的宏平均。"""
    counts = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(counts).astype(np.float64)
    precision = true_positive / np.maximum(counts.sum(axis=0), 1)
    recall = true_positive / np.maximum(counts.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(true_positive.sum() / max(counts.sum(), 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
