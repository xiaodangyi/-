"""验证零预测类别和宏平均计算，避免只看总体准确率。"""

import unittest
import numpy as np
from flower_classification.metrics import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_missing_predictions_are_zero_not_nan(self):
        values = classification_metrics(np.array([[8, 2], [0, 0]]))
        self.assertAlmostEqual(values["accuracy"], 0.8)
        self.assertAlmostEqual(values["macro_recall"], 0.4)
        self.assertAlmostEqual(values["macro_f1"], 4 / 9)

    def test_perfect_predictions(self):
        values = classification_metrics(np.diag([2, 5, 1]))
        for value in values.values():
            self.assertEqual(value, 1.0)
