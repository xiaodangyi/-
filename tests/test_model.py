"""测试手写 ResNet-18 的前向和反向计算。"""

from __future__ import annotations

import unittest

import torch

from flower_classification.model import build_model


class ResNetTests(unittest.TestCase):
    def test_logits_and_gradients(self) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = build_model(num_classes=5)
            images = torch.randn(2, 3, 64, 64)
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (2, 5))
            logits.sum().backward()
            self.assertIsNotNone(model.layer1[0].conv1.weight.grad)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
