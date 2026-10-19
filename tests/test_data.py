"""测试图像扫描与数据划分。"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from flower_classification.data import scan_images, split_records

ROOT = Path(__file__).resolve().parents[1]


class ImageDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp_dir.name)
        for class_index, class_name in enumerate(("daisy", "rose")):
            (self.root / class_name).mkdir()
            for index in range(4):
                image = Image.new("RGB", (24 + index, 20), (index * 30, 80 + class_index * 40, 160))
                image.save(self.root / class_name / f"{index}.png")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_identical_files_stay_in_one_split(self) -> None:
        shutil.copy2(self.root / "daisy" / "0.png", self.root / "daisy" / "duplicate.png")
        records = scan_images(self.root)
        splits = split_records(records, seed=17)
        for digest in records["sha256"].unique():
            locations = [name for name, split in splits.items() if digest in split["sha256"].values]
            self.assertEqual(len(locations), 1)
        for class_name in ("daisy", "rose"):
            for split in splits.values():
                self.assertIn(class_name, split["class_name"].tolist())

    def test_corrupt_image_is_reported_and_excluded(self) -> None:
        (self.root / "daisy" / "broken.png").write_bytes(b"not an image")
        records = scan_images(self.root)
        broken = records[records["path"].str.endswith("broken.png")].iloc[0]
        self.assertFalse(bool(broken["valid"]))
        self.assertTrue(bool(broken["error"]))

    def test_same_bytes_with_conflicting_labels_stop_split(self) -> None:
        shutil.copy2(self.root / "daisy" / "0.png", self.root / "rose" / "wrong-label.png")
        records = scan_images(self.root)
        with self.assertRaisesRegex(ValueError, "类别标签不同"):
            split_records(records)

    def test_each_class_needs_three_distinct_images(self) -> None:
        (self.root / "rose" / "2.png").unlink()
        (self.root / "rose" / "3.png").unlink()
        records = scan_images(self.root)
        with self.assertRaisesRegex(ValueError, "至少一张"):
            split_records(records)


if __name__ == "__main__":
    unittest.main()
