"""检查重编码图片的内容判重与缩放后的感知相似性。"""

from pathlib import Path
import tempfile
import unittest
from PIL import Image, ImageDraw
from scripts.prepare_web_data import image_hashes, is_near
from scripts.collect_web_images import matches_class

ROOT = Path(__file__).resolve().parents[1]


class ImageFingerprintTests(unittest.TestCase):
    def test_hibiscus_is_not_a_rose(self):
        self.assertFalse(matches_class("roses", "Hibiscus rosa-sinensis flower"))
        self.assertTrue(matches_class("roses", "Rosa 'Mostarella' white shrub rose"))

    def test_png_reencoding_keeps_pixel_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            first, second = Path(folder) / "first.png", Path(folder) / "second.png"
            image = Image.new("RGB", (96, 80), "green")
            ImageDraw.Draw(image).ellipse((18, 10, 70, 65), fill="yellow")
            image.save(first, compress_level=0)
            image.save(second, compress_level=9)
            left, right = image_hashes(first), image_hashes(second)
            self.assertNotEqual(left["sha256"], right["sha256"])
            self.assertEqual(left["pixel_sha256"], right["pixel_sha256"])

    def test_resized_photo_is_near_duplicate(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            first, second = Path(folder) / "first.png", Path(folder) / "second.png"
            image = Image.new("RGB", (320, 240), "darkgreen")
            draw = ImageDraw.Draw(image)
            draw.ellipse((60, 30, 220, 190), fill="orange")
            draw.rectangle((145, 100, 155, 235), fill="brown")
            image.save(first)
            image.resize((640, 480), Image.Resampling.LANCZOS).save(second)
            self.assertTrue(is_near(image_hashes(first), image_hashes(second)))
