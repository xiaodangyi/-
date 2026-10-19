"""花卉图片扫描、数据划分和图像预处理。"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_images(data_dir: str | Path) -> pd.DataFrame:
    """扫描类别目录，检查图片能否完整解码并计算内容哈希。"""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{root}")

    records: list[dict[str, object]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            record: dict[str, object] = {
                "path": path.relative_to(root).as_posix(),
                "class_name": class_dir.name,
                "width": np.nan,
                "height": np.nan,
                "mode": "",
                "sha256": "",
                "valid": False,
                "label_conflict": False,
                "error": "",
            }
            try:
                with Image.open(path) as image:
                    record.update(width=image.width, height=image.height, mode=image.mode)
                    image.load()
                record["sha256"] = _sha256(path)
                record["valid"] = True
            except Exception as exc:
                record["error"] = str(exc)
            records.append(record)

    columns = ["path", "class_name", "width", "height", "mode", "sha256", "valid", "label_conflict", "error"]
    result = pd.DataFrame(records, columns=columns)
    valid = result[result["valid"].astype(bool)]
    if not valid.empty:
        label_counts = valid.groupby("sha256")["class_name"].nunique()
        conflict_hashes = set(label_counts[label_counts > 1].index)
        result["label_conflict"] = result["sha256"].isin(conflict_hashes)
    return result


def _split_counts(sample_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """按比例分配数量，并确保三个集合各至少有一个不同图片。"""
    remaining = sample_count - 3
    exact = [remaining * ratio for ratio in ratios]
    counts = [1 + int(value) for value in exact]
    leftover = sample_count - sum(counts)
    order = sorted(range(3), key=lambda index: exact[index] - int(exact[index]), reverse=True)
    for index in order[:leftover]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def split_records(
    records: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """按类别分层划分图片，内容相同的文件始终进入同一个集合。"""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 必须大于 0，且两者之和小于 1")
    if "sha256" not in records.columns:
        raise ValueError("图片清单缺少 sha256 列，请重新扫描数据集")
    valid = records[records["valid"].astype(bool)].copy()
    if "label_conflict" in valid.columns:
        valid = valid[~valid["label_conflict"].astype(bool)]
    if valid.empty:
        raise ValueError("没有找到可用图片")

    groups_by_class: dict[str, list[list[int]]] = {}
    for class_name, class_records in valid.groupby("class_name", sort=True):
        groups = [list(group.index) for _, group in class_records.groupby("sha256", sort=True)]
        if len(groups) < 3:
            raise ValueError(
                f"类别 {class_name} 只有 {len(groups)} 张不同图片；训练、验证、测试集各需至少一张"
            )
        groups_by_class[str(class_name)] = groups
    if len(groups_by_class) < 2:
        raise ValueError("至少需要两个类别才能进行图像分类")

    rng = random.Random(seed)
    ratios = (train_ratio, val_ratio, 1.0 - train_ratio - val_ratio)
    assignments: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_names = ("train", "val", "test")
    for groups in groups_by_class.values():
        rng.shuffle(groups)
        counts = _split_counts(len(groups), ratios)
        cursor = 0
        for name, count in zip(split_names, counts):
            for group in groups[cursor : cursor + count]:
                assignments[name].extend(group)
            cursor += count

    return {
        name: valid.loc[indices].sort_values("path").reset_index(drop=True)
        for name, indices in assignments.items()
    }


def load_manifest(path: str | Path, data_dir: str | Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """读取数据清单，并将相对路径解析为绝对路径。"""
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"数据清单不存在：{manifest}")
    records = pd.read_csv(manifest)
    required = {"path", "class_name"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"数据清单缺少必要列：{', '.join(sorted(missing))}")
    if records.empty:
        raise ValueError(f"数据清单为空：{manifest}")

    if data_dir is not None:
        root = Path(data_dir).expanduser().resolve()
    elif "root_dir" in records.columns:
        roots = records["root_dir"].dropna().astype(str).unique()
        if len(roots) > 1:
            raise ValueError(f"数据清单包含多个数据根目录：{manifest}")
        root = Path(roots[0]).expanduser() if len(roots) else manifest.parent.parent / "raw"
    else:
        root = manifest.parent.parent / "raw"

    def resolve_image(value: object) -> str:
        image_path = Path(str(value)).expanduser()
        return str(image_path.resolve() if image_path.is_absolute() else (root / image_path).resolve())

    records["absolute_path"] = records["path"].map(resolve_image)
    classes = sorted(records["class_name"].astype(str).unique().tolist())
    return records, classes


def image_to_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    """读取图片、居中裁切并按 ImageNet 均值和标准差归一化。"""
    if image_size < 1:
        raise ValueError("image_size 必须大于 0")
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (image_size, image_size),
            method=Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


class FlowerDataset(Dataset[tuple[torch.Tensor, int]]):
    """从 Pandas 数据清单读取图片，并对训练图片进行随机增强。"""

    def __init__(
        self,
        records: pd.DataFrame,
        classes: Sequence[str],
        image_size: int = 224,
        training: bool = False,
    ) -> None:
        if image_size < 1:
            raise ValueError("image_size 必须大于 0")
        if records.empty:
            raise ValueError("数据集不能为空")
        self.records = records.reset_index(drop=True)
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        unknown = set(self.records["class_name"].astype(str)) - set(self.class_to_index)
        if unknown:
            raise ValueError(f"数据中存在未登记的类别：{', '.join(sorted(unknown))}")
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.records.iloc[index]
        path = Path(str(row["absolute_path"]))
        if self.training:
            with Image.open(path) as image:
                image = self._augment(image.convert("RGB"))
                tensor = self._to_tensor(image)
        else:
            tensor = image_to_tensor(path, self.image_size)
        label = self.class_to_index[str(row["class_name"])]
        return tensor, label

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        fitted = ImageOps.fit(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
        )
        array = np.asarray(fitted, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def _augment(image: Image.Image) -> Image.Image:
        """通过翻转、轻微旋转和亮度扰动增加训练样本变化。"""
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.35:
            image = image.rotate(random.uniform(-12, 12), resample=Image.Resampling.BILINEAR)
        if random.random() < 0.3:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        return image
