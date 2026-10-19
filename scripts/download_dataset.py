"""下载并安全解压公开的五类花卉示例数据集。"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def download_archive(destination: Path) -> str:
    """将压缩包下载到项目目录，并返回 SHA-256。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "flower-classification-project/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected_size = int(response.headers.get("Content-Length", "0"))
        if expected_size == 0:
            raise RuntimeError("数据源没有提供文件长度，已停止下载")
        if destination.is_file() and destination.stat().st_size == expected_size:
            print(f"发现完整压缩包，跳过下载：{destination}")
        else:
            temporary = destination.with_suffix(destination.suffix + ".part")
            downloaded = 0
            with temporary.open("wb") as archive_file:
                while block := response.read(1024 * 1024):
                    archive_file.write(block)
                    downloaded += len(block)
                    if downloaded % (20 * 1024 * 1024) < 1024 * 1024:
                        print(f"已下载 {downloaded / 1024 / 1024:.0f} / {expected_size / 1024 / 1024:.0f} MiB")
            if downloaded != expected_size:
                raise RuntimeError(f"下载长度不符：期望 {expected_size} 字节，实际 {downloaded} 字节")
            os.replace(temporary, destination)

    digest = hashlib.sha256()
    with destination.open("rb") as archive_file:
        for block in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_archive(archive_path: Path, staging_dir: Path, output_dir: Path) -> int:
    """仅解压预期目录下的普通文件，拒绝路径穿越和链接。"""
    existing_items = [path for path in output_dir.iterdir() if path.name != ".gitkeep"] if output_dir.exists() else []
    if existing_items:
        raise FileExistsError(f"数据目标目录非空，为避免覆盖而停止：{output_dir}")
    if staging_dir.exists():
        raise FileExistsError(f"暂存目录已存在，请先检查内容：{staging_dir}")
    staging_dir.mkdir(parents=True)
    image_count = 0
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeError("数据压缩包为空")
        for member in members:
            name = member.name.replace("\\", "/")
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise RuntimeError(f"压缩包包含不安全路径：{member.name}")
            if relative.parts[0] != "flower_photos":
                raise RuntimeError(f"压缩包出现非预期顶层目录：{member.name}")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(f"压缩包包含不支持的链接或特殊文件：{member.name}")

            parts = relative.parts[1:]
            if not parts:
                continue
            target = staging_dir.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if Path(parts[-1]).suffix.lower() not in allowed_extensions:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"无法读取压缩包成员：{member.name}")
            with source, target.open("wb") as output_file:
                shutil.copyfileobj(source, output_file)
            image_count += 1

    classes = [path for path in staging_dir.iterdir() if path.is_dir()]
    if len(classes) < 2 or image_count == 0:
        raise RuntimeError("解压后没有得到有效的多类别图片")
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in classes:
        shutil.move(str(class_dir), str(output_dir / class_dir.name))
    staging_dir.rmdir()
    return image_count


def main() -> None:
    parser = argparse.ArgumentParser(description="下载花卉示例数据集到项目 data/raw")
    parser.add_argument("--archive", default=str(ROOT / "data" / "downloads" / "flower_photos.tgz"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args()

    archive_path = Path(args.archive).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    staging_dir = ROOT / "data" / "staging" / "flower_photos"
    digest = download_archive(archive_path)
    image_count = extract_archive(archive_path, staging_dir, output_dir)
    print(f"解压完成：{image_count} 张图片，目录：{output_dir}")
    print(f"压缩包 SHA-256：{digest}")


if __name__ == "__main__":
    main()
