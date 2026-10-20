"""检查网上图片的重复、标签和来源，冻结补充训练与外部检验清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
DCT = np.cos(np.pi * (np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None] / 32)


def image_hashes(path: Path) -> dict[str, str]:
    """同时检查字节、解码像素和感知相似性，不能只凭文件名判重。"""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        pixel_digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        gray = image.convert("L")
        values = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        coefficients = (DCT @ values @ DCT.T).ravel()
        perceptual_bits = coefficients > np.median(coefficients[1:])
        small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
        difference_bits = (small[:, 1:] > small[:, :-1]).ravel()
    encode = lambda bits: format(int("".join("1" if bit else "0" for bit in bits), 2), "016x")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixel_sha256": pixel_digest, "phash": encode(perceptual_bits), "dhash": encode(difference_bits)}


def hamming(first: str, second: str) -> int:
    return bin(int(first, 16) ^ int(second, 16)).count("1")


def is_near(first, second) -> bool:
    return hamming(first["phash"], second["phash"]) <= 6 and hamming(first["dhash"], second["dhash"]) <= 8


def audit(source: Path, audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    cache = audit_dir / "original_image_hashes.csv"
    if cache.exists():
        originals = pd.read_csv(cache, dtype=str).to_dict("records")
    else:
        originals = []
        inventory = pd.read_csv(ROOT / "outputs/analysis/image_inventory.csv")
        for index, row in enumerate(inventory.to_dict("records")):
            if not row["valid"]:
                continue
            path = ROOT / "data/raw" / row["path"]
            originals.append({"path": str(path.relative_to(ROOT)), **image_hashes(path)})
            if (index + 1) % 500 == 0:
                print(f"原图内容检查 {index + 1}/{len(inventory)}", flush=True)
        pd.DataFrame(originals).to_csv(cache, index=False)
    exact = {row["sha256"] for row in originals}
    pixels = {row["pixel_sha256"] for row in originals}
    accepted, exclusions = [], []
    rows = pd.read_csv(source / "source_manifest.csv").sort_values(["class_name", "page_id"]).to_dict("records")
    for index, row in enumerate(rows):
        hashes = image_hashes(ROOT / row["path"])
        if hashes["sha256"] != row["sha256"]:
            raise RuntimeError("下载图片已变化：" + row["path"])
        candidate = {**row, **hashes}
        reason = ""
        if hashes["sha256"] in exact or hashes["pixel_sha256"] in pixels:
            reason = "与原数据集或此前候选内容相同"
        else:
            for prior in originals + accepted:
                if is_near(candidate, prior):
                    reason = "感知哈希近似，保守排除：" + prior["path"]
                    break
        if reason:
            exclusions.append({**candidate, "exclusion_reason": reason})
        else:
            accepted.append(candidate)
            exact.add(hashes["sha256"])
            pixels.add(hashes["pixel_sha256"])
        if (index + 1) % 50 == 0:
            print(f"候选重复检查 {index + 1}/{len(rows)}", flush=True)
    clean = pd.DataFrame(accepted)
    clean.to_csv(audit_dir / "deduplicated_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(exclusions).to_csv(audit_dir / "duplicate_exclusions.csv", index=False, encoding="utf-8-sig")
    review_path = audit_dir / "review.csv"
    previous = pd.read_csv(review_path).set_index("page_id").to_dict("index") if review_path.exists() else {}
    reviews = [{"page_id": row["page_id"], "class_name": row["class_name"], "path": row["path"], "decision": previous.get(row["page_id"], {}).get("decision", "pending"), "reason": previous.get(row["page_id"], {}).get("reason", "")} for row in accepted]
    pd.DataFrame(reviews).to_csv(review_path, index=False, encoding="utf-8-sig")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    for class_name in CLASSES:
        samples = clean[clean.class_name == class_name].to_dict("records")
        for start in range(0, len(samples), 48):
            page = samples[start:start + 48]
            sheet = Image.new("RGB", (8 * 160, 6 * 160), "white")
            draw = ImageDraw.Draw(sheet)
            for cell, row in enumerate(page):
                x, y = (cell % 8) * 160, (cell // 8) * 160
                with Image.open(ROOT / row["path"]) as image:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (152, 128))
                sheet.paste(thumb, (x + (160 - thumb.width) // 2, y))
                draw.text((x + 3, y + 130), str(row["page_id"]), fill="black", font=font)
                draw.text((x + 3, y + 145), class_name, fill="black", font=font)
            sheet.save(audit_dir / f"review-{class_name}-{start // 48 + 1}.jpg", quality=92)
    print(f"内容去重后 {len(accepted)} 张；保守排除 {len(exclusions)} 张。请检查缩略图并逐图记录决定：{review_path}", flush=True)


def prepare(source: Path, audit_dir: Path, destination: Path, holdout_count: int) -> None:
    if destination.exists():
        raise FileExistsError("冻结清单目录已存在，禁止覆盖：" + str(destination))
    clean = pd.read_csv(audit_dir / "deduplicated_candidates.csv")
    reviews = pd.read_csv(audit_dir / "review.csv").fillna("")
    if not set(reviews.decision) <= {"approve", "reject"}:
        raise RuntimeError("存在尚未人工检查的候选图片")
    approved = clean[clean.page_id.isin(reviews.loc[reviews.decision == "approve", "page_id"])].copy()
    for row in approved.to_dict("records"):
        if hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("图片在检查后变化：" + row["path"])
    # 按全局作者分组留出，避免同一作者背景相近的照片跨训练和外部检验。
    reserved = set()
    for class_name in ["daisy", "dandelion", "sunflowers", "roses", "tulips"]:
        frame = approved[approved.class_name == class_name]
        authors = sorted(frame.author.unique(), key=lambda value: hashlib.sha256(("20261020:" + value).encode()).hexdigest())
        for author in authors:
            if len(frame[frame.author.isin(reserved)]) >= holdout_count:
                break
            trial = reserved | {author}
            if any(len(approved[(approved.class_name == priority) & ~approved.author.isin(trial)]) < 15 for priority in ("roses", "tulips")):
                continue
            reserved.add(author)
        if len(frame[frame.author.isin(reserved)]) < holdout_count:
            raise RuntimeError(f"{class_name} 无法同时满足作者分离和 {holdout_count} 张留出要求")
    holdout_parts = []
    for class_name in CLASSES:
        frame = approved[(approved.class_name == class_name) & approved.author.isin(reserved)].copy()
        frame["order"] = frame.page_id.map(lambda value: hashlib.sha256(("20261020:" + str(value)).encode()).hexdigest())
        holdout_parts.append(frame.sort_values("order").head(holdout_count).drop(columns="order"))
    holdout = pd.concat(holdout_parts, ignore_index=True)
    supplement = approved[approved.class_name.isin(["roses", "tulips"]) & ~approved.author.isin(reserved)].copy()
    if set(holdout.author) & set(supplement.author):
        raise RuntimeError("作者分组检查失败")
    destination.mkdir(parents=True)

    def local_records(frame):
        result = frame.copy()
        result["root_dir"] = str(ROOT)
        result["valid"] = True
        result["label_conflict"] = False
        return result

    for split in ("train", "val", "test"):
        original = pd.read_csv(ROOT / "data/processed" / f"{split}.csv")
        original["path"] = original.path.map(lambda value: "data/raw/" + value)
        original = local_records(original)
        if split == "train":
            original = pd.concat([original, local_records(supplement)], ignore_index=True)
        original.to_csv(destination / f"{split}.csv", index=False, encoding="utf-8-sig")
    local_records(holdout).to_csv(destination / "external_test.csv", index=False, encoding="utf-8-sig")
    supplement["assigned_split"] = "supplementary_train"
    holdout["assigned_split"] = "external_test"
    attribution_dir = ROOT / "data_provenance"
    attribution_dir.mkdir(exist_ok=True)
    provenance = pd.concat([supplement, holdout], ignore_index=True)
    provenance.to_csv(attribution_dir / "commons-20261020.csv", index=False, encoding="utf-8-sig")
    reviews.to_csv(attribution_dir / "commons-20261020-review.csv", index=False, encoding="utf-8-sig")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.glob("*.csv")}
    summary = {"source": "Wikimedia Commons", "holdout_per_class": holdout_count, "supplementary_train": supplement.groupby("class_name").size().to_dict(), "external_test": holdout.groupby("class_name").size().to_dict(), "author_overlap": 0, "manifest_sha256": hashes, "selection_policy": "外部检验在模型选择前按内容去重、人工复核和全局作者分组固定，仅在最终方案确定后评估。"}
    (destination / "frozen_dataset.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (attribution_dir / "commons-20261020-split.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="审查补充图片并建立冻结清单")
    parser.add_argument("--source", default="data/web-20261020")
    parser.add_argument("--audit-dir", default="outputs/improvement-20261020/data-audit")
    parser.add_argument("--output-dir", default="data/processed-enriched-20261020")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--holdout-per-class", type=int, default=20)
    args = parser.parse_args()
    source, audit_dir, destination = [(ROOT / value).resolve() for value in (args.source, args.audit_dir, args.output_dir)]
    if any(ROOT not in path.parents for path in (source, audit_dir, destination)):
        raise ValueError("所有输入输出必须在项目目录内")
    if args.prepare:
        prepare(source, audit_dir, destination, args.holdout_per_class)
    else:
        audit(source, audit_dir)


if __name__ == "__main__":
    main()
