"""从 Wikimedia Commons 查询逐图许可并下载用于人工检查的花卉图片。"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
AGENT = "FlowerClassificationStudent/1.0 (educational classification experiment; Wikimedia attribution retained)"
QUERIES = {
    "roses": ["Rosa flower filetype:bitmap", 'Rosa flower -intitle:"d.j.b" -intitle:"actm" filetype:bitmap'],
    "tulips": ["Tulipa flower filetype:bitmap", 'Tulipa flower -intitle:"d.j.b" -intitle:"actm" filetype:bitmap'],
    "daisy": ["Leucanthemum vulgare flower filetype:bitmap", "Leucanthemum flower filetype:bitmap"],
    "dandelion": ["Taraxacum officinale flower filetype:bitmap", "Taraxacum flower filetype:bitmap"],
    "sunflowers": ["Helianthus annuus flower filetype:bitmap", "Helianthus annuus inflorescence filetype:bitmap"],
}
GENERA = {"roses": "Rosa", "tulips": "Tulipa", "daisy": "Leucanthemum", "dandelion": "Taraxacum", "sunflowers": "Helianthus"}


def matches_class(class_name: str, text: str) -> bool:
    """核对植物属名，避免把含有 rosa 的扶桑学名当成玫瑰属。"""
    if class_name == "roses" and re.search(r"\b(?:Hibiscus|Alcea|Althaea|Adenium|Helleborus)\b", text, re.I):
        return False
    return bool(re.search(r"\b" + GENERA[class_name] + r"\b", text, re.I))


def plain(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", value))).strip()


def get_bytes(url: str) -> bytes:
    """限制单次响应大小，并对短暂网络或限流错误退避重试。"""
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=40) as response:
                body = response.read(16 * 1024 * 1024 + 1)
            if len(body) > 16 * 1024 * 1024:
                raise ValueError("响应超过 16 MiB 限制")
            return body
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("请求未完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="获取许可明确的花卉候选图片，不自动写入训练集")
    parser.add_argument("--output-dir", default="data/web-20261020")
    parser.add_argument("--priority-limit", type=int, default=140)
    parser.add_argument("--other-limit", type=int, default=50)
    parser.add_argument("--expand", action="store_true", help="在现有目录基础上翻页补充候选，不改动已下载图片")
    parser.add_argument("--replay", default=None, help="从版本库中固定的来源清单下载，校验已有 SHA-256")
    args = parser.parse_args()
    target = (ROOT / args.output_dir).resolve()
    if ROOT not in target.parents:
        raise ValueError("下载目录必须位于项目内")
    target.mkdir(parents=True, exist_ok=True)
    catalog = target / "catalog.json"
    if args.replay:
        candidates = pd.read_csv(ROOT / args.replay, keep_default_na=False).to_dict("records")
        print("使用版本库中固定的图片来源和哈希。", flush=True)
    elif catalog.exists() and not args.expand:
        candidates = json.loads(catalog.read_text(encoding="utf-8"))
        print("使用已冻结的图片来源目录。", flush=True)
    else:
        candidates = json.loads(catalog.read_text(encoding="utf-8")) if catalog.exists() else []
        used_ids = {row["page_id"] for row in candidates}
        for class_name, queries in QUERIES.items():
            by_author = Counter(row["author"] for row in candidates if row["class_name"] == class_name)
            limit = args.priority_limit if class_name in {"roses", "tulips"} else args.other_limit
            selected = sum(by_author.values())
            searches = [(query, offset) for offset in ([100, 200] if args.expand else [0]) for query in queries]
            for query_index, (query, offset) in enumerate(searches):
                if selected >= limit:
                    break
                parameters = {"action": "query", "generator": "search", "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": 100, "gsroffset": offset, "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 640, "format": "json", "formatversion": 2}
                result = json.loads(get_bytes(API + "?" + urllib.parse.urlencode(parameters)))
                (target / f"query-{class_name}-{offset}-{query_index}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                if "error" in result:
                    raise RuntimeError(str(result["error"]))
                pages = sorted(result.get("query", {}).get("pages", []), key=lambda page: page.get("index", 0))
                for page in pages:
                    if selected >= limit or page["pageid"] in used_ids:
                        continue
                    info = page.get("imageinfo", [{}])[0]
                    metadata = info.get("extmetadata", {})
                    value = lambda key: plain(metadata.get(key, {}).get("value", ""))
                    license_name, license_url = value("LicenseShortName"), value("LicenseUrl")
                    if not re.match(r"^CC BY(?:-SA)? [1-4]", license_name) and not license_name.startswith("CC0"):
                        continue
                    if not license_url.startswith(("https://creativecommons.org/", "http://creativecommons.org/")):
                        continue
                    author = value("Artist")
                    description = value("ImageDescription")
                    text = page["title"] + " " + description
                    if not matches_class(class_name, text):
                        continue
                    if re.search(r"\b(herbarium|illustration|painting|drawing|stamp|seeds?|fruits?|pollen|dried)\b", page["title"], re.I):
                        continue
                    if not author or by_author[author] >= 12 or info.get("mime") not in {"image/jpeg", "image/png"}:
                        continue
                    if min(info.get("width", 0), info.get("height", 0)) < 224:
                        continue
                    suffix = ".png" if info["mime"] == "image/png" else ".jpg"
                    candidates.append({"class_name": class_name, "page_id": page["pageid"], "title": page["title"], "source_page": info.get("descriptionurl", "https://commons.wikimedia.org/?curid=" + str(page["pageid"])), "image_url": info.get("thumburl", info["url"]), "original_url": info["url"], "author": author, "license": license_name, "license_url": license_url, "description": description, "credit": value("Credit"), "path": (target / "images" / class_name / ("commons_" + str(page["pageid"]) + suffix)).relative_to(ROOT).as_posix(), "query": query})
                    used_ids.add(page["pageid"])
                    by_author[author] += 1
                    selected += 1
                time.sleep(1)
            print(f"来源目录：{class_name} {selected} 张，{len(by_author)} 位署名者", flush=True)
        catalog.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    def download(row):
        destination = (ROOT / row["path"]).resolve()
        if target not in destination.parents:
            raise ValueError("图片路径超出指定下载目录")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            body = get_bytes(row["image_url"])
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(body)
            with Image.open(temporary) as image:
                image.load()
                if min(image.size) < 180:
                    raise ValueError("缩略图过小")
            temporary.replace(destination)
        with Image.open(destination) as image:
            image.load()
            width, height = image.size
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if row.get("sha256") and row["sha256"] != digest:
            raise RuntimeError("服务器图片内容与固定哈希不同，禁止静默替换")
        return {**row, "sha256": digest, "width": width, "height": height, "review_status": "pending"}

    downloaded, failures = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = {executor.submit(download, row): row for row in candidates}
        for future in as_completed(tasks):
            row = tasks[future]
            try:
                downloaded.append(future.result())
            except Exception as exc:
                failures.append({"page_id": row["page_id"], "class_name": row["class_name"], "error": str(exc)})
            if (len(downloaded) + len(failures)) % 20 == 0:
                print(f"下载进度：成功 {len(downloaded)}，失败 {len(failures)}，共 {len(candidates)}", flush=True)
                pd.DataFrame(downloaded).to_csv(target / "source_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(downloaded).sort_values(["class_name", "page_id"]).to_csv(target / "source_manifest.csv", index=False, encoding="utf-8-sig")
    (target / "download_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print("下载完成，下一步必须检查标签、内容重复和图片质量，再建立训练与外部检验清单。", flush=True)


if __name__ == "__main__":
    main()
