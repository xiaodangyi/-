# 补充花卉图片来源与复现

来源为 [Wikimedia Commons](https://commons.wikimedia.org/)，通过其公开的 [MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page) 查询文件页面、作者、图片描述、逐图许可及缩略图下载地址。图片本身仍归原作者所有，逐图适用的 CC BY、CC BY-SA 或 CC0 许可见 CSV 中的 `license` 和 `license_url`，不得把网页说明的许可当作所有图片统一许可。

## 文件

- `commons-20261020-candidates.csv`：347 张原始候选的来源、署名、许可、下载地址与 SHA-256，可用于按固定资源复现下载。
- `commons-20261020-review.csv`：内容判重后的 341 张候选的逐图检查结果与排除原因。
- `commons-20261020.csv`：实际分配的 78 张补充训练图和 100 张外部检验图，含作者、许可、来源页面、图片路径、哈希和集合名称。
- `commons-20261020-split.json`：各类数量、作者隔离和四份清单的固定哈希。

图片以 Wikimedia 提供的缩略图形式下载，服务器可能对原图缩放或重新编码。作者、原图链接、文件页面和原许可均保留；当前 Git 仓库只保存元数据，不分发图片。展示、另行分发或使用这些图片时，应保留相应署名和许可信息，并遵守各图片的具体许可要求。

## 人工检查与划分

先与原数据集和其他候选比较字节哈希、解码像素哈希及感知哈希，排除 6 张相同或高度相近的图像。逐图查看剩余图片，排除 84 张非目标植物、图画、明显信息不足或近似连拍图像。目标标签对应 Rosa、Tulipa、Leucanthemum、Taraxacum、Helianthus；单瓣野生玫瑰和实际符合属名的郁金香自然变体仍可保留。

从人工通过的 257 张图片中，每类固定 20 张外部检验图，且其作者不出现在新增训练集合中。仅把另一个作者组中的 45 张玫瑰和 33 张郁金香加入训练；剩余 79 张不参与本轮实验。这里只能确认文件署名字段的分组隔离，不能保证不同账号一定代表不同自然人。

这 100 张图片在最终模型选择前固定，预测结果不用于选择超参数。它们仍是小规模、人工筛选的网络图片检验，不能代表所有真实应用场景。原测试集已看过，仅作历史基准复核。

## 复现流程

先按根目录 README 生成原有数据、分析报告和原始划分，再在项目根目录执行：

```powershell
$python = 'C:\Users\LENOVO\Documents\Codex\2026-10-18\gen\anaconda-envs\ai-cuda\python.exe'
& $python -s scripts/collect_web_images.py --replay data_provenance/commons-20261020-candidates.csv
& $python -s scripts/prepare_web_data.py
Copy-Item -LiteralPath '.\data_provenance\commons-20261020-review.csv' -Destination '.\outputs\improvement-20261020\data-audit\review.csv'
& $python -s scripts/prepare_web_data.py --prepare
```

清单目录已存在时，准备脚本会拒绝覆盖。重放下载若图片内容发生变化，SHA-256 校验会明确报错，应核对来源，不能把变化后的图片静默作为同一次实验的数据。
