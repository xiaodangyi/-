# 花卉图像分类

这是灵境竞赛组 AI 方向第一阶段考核项目。项目完成以下闭环：

1. 使用 Python、NumPy、Pandas 扫描并整理花卉图片数据。
2. 使用 Matplotlib 输出类别分布、图片尺寸和训练曲线。
3. 不依赖 torchvision，使用 PyTorch 手动实现 ResNet-18。
4. 完成训练、验证、测试、混淆矩阵和单张图片预测。
5. 使用 Git 保留数据分析、模型、训练和文档的开发历史。

## 环境

本项目使用已经配置好的 Anaconda CUDA 环境：

```powershell
$python = "C:\Users\LENOVO\Documents\Codex\2026-10-18\gen\anaconda-envs\ai-cuda\python.exe"
& $python -s -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

已验证的关键版本：Python 3.9.25、PyTorch 2.7.1+cu128、NumPy 1.26.4、Pandas 2.3.3、Matplotlib 3.9.2，CUDA 可用并识别 NVIDIA GeForce RTX 5070 Laptop GPU。
PowerShell 中直接运行 `python` 会优先加载用户级 CPU 版 PyTorch；使用环境解释器并添加 `-s` 可以让 Python 忽略用户级包，确保导入 Anaconda 环境中的 CUDA 版本。

通用依赖见 `requirements.txt`。如果需要在另一台具备兼容 NVIDIA 驱动的机器上安装同一 PyTorch 构建，可使用 `requirements-cu128.txt`。PyTorch 安装包自带所需 CUDA 运行库，普通训练通常不需要另外安装完整 CUDA Toolkit：

```powershell
python -m pip install -r requirements.txt
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-cu128.txt
```

在当前机器上不需要重复安装，直接使用上面的 Anaconda 环境即可。

## 数据目录

把图片按“一个类别一个文件夹”的形式放在 `data/raw` 中，例如：

```text
data/raw/
├── rose/
│   ├── rose_001.jpg
│   └── rose_002.jpg
├── tulip/
│   └── tulip_001.jpg
└── sunflower/
    └── sunflower_001.jpg
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。类别名称来自文件夹名称，图片路径使用相对路径记录，便于在 Git 中复现实验配置。
扫描会记录重复内容和跨类别标签冲突；冲突图片保留在原始目录，但不会进入训练、验证或测试清单。

## 运行流程

在项目根目录执行：

```powershell
$python = "C:\Users\LENOVO\Documents\Codex\2026-10-18\gen\anaconda-envs\ai-cuda\python.exe"

# 下载公开数据集，压缩包和解压数据都放在 E 盘项目目录
& $python -s scripts/download_dataset.py
# 第一步：扫描图片并生成数据分析报告
& $python -s scripts/analyze_dataset.py --data-dir data/raw --output-dir outputs/analysis

# 第二步：固定随机种子，按类别分层划分训练集、验证集和测试集
& $python -s scripts/split_dataset.py --data-dir data/raw --output-dir data/processed

# 第三步：训练手写 ResNet-18
& $python -s scripts/train.py --manifest-dir data/processed --data-dir data/raw --output-dir outputs/experiment-001 --epochs 30

# 第四步：在测试集上评估并输出混淆矩阵
& $python -s scripts/evaluate.py --manifest-dir data/processed --data-dir data/raw --checkpoint outputs/experiment-001/best.pt --output-dir outputs/experiment-001/evaluation

# 第五步：预测一张新图片
& $python -s scripts/predict.py --checkpoint outputs/experiment-001/best.pt --image path/to/flower.jpg
```

如果暂时没有 GPU，可以给训练命令增加 `--device cpu`。训练脚本会自动保存最佳模型、训练日志、类别名称和曲线图。
每隔 10 个批次会显示一次进度，可通过 `--log-every` 调整。输出目录非空时会停止，防止覆盖已有结果。

### 在可见终端中训练

在项目根目录打开 PowerShell，然后运行：

```powershell
powershell.exe -NoLogo -NoProfile -NoExit -File .\scripts\start_training.ps1
```

窗口标题包含 `LIVE Training`，会显示设备、当前轮次、批次、损失和准确率。每次默认生成新的实验目录，完整终端日志保存在 `outputs/terminal-logs`。此启动脚本使用当前机器的 Anaconda 路径，换机器时用 `-Python` 指定解释器。

在实际训练窗口按 **Ctrl+C** 可中断训练，已写入的最佳模型和完整轮次日志会保留；尚未完成的当前轮次不会保存，当前版本不支持断点续训。曲线图仅在全部轮次完成后生成。

`scripts/view_training.ps1` 是另一个只读日志查看工具，它只展示已有 `history.csv`；在查看窗口按 Ctrl+C 只会结束查看。请根据窗口标题区分。

## 项目步骤说明

- 数据分析：先确认类别数量、图片尺寸和损坏文件，避免把数据问题误判成模型问题。
- 数据划分：按类别分层并固定随机种子，保证训练结果可复现，防止某个类别只出现在训练集或测试集。
- 模型实现：残差块通过“卷积变换结果 + 输入捷径”学习特征，深层网络更容易训练。
- 训练与验证：训练集用于更新参数，验证集用于选择最佳模型，测试集只在最后评价泛化能力。
- 评估与预测：除总体准确率外输出每个类别的识别情况，便于定位模型偏差。

## 参考项目与开发边界

参考 `dm9717/flower-classification` 用于了解数据增强、训练验证循环和混淆矩阵等实验组织方式；具体阅读记录见 `docs/参考项目研究.md`。本项目重新设计数据清单、数据集读取、网络结构、训练循环和评估流程，不复制参考项目代码。

项目说明、开发步骤、个人感悟草稿和录屏脚本见 `docs/`。
验收视频的完整 PowerShell 指令、外部图片识别方法、模型效果分析和改进顺序见 [录屏操作指南](docs/录屏脚本.md)。外部演示图片放在 `data/demo_images`，该目录中的图片默认不提交到 Git，也不会自动加入训练。

2026-10-20 的受控改进实验见 [改进实验记录](docs/改进实验记录.md)，固定参数计划位于 `configs/ablation-20261020.json`。`train.py` 支持 `--augmentation crop-color`、`--label-smoothing`、`--weight-decay` 和可选的 `--patience`；默认值保持原有基线设置。可运行 `powershell.exe -NoProfile -NoExit -File .\scripts\start_experiments.ps1` 在可见窗口依次执行计划。

数据集、模型权重、训练输出和个人文件默认不提交到 Git；项目代码、依赖说明、实验配置和分析结果说明会持续提交到远程仓库。
