# 打开此脚本所在项目的实验流程，所有日志与缓存均放在 E 盘项目中。
param(
    [string]$Plan = 'configs/ablation-20261020.json',
    [string]$Python = 'C:\Users\LENOVO\Documents\Codex\2026-10-18\gen\anaconda-envs\ai-cuda\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONNOUSERSITE = '1'
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$Host.UI.RawUI.WindowTitle = 'Flower Classification - Controlled Experiments'
Push-Location -LiteralPath $projectRoot
try {
    Write-Host '正在进行实际 GPU 对照实验；每项实验参数会显示在下方。Ctrl+C 可中断。' -ForegroundColor Cyan
    & $Python -s -u scripts/run_experiments.py --plan $Plan
    if ($LASTEXITCODE -ne 0) { Write-Host ('实验停止，退出码：' + $LASTEXITCODE) -ForegroundColor Yellow }
} finally { Pop-Location }
