# 在当前可见终端运行训练，使用已有 Anaconda CUDA 环境。
param(
    [string]$Python = 'C:\Users\LENOVO\Documents\Codex\2026-10-18\gen\anaconda-envs\ai-cuda\python.exe',
    [string]$ExperimentName = ('experiment-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$Epochs = 30
)

$ErrorActionPreference = 'Stop'
if ($ExperimentName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') {
    throw '实验名称只能包含英文字母、数字、下划线和连字符。'
}
if ($Epochs -lt 1) { throw '训练轮数必须大于 0。' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw ('找不到 Anaconda 解释器：' + $Python)
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experimentPath = Join-Path $projectRoot ('outputs\' + $ExperimentName)
if (Test-Path -LiteralPath $experimentPath) {
    throw ('实验目录已存在，禁止覆盖：' + $experimentPath)
}
$logDirectory = Join-Path $projectRoot 'outputs\terminal-logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory ($ExperimentName + '.log')
if (Test-Path -LiteralPath $logPath) { throw ('终端日志已存在：' + $logPath) }
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONNOUSERSITE = '1'
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$Host.UI.RawUI.WindowTitle = ('Flower Classification - LIVE Training - ' + $ExperimentName)
Push-Location -LiteralPath $projectRoot
try {
    Start-Transcript -LiteralPath $logPath -NoClobber | Out-Null
    try {
        Write-Host '正在运行实际训练；按 Ctrl+C 可中断。' -ForegroundColor Cyan
        Write-Host ('本次实验：' + $ExperimentName)
        Write-Host ('终端日志：' + $logPath)
        & $Python -s -u scripts/train.py --manifest-dir data/processed --data-dir data/raw --output-dir $experimentPath --epochs $Epochs --batch-size 32 --image-size 224 --device cuda --seed 42 --log-every 10
        if ($LASTEXITCODE -ne 0) {
            Write-Host ('训练已停止，退出码：' + $LASTEXITCODE + '。请查看上方信息。') -ForegroundColor Yellow
        } else {
            Write-Host ('训练完成，结果保存在：' + $experimentPath) -ForegroundColor Green
        }
    } finally {
        Stop-Transcript | Out-Null
    }
} finally {
    Pop-Location
}
