# 显示训练脚本实际写出的历史记录，并持续检查新增轮次。
param(
    [string]$ExperimentDir = (Join-Path $PSScriptRoot '..\outputs\experiment-001')
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = 'Flower Classification - Training Progress'
$trainingHistory = Join-Path $ExperimentDir 'history.csv'
$displayedEpoch = 0

Write-Host '花卉分类训练进度：读取真实的 history.csv' -ForegroundColor Cyan
Write-Host ('日志位置：' + $trainingHistory)
Write-Host '本窗口仅查看日志。Ctrl+C 结束查看，不会停止其他窗口中的训练。'
Write-Host ''
Write-Host '轮次    训练损失    训练准确率    验证损失    验证准确率'

while ($true) {
    if (Test-Path -LiteralPath $trainingHistory) {
        try {
            $trainingRows = @(Import-Csv -LiteralPath $trainingHistory)
            foreach ($trainingRow in $trainingRows) {
                if ([int]$trainingRow.epoch -gt $displayedEpoch -and $trainingRow.val_accuracy) {
                    Write-Host ('{0,3}/30  {1,10:F4}  {2,10:P2}  {3,10:F4}  {4,10:P2}' -f
                        [int]$trainingRow.epoch,
                        [double]$trainingRow.train_loss,
                        [double]$trainingRow.train_accuracy,
                        [double]$trainingRow.val_loss,
                        [double]$trainingRow.val_accuracy)
                    $displayedEpoch = [int]$trainingRow.epoch
                }
            }
        } catch {
            # 写入日志的瞬间可能读取失败，下一次轮询重新尝试。
        }
    }
    Start-Sleep -Seconds 2
}
