# Minimal training set, sequentially; the longest run (A2, ~46 h) only with -WithA2, last.
#   conda activate thesis
#   powershell -ExecutionPolicy Bypass -File experiments\windows\run_min_set.ps1 [-WithA2] [-Only A1,R5]
param([switch]$WithA2, [string[]]$Only = @())
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Root
$LogDir = "experiments\results\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$runs = [ordered]@{
    "A1" = @("--model", "lite",       "--args", "lite",          "--name", "A1_Lite")
    "R5" = @("--model", "lite",       "--args", "svd",           "--name", "R5_Lite-matched")
    "F2" = @("--model", "lite-svd-f", "--args", "svd", "--amp",  "--name", "F2_Lite-SVD-F_AMP")
    "F1" = @("--model", "lite-svd-f", "--args", "svd",           "--name", "F1_Lite-SVD-F_FP32")
}
if ($WithA2) { $runs["A2"] = @("--model", "lite-svd", "--args", "svd", "--name", "A2_Lite-SVD") }

foreach ($id in $runs.Keys) {
    if ($Only.Count -and $Only -notcontains $id) { continue }
    $a = $runs[$id]
    $name = $a[-1]
    if (Test-Path "runs\$name") { Write-Host "skip ${id}: runs\$name exists" -ForegroundColor Yellow; continue }
    Write-Host "=== $id $name $(Get-Date -Format s)" -ForegroundColor Cyan
    $t = Measure-Command { python experiments\train.py @a 2>&1 | Tee-Object -FilePath "$LogDir\$name.txt" | Out-Host }
    Write-Host "=== $id exit $LASTEXITCODE, $([math]::Round($t.TotalHours, 2)) h" -ForegroundColor Cyan
    if ($LASTEXITCODE -ne 0) { Write-Host "stopping queue: $id failed" -ForegroundColor Red; exit 1 }
}
Get-Content experiments\train_log.jsonl -Tail 5
