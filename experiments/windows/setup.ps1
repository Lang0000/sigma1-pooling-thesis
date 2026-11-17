# One-time environment setup on Windows (run from the repo root):
#   powershell -ExecutionPolicy Bypass -File experiments\windows\setup.ps1 [-CudaTag cu128] [-EnvName thesis]
param(
    [string]$EnvName = "thesis",
    [string]$CudaTag = "cu128"    # RTX 50xx (sm_120) needs CUDA >= 12.8; a newer cu12x/cu13x wheel also works
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Root

function Run([string]$cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "failed: $cmd" }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw "conda not found: install Miniconda and run 'conda init powershell'" }

$envs = conda env list | Out-String
if ($envs -notmatch "(?m)^$EnvName\s") { Run "conda create -n $EnvName python=3.13 -y" }
$Py = "conda run --no-capture-output -n $EnvName python"

Run "$Py -m pip install --upgrade pip"
Run "$Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$CudaTag"
Run "$Py -m pip install -e ."
Run "$Py -m pip install timm pandas matplotlib scipy fvcore nvidia-ml-py ultralytics-thop"
# S2.2 versions + S3.2 model build (expected: 4556695 params, stride [8, 16, 32]); S3.1 pooling checks
Run "$Py experiments\check_build.py"
Push-Location experiments; Run "$Py test_pool.py"; Pop-Location

# environment snapshot
$EnvDir = "experiments\results\00_env"
New-Item -ItemType Directory -Force $EnvDir | Out-Null
Invoke-Expression "$Py -m pip freeze" | Out-File -Encoding utf8 "$EnvDir\pip_freeze_pc.txt"
nvidia-smi | Out-File -Encoding utf8 "$EnvDir\nvidia_smi_pc.txt"
Write-Host "OK. Next: conda activate $EnvName; python experiments\bench_step.py" -ForegroundColor Green
