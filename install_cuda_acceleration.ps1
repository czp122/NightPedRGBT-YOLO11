$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

Write-Host "Installing PyTorch 2.8.0 with CUDA 12.6 support..."
& $python -m pip install --upgrade --force-reinstall `
    torch==2.8.0 torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed with exit code $LASTEXITCODE"
}

& $python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"

Write-Host "If CUDA available is False, update the NVIDIA driver and run this check again."
