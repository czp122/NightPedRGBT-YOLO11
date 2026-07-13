$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$dist = Join-Path $root "dist"
$build = Join-Path $root "build"
$zip = Join-Path $dist "NightPedestrianDetection-Windows-x64.zip"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

& $python -m PyInstaller --noconfirm --clean (Join-Path $root "NightPedestrianDetection.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

$runtimeModel = Join-Path $dist "NightPedestrianDetection\_internal\models\rgbt_best.pt"
if (-not (Test-Path -LiteralPath $runtimeModel)) {
    throw "Bundled RGBT model not found: $runtimeModel"
}

Copy-Item `
    -LiteralPath (Join-Path $root "packaging\使用说明.txt") `
    -Destination (Join-Path $dist "NightPedestrianDetection\使用说明.txt") `
    -Force

if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
Compress-Archive -Path (Join-Path $dist "NightPedestrianDetection") -DestinationPath $zip -CompressionLevel Optimal

Write-Host "EXE: $(Join-Path $dist 'NightPedestrianDetection\NightPedestrianDetection.exe')"
Write-Host "ZIP: $zip"
