$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$dist = Join-Path $root "dist"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

$version = (& $python -c "from app_utils.version import APP_VERSION; print(APP_VERSION)").Trim()
$runtime = (& $python -c "import torch; print('CUDA' + str(torch.version.cuda).replace('.', '') if torch.version.cuda else 'CPU')").Trim()
$zip = Join-Path $dist "NightPedestrianDetection-v$version-Windows-x64-$runtime.zip"

& $python -m PyInstaller --noconfirm --clean (Join-Path $root "NightPedestrianDetection.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

$runtimeRoot = Join-Path $dist "NightPedestrianDetection"
$requiredFiles = @(
    "NightPedestrianDetection.exe",
    "_internal\yolo11n.pt",
    "_internal\models\rgbt_best.pt",
    "_internal\configs\bytetrack_realtime.yaml"
)
foreach ($relativePath in $requiredFiles) {
    $requiredPath = Join-Path $runtimeRoot $relativePath
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required runtime file not found: $requiredPath"
    }
}

$instructions = @(Get-ChildItem -LiteralPath (Join-Path $root "packaging") -Filter "*.txt")
if ($instructions.Count -ne 1) {
    throw "Expected exactly one packaging instructions file, found $($instructions.Count)"
}
Copy-Item `
    -LiteralPath $instructions[0].FullName `
    -Destination (Join-Path $runtimeRoot $instructions[0].Name) `
    -Force

if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
$compressed = $false
for ($attempt = 1; $attempt -le 5; $attempt++) {
    try {
        if (Test-Path -LiteralPath $zip) {
            Remove-Item -LiteralPath $zip -Force
        }
        Compress-Archive `
            -Path (Join-Path $dist "NightPedestrianDetection") `
            -DestinationPath $zip `
            -CompressionLevel Optimal `
            -ErrorAction Stop
        $compressed = $true
        break
    }
    catch {
        if ($attempt -eq 5) {
            throw
        }
        Write-Warning "ZIP compression attempt $attempt failed; retrying after file locks settle. $($_.Exception.Message)"
        Start-Sleep -Seconds (2 * $attempt)
    }
}
if (-not $compressed) {
    throw "ZIP compression failed: $zip"
}

Write-Host "EXE: $(Join-Path $dist 'NightPedestrianDetection\NightPedestrianDetection.exe')"
Write-Host "ZIP: $zip"
