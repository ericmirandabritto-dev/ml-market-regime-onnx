param(
    [string]$ModelPath = "artifacts\model.onnx",
    [string]$TargetRelativePath = "ML_MarketRegime_ONNX\model.onnx"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$mql5Root = Resolve-Path (Join-Path $projectRoot "..\..\..")
$targetPath = Join-Path (Join-Path $mql5Root "Files") $TargetRelativePath
$sourcePath = Resolve-Path $ModelPath
$sourceDir = Split-Path -Parent $sourcePath
$targetDir = Split-Path $targetPath -Parent

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force

foreach ($sidecarName in @("model_metadata.json", "runtime_contract.csv")) {
    $sidecarSource = Join-Path $sourceDir $sidecarName
    if (Test-Path $sidecarSource) {
        Copy-Item -LiteralPath $sidecarSource -Destination (Join-Path $targetDir $sidecarName) -Force
    }
}

Write-Host "Modelo copiado para: $targetPath"
