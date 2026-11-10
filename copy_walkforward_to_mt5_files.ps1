param(
    [string]$PackageDir = "artifacts",
    [string]$TargetRelativeDir = "ML_MarketRegime_ONNX\walkforward",
    [switch]$UseCommonFiles
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$mql5Root = Resolve-Path (Join-Path $projectRoot "..\..\..")
$terminalDataRoot = Split-Path $mql5Root -Parent
$terminalBaseRoot = Split-Path $terminalDataRoot -Parent

if ($UseCommonFiles) {
    $targetRoot = Join-Path $terminalBaseRoot "Common\Files"
}
else {
    $targetRoot = Join-Path $mql5Root "Files"
}

$targetDir = Join-Path $targetRoot $TargetRelativeDir
$sourceDir = Resolve-Path $PackageDir

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Get-ChildItem -LiteralPath $sourceDir | Copy-Item -Destination $targetDir -Recurse -Force

Write-Host "Pacote walk-forward copiado para: $targetDir"
