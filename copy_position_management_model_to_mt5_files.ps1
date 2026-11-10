param(
    [string]$PackageDir = "artifacts_position_management\runtime_package",
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

$sourceDir = Resolve-Path $PackageDir

New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
Get-ChildItem -LiteralPath $sourceDir | Copy-Item -Destination $targetRoot -Recurse -Force

Write-Host "Pacote de gestao copiado para: $targetRoot"
