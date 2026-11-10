param(
    [string]$PythonExe = "",
    [string]$ConfigFile = "config.research_db.example.yaml",
    [string]$DatabaseFile = "",
    [switch]$Force,
    [switch]$IncludeOosPredictions
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Resolve-PythonExe {
    param([string]$Requested)
    if ($Requested -and (Test-Path $Requested)) { return $Requested }
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) { return $venvPython }
    if ($env:VIRTUAL_ENV) {
        $activeVenvPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path $activeVenvPython) { return $activeVenvPython }
    }
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) { return $pythonCmd.Source }
    return "py -3"
}

$resolvedPython = Resolve-PythonExe -Requested $PythonExe
$arguments = @(
    "src\run_research_db_sync.py",
    "--config", $ConfigFile
)

if ($DatabaseFile) {
    $arguments += @("--db", $DatabaseFile)
}

if ($Force) {
    $arguments += "--force"
}
if ($IncludeOosPredictions) {
    $arguments += "--include-oos-predictions"
}

Write-Host "Usando interpretador: $resolvedPython"
Write-Host "Config: $ConfigFile"
if ($DatabaseFile) {
    Write-Host "Banco: $DatabaseFile"
}
if ($Force) {
    Write-Host "Modo: force"
}
if ($IncludeOosPredictions) {
    Write-Host "OOS predictions: enabled"
}

if ($resolvedPython -eq "py -3") {
    & py -3 @arguments
}
else {
    & $resolvedPython @arguments
}
