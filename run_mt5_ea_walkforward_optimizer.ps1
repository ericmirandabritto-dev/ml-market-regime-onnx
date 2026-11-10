param(
    [string]$PythonExe = "",
    [string]$ConfigFile = "config.mt5_ea_walkforward.example.yaml",
    [switch]$DryRun
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
    "src\run_mt5_ea_walkforward_optimizer.py",
    "--config", $ConfigFile
)

if ($DryRun) {
    $arguments += "--dry-run"
}

Write-Host "Usando interpretador: $resolvedPython"
Write-Host "Config: $ConfigFile"
if ($DryRun) {
    Write-Host "Modo: dry-run"
}

if ($resolvedPython -eq "py -3") {
    & py -3 @arguments
}
else {
    & $resolvedPython @arguments
}
