param(
    [string]$PythonExe = "",
    [string]$CandlesFile = "data\raw\us100_m5_candles.csv",
    [string]$ReportFile = "report_test\ReportTester-82134390.xlsx",
    [string]$ConfigFile = "config.position_management.example.yaml",
    [string]$OutputRoot = "datasets\position_management",
    [ValidateSet("auto", "csv", "parquet")]
    [string]$OutputFormat = "auto",
    [string]$Symbol = "",
    [string]$Timeframe = "",
    [switch]$AllowGlobalPython
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$defaultGlobalPython = "C:\Program Files\Python312\python.exe"
$userProvidedPython = $PythonExe -ne ""

function Test-PythonDependencies {
    param(
        [string]$Exe
    )

    try {
        & $Exe -c "import importlib.util, sys; required = ['numpy', 'pandas', 'yaml', 'openpyxl', 'et_xmlfile']; missing = [name for name in required if importlib.util.find_spec(name) is None]; sys.exit(0 if not missing else 1)" *>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

if ($PythonExe -eq "") {
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = "python"
    }
}

if (-not (Test-PythonDependencies -Exe $PythonExe)) {
    if ($userProvidedPython) {
        throw @"
O interpretador selecionado nao consegue importar as dependencias do pipeline de datasets:
$PythonExe

Opcoes:
- recriar o ambiente: .\setup_venv.ps1 -ForceRecreate
- instalar no .venv atual: .venv\Scripts\python.exe -m pip install -r requirements.txt
"@
    }

    if (Test-Path $venvPython) {
        throw @"
O .venv do projeto existe, mas ainda nao tem as dependencias do pipeline de datasets:
$venvPython

Instale no ambiente do projeto e rode novamente:
- .\setup_venv.ps1
- .venv\Scripts\python.exe -m pip install -r requirements.txt

Se quiser usar um interpretador externo mesmo assim, passe -PythonExe explicitamente.
Para liberar fallback automatico para o Python global quando nao houver .venv valido, use -AllowGlobalPython.
"@
    }

    if (
        $AllowGlobalPython -and
        (Test-Path $defaultGlobalPython) -and
        ($PythonExe -ne $defaultGlobalPython) -and
        (Test-PythonDependencies -Exe $defaultGlobalPython)
    ) {
        Write-Host "Interpretador padrao sem dependencias. Usando Python global: $defaultGlobalPython"
        $PythonExe = $defaultGlobalPython
    }
    else {
        throw @"
O interpretador padrao nao consegue importar as dependencias do pipeline de datasets:
$PythonExe

Opcoes:
- criar o ambiente do projeto: .\setup_venv.ps1
- instalar dependencias no interpretador desejado: python -m pip install -r requirements.txt
- permitir fallback para o Python global: .\run_build_position_management_datasets.ps1 -AllowGlobalPython
"@
    }
}

$arguments = @(
    "src\build_position_management_datasets.py",
    "--candles", $CandlesFile,
    "--report", $ReportFile,
    "--config", $ConfigFile,
    "--output-root", $OutputRoot,
    "--output-format", $OutputFormat
)

if ($Symbol -ne "") {
    $arguments += @("--symbol", $Symbol)
}

if ($Timeframe -ne "") {
    $arguments += @("--timeframe", $Timeframe)
}

Write-Host "Usando interpretador: $PythonExe"
Write-Host "Candles: $CandlesFile"
Write-Host "Report: $ReportFile"
Write-Host "OutputRoot: $OutputRoot"

& $PythonExe @arguments
