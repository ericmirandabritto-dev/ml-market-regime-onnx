param(
    [string]$PythonExe = "",
    [string]$InputFile = "..\..\..\Files\ML_MarketRegime_ONNX\us100_m5_candles.csv",
    [string]$ArtifactsDir = "artifacts",
    [string]$ConfigFile = "config.example.yaml",
    [ValidateSet("single", "walkforward")]
    [string]$Mode = "single",
    [int]$TrainWindow = 0,
    [int]$ValidationWindow = -1,
    [int]$TradeWindow = 0,
    [int]$StepWindow = 0,
    [string]$PurgeBars = "",
    [int]$MinFolds = 0,
    [int]$MaxFolds = 0,
    [int]$MinClassCountPerTrain = 0,
    [int]$FoldProgressEvery = 1,
    [string]$RuntimeRoot = "",
    [string]$Symbol = "US100",
    [string]$Timeframe = "M5",
    [switch]$IncrementalExport,
    [switch]$ValidateOnnx = $true
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$mql5Root = Resolve-Path (Join-Path $projectRoot "..\..\..")
$terminalDataRoot = Split-Path $mql5Root -Parent
$terminalBaseRoot = Split-Path $terminalDataRoot -Parent
$terminalFilesDir = Join-Path $mql5Root "Files"
$commonFilesDir = Join-Path $terminalBaseRoot "Common\Files"
$defaultGlobalPython = "C:\Program Files\Python312\python.exe"

function Test-PythonDependencies {
    param(
        [string]$Exe
    )

    try {
        & $Exe -c "import numpy, pandas, sklearn, yaml, onnx, skl2onnx, onnxruntime" *>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Resolve-InputFilePath {
    param(
        [string]$RequestedPath
    )

    if ($RequestedPath -and (Test-Path -LiteralPath $RequestedPath)) {
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    $candidates = @(
        $RequestedPath,
        (Join-Path $projectRoot "data\raw\us100_m5_candles.csv"),
        (Join-Path $projectRoot "data\raw\candles.csv"),
        (Join-Path $terminalFilesDir "ML_MarketRegime_ONNX\us100_m5_candles.csv"),
        (Join-Path $terminalFilesDir "us100_m5_candles.csv"),
        (Join-Path $commonFilesDir "ML_MarketRegime_ONNX\us100_m5_candles.csv"),
        (Join-Path $commonFilesDir "us100_m5_candles.csv")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    $candidates = @($candidates)

    if ($candidates.Count -gt 0) {
        return (Resolve-Path -LiteralPath $candidates[0]).Path
    }

    return $null
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
    if ((Test-Path $defaultGlobalPython) -and (Test-PythonDependencies -Exe $defaultGlobalPython)) {
        Write-Host "O interpretador selecionado nao tem as dependencias. Usando Python global: $defaultGlobalPython"
        $PythonExe = $defaultGlobalPython
    }
    else {
        throw @"
O interpretador selecionado nao consegue importar as dependencias do projeto:
$PythonExe

Opcoes:
- recriar o ambiente: .\setup_venv.ps1 -ForceRecreate
- instalar no .venv atual: .venv\Scripts\python.exe -m pip install -r requirements.txt
- usar o Python global explicitamente:
  .\run_train_us100.ps1 -PythonExe "$defaultGlobalPython"
"@
    }
}

if ($PSBoundParameters.ContainsKey("InputFile") -and -not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
    throw "Arquivo de entrada informado nao encontrado: $InputFile"
}
$resolvedInput = Resolve-InputFilePath -RequestedPath $InputFile
if (-not $resolvedInput) {
    throw @"
Arquivo de entrada nao encontrado.

Caminho solicitado:
$InputFile

Locais verificados:
- $terminalFilesDir\ML_MarketRegime_ONNX\us100_m5_candles.csv
- $terminalFilesDir\us100_m5_candles.csv
- $commonFilesDir\ML_MarketRegime_ONNX\us100_m5_candles.csv
- $commonFilesDir\us100_m5_candles.csv
- $projectRoot\data\raw\us100_m5_candles.csv

Exporte o CSV no MT5 ou informe manualmente:
.\run_train_us100.ps1 -InputFile "CAMINHO_DO_ARQUIVO.csv"
"@
}

$arguments = @(
    "src\train_market_regime_onnx.py",
    "--input", $resolvedInput,
    "--artifacts", $ArtifactsDir,
    "--mode", $Mode
)

if ($ValidateOnnx) {
    $arguments += "--validate-onnx"
}

if ($Mode -eq "walkforward") {
    if ($TrainWindow -gt 0) {
        $arguments += @("--train-window", $TrainWindow)
    }
    if ($ValidationWindow -ge 0) {
        $arguments += @("--validation-window", $ValidationWindow)
    }
    if ($TradeWindow -gt 0) {
        $arguments += @("--trade-window", $TradeWindow)
    }
    if ($StepWindow -gt 0) {
        $arguments += @("--step-window", $StepWindow)
    }
    if ($PurgeBars -ne "") {
        $arguments += @("--purge-bars", $PurgeBars)
    }
    if ($MinFolds -gt 0) {
        $arguments += @("--min-folds", $MinFolds)
    }
    if ($MaxFolds -gt 0) {
        $arguments += @("--max-folds", $MaxFolds)
    }
    if ($MinClassCountPerTrain -gt 0) {
        $arguments += @("--min-class-count-per-train", $MinClassCountPerTrain)
    }
    if ($FoldProgressEvery -gt 0) {
        $arguments += @("--fold-progress-every", $FoldProgressEvery)
    }
    if ($RuntimeRoot -ne "") {
        $arguments += @("--runtime-root", $RuntimeRoot)
    }
    if ($Symbol -ne "") {
        $arguments += @("--symbol", $Symbol)
    }
    if ($Timeframe -ne "") {
        $arguments += @("--timeframe", $Timeframe)
    }
    if ($IncrementalExport) {
        $arguments += "--incremental-export"
    }
}

Write-Host "Usando interpretador: $PythonExe"
Write-Host "Modo: $Mode"
Write-Host "Treinando com input: $resolvedInput"

if ($PSBoundParameters.ContainsKey("ConfigFile") -or (Test-Path -LiteralPath $ConfigFile -PathType Leaf)) {
    $arguments += @("--config", $ConfigFile)
}
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "O processo Python terminou com codigo $LASTEXITCODE"
}
