param(
    [string]$PythonExe = "",
    [string]$InputFile = "..\..\..\Files\ML_MarketRegime_ONNX\us100_m5_candles.csv",
    [string]$ConfigFile = "config.example.yaml",
    [string]$OutputFile = "reports\walkforward_optimization.csv",
    [string]$Models = "logistic_regression,random_forest",
    [string]$Horizons = "4,8,12,24",
    [string]$Thresholds = "0.8,1.0,1.2,1.5,2.0",
    [string]$TrainWindows = "10000,15000,20000,25000",
    [string]$ValidationWindows = "0",
    [string]$TradeWindows = "1000,2500,5000",
    [string]$StepWindows = "1000,2500",
    [string]$MinClassCountsPerTrain = "50,100,150,200",
    [string]$LogisticCs = "",
    [string]$LogisticMaxIters = "",
    [string]$RfNEstimators = "",
    [string]$RfMaxDepths = "",
    [string]$RfMinSamplesLeaf = "",
    [string]$LookbackBars = "",
    [string]$Epochs = "",
    [string]$BatchSizes = "",
    [string]$LearningRates = "",
    [string]$WeightDecays = "",
    [string]$Dropouts = "",
    [string]$MlpHiddenSizes = "",
    [string]$LstmHiddenSizes = "",
    [string]$LstmNumLayers = "",
    [string]$CnnChannels = "",
    [string]$CnnKernelSizes = "",
    [string]$TransformerDModels = "",
    [string]$TransformerNheads = "",
    [string]$TransformerNumLayers = "",
    [string]$TransformerDimFeedforwards = "",
    [string]$PurgeBars = "auto",
    [int]$MinFolds = 6,
    [int]$MaxFolds = 0,
    [int]$ProgressEvery = 10,
    [int]$TopK = 20,
    [ValidateSet("grid", "random", "ga", "hybrid")]
    [string]$SearchMode = "grid",
    [double]$SampleRatio = 0.10,
    [int]$MaxExperiments = 0,
    [int]$RandomSeed = 42,
    [int]$GaPopulationSize = 12,
    [double]$GaRandomFrac = 0.70,
    [double]$GaEliteFrac = 0.20,
    [double]$GaMutationRate = 0.15,
    [int]$GaPatience = 3,
    [int]$GaTournamentSize = 3,
    [int]$GaMaxGenerations = 0
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

function Count-ListItems {
    param(
        [string]$Raw
    )

    if (-not $Raw) {
        return 0
    }

    return @($Raw.Split(",") | Where-Object { $_.Trim() -ne "" }).Count
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
        throw "Nao foi possivel encontrar um Python com as dependencias do projeto."
    }
}

if ($PSBoundParameters.ContainsKey("InputFile") -and -not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
    throw "Arquivo de entrada informado nao encontrado: $InputFile"
}
$resolvedInput = Resolve-InputFilePath -RequestedPath $InputFile
if (-not $resolvedInput) {
    throw "Arquivo de entrada nao encontrado: $InputFile"
}

$arguments = @(
    "src\sweep_market_regime.py",
    "--input", $resolvedInput,
    "--output", $OutputFile,
    "--evaluation-mode", "walkforward",
    "--search-mode", $SearchMode,
    "--sample-ratio", $SampleRatio,
    "--max-experiments", $MaxExperiments,
    "--random-seed", $RandomSeed,
    "--ga-population-size", $GaPopulationSize,
    "--ga-random-frac", $GaRandomFrac,
    "--ga-elite-frac", $GaEliteFrac,
    "--ga-mutation-rate", $GaMutationRate,
    "--ga-patience", $GaPatience,
    "--ga-tournament-size", $GaTournamentSize,
    "--ga-max-generations", $GaMaxGenerations,
    "--models", $Models,
    "--horizons", $Horizons,
    "--thresholds", $Thresholds,
    "--train-windows", $TrainWindows,
    "--validation-windows", $ValidationWindows,
    "--trade-windows", $TradeWindows,
    "--step-windows", $StepWindows,
    "--min-class-counts-per-train", $MinClassCountsPerTrain,
    "--purge-bars", $PurgeBars,
    "--min-folds", $MinFolds,
    "--max-folds", $MaxFolds,
    "--progress-every", $ProgressEvery,
    "--top-k", $TopK
)

$optionalArguments = @(
    @("--logistic-cs", $LogisticCs),
    @("--logistic-max-iters", $LogisticMaxIters),
    @("--rf-n-estimators", $RfNEstimators),
    @("--rf-max-depths", $RfMaxDepths),
    @("--rf-min-samples-leaf", $RfMinSamplesLeaf),
    @("--lookback-bars", $LookbackBars),
    @("--epochs", $Epochs),
    @("--batch-sizes", $BatchSizes),
    @("--learning-rates", $LearningRates),
    @("--weight-decays", $WeightDecays),
    @("--dropouts", $Dropouts),
    @("--mlp-hidden-sizes", $MlpHiddenSizes),
    @("--lstm-hidden-sizes", $LstmHiddenSizes),
    @("--lstm-num-layers", $LstmNumLayers),
    @("--cnn-channels", $CnnChannels),
    @("--cnn-kernel-sizes", $CnnKernelSizes),
    @("--transformer-d-models", $TransformerDModels),
    @("--transformer-nheads", $TransformerNheads),
    @("--transformer-num-layers", $TransformerNumLayers),
    @("--transformer-dim-feedforwards", $TransformerDimFeedforwards)
)

foreach ($optionalArgument in $optionalArguments) {
    if ($optionalArgument[1] -and $optionalArgument[1].Trim() -ne "") {
        $arguments += $optionalArgument
    }
}

if ($ValidationWindows -eq "") {
    $validationCount = 0
}
else {
    $validationCount = Count-ListItems -Raw $ValidationWindows
}

$combinationCount =
    (Count-ListItems -Raw $Models) *
    (Count-ListItems -Raw $Horizons) *
    (Count-ListItems -Raw $Thresholds) *
    (Count-ListItems -Raw $TrainWindows) *
    [Math]::Max(1, $validationCount) *
    (Count-ListItems -Raw $TradeWindows) *
    (Count-ListItems -Raw $StepWindows) *
    (Count-ListItems -Raw $MinClassCountsPerTrain)

Write-Host "Usando interpretador: $PythonExe"
Write-Host "Otimizando walk-forward com input: $resolvedInput"
Write-Host "Saida principal: $OutputFile"
Write-Host "Combinacoes brutas planejadas: $combinationCount"
if ($SearchMode -eq "grid") {
    Write-Host "Modo de busca: grid"
}
else {
    $estimatedBudget = [Math]::Ceiling($combinationCount * $SampleRatio)
    if ($MaxExperiments -gt 0) {
        $estimatedBudget = [Math]::Min($estimatedBudget, $MaxExperiments)
    }
    Write-Host "Modo de busca: $SearchMode"
    Write-Host "Budget estimado antes dos filtros por familia: $estimatedBudget"
}
if ($optionalArguments.Where({ $_[1] -and $_[1].Trim() -ne "" }).Count -gt 0) {
    Write-Host "Observacao: a contagem acima nao inclui multiplicadores especificos de modelo para grids avancadas."
}
Write-Host "Progresso detalhado: a cada $ProgressEvery experimentos"
if ($MaxFolds -gt 0) {
    Write-Host "Cap de folds por experimento: $MaxFolds"
}
else {
    Write-Host "Cap de folds por experimento: desabilitado"
}

if ($PSBoundParameters.ContainsKey("ConfigFile") -or (Test-Path -LiteralPath $ConfigFile -PathType Leaf)) {
    $arguments += @("--config", $ConfigFile)
}
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "O processo Python terminou com codigo $LASTEXITCODE"
}

$outputPath = Join-Path $projectRoot $OutputFile
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($outputPath)
$extension = [System.IO.Path]::GetExtension($outputPath)
$parentDir = Split-Path $outputPath -Parent

Write-Host "Arquivos gerados:"
Write-Host "- $outputPath"
Write-Host "- $(Join-Path $parentDir ($baseName + '.top_score' + $extension))"
Write-Host "- $(Join-Path $parentDir ($baseName + '.top_mean_balanced_accuracy_trade' + $extension))"
Write-Host "- $(Join-Path $parentDir ($baseName + '.top_mean_accuracy_trade' + $extension))"
Write-Host "- $(Join-Path $parentDir ($baseName + '.best_summary' + $extension))"
