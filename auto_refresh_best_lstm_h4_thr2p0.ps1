param(
    [string]$PythonExe = "",
    [string]$InputFile = "data\raw\us100_m5_candles.csv",
    [string]$ConfigFile = "config.best_lstm_h4_thr2p0.yaml",
    [string]$ArtifactsDir = "artifacts_auto_refresh_best_lstm_h4_thr2p0",
    [string]$DeployRelativeDir = "ML_MarketRegime_ONNX\ea_optimizer_best_models\best_lstm_h4_thr2p0",
    [string]$RequestFile = "ML_MarketRegime_ONNX\ea_optimizer_best_models\best_lstm_h4_thr2p0\refresh_request.csv",
    [string]$StatusFile = "ML_MarketRegime_ONNX\ea_optimizer_best_models\best_lstm_h4_thr2p0\refresh_status.json",
    [string]$Symbol = "US100",
    [string]$Timeframe = "M5",
    [int]$Bars = 50000,
    [string]$TerminalPath = "",
    [int]$PollSeconds = 60,
    [int]$RefreshCooldownMinutes = 30,
    [string]$RegressionBackfillFrom = "2000-01-01",
    [int]$RegressionBackfillCooldownMinutes = 720,
    [int]$FoldProgressEvery = 1,
    [int]$TrainWindow = 0,
    [int]$ValidationWindow = -1,
    [int]$TradeWindow = 0,
    [int]$StepWindow = 0,
    [string]$PurgeBars = "",
    [int]$MinFolds = 0,
    [int]$MaxFolds = 0,
    [int]$MinClassCountPerTrain = 0,
    [switch]$ValidateOnnx,
    [switch]$Watch,
    [switch]$Force,
    [switch]$DisableRegressionBackfill,
    [switch]$SkipDataUpdate,
    [switch]$SkipDeployLive,
    [switch]$SkipDeployCommon
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$baseScript = Join-Path $projectRoot "auto_refresh_walkforward.ps1"

if (-not (Test-Path -LiteralPath $baseScript)) {
    throw "Script base nao encontrado: $baseScript"
}

$invokeParams = @{
    PythonExe              = $PythonExe
    InputFile              = $InputFile
    ConfigFile             = $ConfigFile
    ArtifactsDir           = $ArtifactsDir
    DeployRelativeDir      = $DeployRelativeDir
    RequestFile            = $RequestFile
    StatusFile             = $StatusFile
    Symbol                 = $Symbol
    Timeframe              = $Timeframe
    Bars                   = $Bars
    TerminalPath           = $TerminalPath
    PollSeconds            = $PollSeconds
    RefreshCooldownMinutes = $RefreshCooldownMinutes
    RegressionBackfillFrom = $RegressionBackfillFrom
    RegressionBackfillCooldownMinutes = $RegressionBackfillCooldownMinutes
    FoldProgressEvery      = $FoldProgressEvery
    TrainWindow            = $TrainWindow
    ValidationWindow       = $ValidationWindow
    TradeWindow            = $TradeWindow
    StepWindow             = $StepWindow
    PurgeBars              = $PurgeBars
    MinFolds               = $MinFolds
    MaxFolds               = $MaxFolds
    MinClassCountPerTrain  = $MinClassCountPerTrain
}

if ($ValidateOnnx) {
    $invokeParams.ValidateOnnx = $true
}

if ($Watch) {
    $invokeParams.Watch = $true
}

if ($Force) {
    $invokeParams.Force = $true
}

if ($DisableRegressionBackfill) {
    $invokeParams.DisableRegressionBackfill = $true
}

if ($SkipDataUpdate) {
    $invokeParams.SkipDataUpdate = $true
}

if ($SkipDeployLive) {
    $invokeParams.SkipDeployLive = $true
}

if ($SkipDeployCommon) {
    $invokeParams.SkipDeployCommon = $true
}

& $baseScript @invokeParams
