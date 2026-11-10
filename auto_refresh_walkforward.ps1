param(
    [string]$PythonExe = "",
    [string]$InputFile = "data\raw\us100_m5_candles.csv",
    [string]$ConfigFile = "config.best_rf_h4_thr1p5_short_memory.yaml",
    [string]$ArtifactsDir = "artifacts_auto_refresh",
    [string]$DeployRelativeDir = "ML_MarketRegime_ONNX\walkforward",
    [string]$RequestFile = "ML_MarketRegime_ONNX\walkforward\refresh_request.csv",
    [string]$StatusFile = "ML_MarketRegime_ONNX\walkforward\refresh_status.json",
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
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$mql5Root = Resolve-Path (Join-Path $projectRoot "..\..\..")
$terminalDataRoot = Split-Path $mql5Root -Parent
$terminalBaseRoot = Split-Path $terminalDataRoot -Parent
$terminalFilesDir = Join-Path $mql5Root "Files"
$commonFilesDir = Join-Path $terminalBaseRoot "Common\Files"
$defaultGlobalPython = "C:\Program Files\Python312\python.exe"
$requestPath = Join-Path $commonFilesDir $RequestFile
$statusPath = Join-Path $commonFilesDir $StatusFile
$inputFileExplicitlyProvided = $PSBoundParameters.ContainsKey("InputFile")
$script:InputResolutionReport = @()
$script:TrainingDatasetInfoCache = $null
$script:LastRegressionBackfillSignature = ""
$script:LastRegressionBackfillAt = $null

function Write-StageLog {
    param(
        [string]$Stage,
        [string]$Message
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Stage] $Message"
}

function Format-LogDateTime {
    param(
        $Value
    )

    if ($null -eq $Value) {
        return "n/a"
    }

    if ($Value -is [datetime]) {
        return $Value.ToString("yyyy-MM-dd HH:mm:ss")
    }

    return [string]$Value
}

function Format-Elapsed {
    param(
        [TimeSpan]$Duration
    )

    if ($null -eq $Duration) {
        return "0.000s"
    }

    return ("{0:N3}s" -f $Duration.TotalSeconds)
}

function Get-DurationSeconds {
    param(
        [TimeSpan]$Duration
    )

    if ($null -eq $Duration) {
        return 0
    }

    return [Math]::Round($Duration.TotalSeconds, 3)
}

function Get-DirectoryStats {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{
            Path = $Path
            Exists = $false
            FileCount = 0
            TotalBytes = 0
            LatestWriteTime = $null
        }
    }

    $files = @(Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue)
    $latestWrite = $null
    if ($files.Count -gt 0) {
        $latestWrite = ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
    }

    $totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $totalBytes) {
        $totalBytes = 0
    }

    return [pscustomobject]@{
        Path = $Path
        Exists = $true
        FileCount = $files.Count
        TotalBytes = $totalBytes
        LatestWriteTime = $latestWrite
    }
}

function Format-DirectoryStats {
    param(
        $Stats
    )

    if (-not $Stats) {
        return "files=n/a | bytes=n/a | latest_write=n/a"
    }

    return "files=$($Stats.FileCount) | bytes=$($Stats.TotalBytes) | latest_write=$(Format-LogDateTime $Stats.LatestWriteTime)"
}

function Get-RemainingRowsToRefresh {
    param(
        $CsvInfo,
        $RuntimeManifest
    )

    if (-not $CsvInfo -or -not $RuntimeManifest -or $RuntimeManifest.NextRefreshRowCount -le 0) {
        return $null
    }

    return [Math]::Max(0, ([int]$RuntimeManifest.NextRefreshRowCount - [int]$CsvInfo.RowCount))
}

function Get-InputRegressionInfo {
    param(
        $CsvInfo,
        $RuntimeManifest
    )

    if (-not $CsvInfo -or -not $RuntimeManifest) {
        return $null
    }

    $rowsBehind = $null
    if ($RuntimeManifest.LatestDatasetRows -gt 0 -and $CsvInfo.RowCount -lt $RuntimeManifest.LatestDatasetRows) {
        $rowsBehind = ([int]$RuntimeManifest.LatestDatasetRows - [int]$CsvInfo.RowCount)
    }

    $timeBehindSeconds = $null
    if ($RuntimeManifest.LatestDatasetTime -and $CsvInfo.LatestTime -and $CsvInfo.LatestTime -lt $RuntimeManifest.LatestDatasetTime) {
        $timeBehindSeconds = [Math]::Round(($RuntimeManifest.LatestDatasetTime - $CsvInfo.LatestTime).TotalSeconds, 0)
    }

    if ($null -eq $rowsBehind -and $null -eq $timeBehindSeconds) {
        return $null
    }

    return [pscustomobject]@{
        InputRows = [int]$CsvInfo.RowCount
        ManifestRows = [int]$RuntimeManifest.LatestDatasetRows
        RowsBehind = if ($null -ne $rowsBehind) { [int]$rowsBehind } else { 0 }
        InputLatestTime = $CsvInfo.LatestTime
        ManifestLatestTime = $RuntimeManifest.LatestDatasetTime
        TimeBehindSeconds = if ($null -ne $timeBehindSeconds) { [int]$timeBehindSeconds } else { 0 }
    }
}

function Parse-OptionalDateTime {
    param(
        [string]$Value
    )

    if (-not $Value) {
        return $null
    }

    $formats = @(
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "yyyy.MM.dd HH:mm:ss",
        "yyyy.MM.dd"
    )

    foreach ($format in $formats) {
        $parsed = Parse-TrainingDatetime -Value $Value -Format $format
        if ($parsed) {
            return $parsed
        }
    }

    return $null
}

function Get-RegressionBackfillPlan {
    param(
        $CsvInfo,
        $RuntimeManifest,
        $Regression
    )

    if ($DisableRegressionBackfill -or $SkipDataUpdate -or -not $CsvInfo -or -not $RuntimeManifest -or -not $Regression) {
        return $null
    }

    $dateFrom = Parse-OptionalDateTime -Value $RegressionBackfillFrom
    if (-not $dateFrom) {
        throw "RegressionBackfillFrom invalido: $RegressionBackfillFrom"
    }

    $dateTo = $null
    if ($CsvInfo.LatestTime) {
        $dateTo = $CsvInfo.LatestTime
    }
    elseif ($RuntimeManifest.LatestDatasetTime) {
        $dateTo = $RuntimeManifest.LatestDatasetTime
    }

    $signature = @(
        [string]$CsvInfo.RowCount,
        (Format-LogDateTime $CsvInfo.EarliestTime),
        (Format-LogDateTime $CsvInfo.LatestTime),
        [string]$Regression.RowsBehind,
        [string]$Regression.TimeBehindSeconds,
        (Format-LogDateTime $dateFrom),
        (Format-LogDateTime $dateTo)
    ) -join "|"

    return [pscustomobject]@{
        DateFrom = $dateFrom
        DateTo = $dateTo
        Signature = $signature
    }
}

function Ensure-InputHistorySynchronized {
    param(
        $CsvInfo,
        $DatasetInfo,
        $RuntimeManifest,
        [string]$Context
    )

    $currentDatasetInfo = if ($DatasetInfo) { $DatasetInfo } else { $CsvInfo }
    $initialRegression = Get-InputRegressionInfo -CsvInfo $currentDatasetInfo -RuntimeManifest $RuntimeManifest
    if (-not $initialRegression) {
        return [pscustomobject]@{
            Attempted = $false
            Recovered = $true
            CsvInfo = $CsvInfo
            DatasetInfo = $DatasetInfo
            InputRegression = $null
        }
    }

    $backfillPlan = Get-RegressionBackfillPlan -CsvInfo $CsvInfo -RuntimeManifest $RuntimeManifest -Regression $initialRegression
    if (-not $backfillPlan) {
        return [pscustomobject]@{
            Attempted = $false
            Recovered = $false
            CsvInfo = $CsvInfo
            DatasetInfo = $DatasetInfo
            InputRegression = $initialRegression
        }
    }

    if (
        $script:LastRegressionBackfillAt -and
        $script:LastRegressionBackfillSignature -eq $backfillPlan.Signature -and
        (((Get-Date) - $script:LastRegressionBackfillAt).TotalMinutes -lt $RegressionBackfillCooldownMinutes)
    ) {
        Write-StageLog "backfill" "Backfill automatico em cooldown | context=$Context | next_retry_after_minutes=$RegressionBackfillCooldownMinutes | signature=$($backfillPlan.Signature)"
        return [pscustomobject]@{
            Attempted = $false
            Recovered = $false
            CsvInfo = $CsvInfo
            DatasetInfo = $DatasetInfo
            InputRegression = $initialRegression
        }
    }

    Write-StageLog "backfill" "Tentando recompor historico de entrada | context=$Context | rows_behind=$($initialRegression.RowsBehind) | time_behind_seconds=$($initialRegression.TimeBehindSeconds) | csv_earliest=$(Format-LogDateTime $CsvInfo.EarliestTime) | csv_latest=$(Format-LogDateTime $CsvInfo.LatestTime) | date_from=$(Format-LogDateTime $backfillPlan.DateFrom) | date_to=$(Format-LogDateTime $backfillPlan.DateTo)"
    $script:LastRegressionBackfillSignature = $backfillPlan.Signature
    $script:LastRegressionBackfillAt = Get-Date

    $null = Invoke-DataUpdate -DateFrom ($backfillPlan.DateFrom.ToString("yyyy-MM-dd HH:mm:ss")) -DateTo $(if ($backfillPlan.DateTo) { $backfillPlan.DateTo.ToString("yyyy-MM-dd HH:mm:ss") } else { "" }) -Stage "backfill" -Operation "Repondo historico do CSV via MetaTrader 5"

    $csvInfoAfter = Get-CsvInfo -Path $resolvedInput
    $datasetInfoAfter = Get-TrainingDatasetInfo -InputPath $resolvedInput -CsvInfo $csvInfoAfter
    $finalDatasetInfo = if ($datasetInfoAfter) { $datasetInfoAfter } else { $csvInfoAfter }
    $finalRegression = Get-InputRegressionInfo -CsvInfo $finalDatasetInfo -RuntimeManifest $RuntimeManifest

    Write-StageLog "backfill" "Backfill automatico concluido | context=$Context | csv_rows=$(if ($csvInfoAfter) { $csvInfoAfter.RowCount } else { 'n/a' }) | csv_earliest=$(Format-LogDateTime $(if ($csvInfoAfter) { $csvInfoAfter.EarliestTime } else { $null })) | csv_latest=$(Format-LogDateTime $(if ($csvInfoAfter) { $csvInfoAfter.LatestTime } else { $null })) | dataset_rows=$(if ($datasetInfoAfter) { $datasetInfoAfter.RowCount } else { 'n/a' }) | dataset_latest=$(Format-LogDateTime $(if ($datasetInfoAfter) { $datasetInfoAfter.LatestTime } else { $null })) | rows_behind=$(if ($finalRegression) { $finalRegression.RowsBehind } else { 0 })"

    return [pscustomobject]@{
        Attempted = $true
        Recovered = (-not $finalRegression)
        CsvInfo = $csvInfoAfter
        DatasetInfo = $datasetInfoAfter
        InputRegression = $finalRegression
    }
}

function Get-PythonDependencyStatus {
    param(
        [string]$Exe
    )

    if (-not $Exe -or -not (Test-Path $Exe)) {
        return [pscustomobject]@{
            Ok = $false
            Error = "python_not_found"
        }
    }

    $requiredModules = @("numpy", "pandas", "sklearn", "onnx", "skl2onnx", "onnxruntime")
    if (-not $SkipDataUpdate) {
        $requiredModules += "MetaTrader5"
    }

    $errors = New-Object System.Collections.Generic.List[string]
    foreach ($moduleName in $requiredModules) {
        try {
            $output = & $Exe -c "import $moduleName" 2>&1
            if ($LASTEXITCODE -ne 0) {
                $errorText = (($output | Out-String).Trim())
                if (-not $errorText) {
                    $errorText = "import_failed"
                }
                $errors.Add("${moduleName}: $errorText")
            }
        }
        catch {
            $errors.Add("${moduleName}: $($_.Exception.Message)")
        }
    }

    return [pscustomobject]@{
        Ok = ($errors.Count -eq 0)
        Error = ($errors -join " | ")
    }
}

function Resolve-InputFilePath {
    param(
        [string]$RequestedPath
    )

    $rawCandidates = New-Object System.Collections.Generic.List[string]
    if ($RequestedPath) {
        $rawCandidates.Add($RequestedPath)
        if (-not [System.IO.Path]::IsPathRooted($RequestedPath)) {
            $rawCandidates.Add((Join-Path $projectRoot $RequestedPath))
        }
    }

    $rawCandidates.Add((Join-Path $terminalFilesDir "ML_MarketRegime_ONNX\us100_m5_candles.csv"))
    $rawCandidates.Add((Join-Path $terminalFilesDir "us100_m5_candles.csv"))
    $rawCandidates.Add((Join-Path $commonFilesDir "ML_MarketRegime_ONNX\us100_m5_candles.csv"))
    $rawCandidates.Add((Join-Path $commonFilesDir "us100_m5_candles.csv"))
    $rawCandidates.Add((Join-Path $projectRoot "data\raw\us100_m5_candles.csv"))
    $rawCandidates.Add((Join-Path $projectRoot "data\raw\candles.csv"))

    $candidatePaths = $rawCandidates | Where-Object { $_ } | Select-Object -Unique
    $existingCandidates = New-Object System.Collections.Generic.List[object]

    foreach ($candidatePath in $candidatePaths) {
        if (-not (Test-Path -LiteralPath $candidatePath)) {
            continue
        }

        $resolvedPath = (Resolve-Path -LiteralPath $candidatePath).Path
        $csvInfo = Get-CsvInfo -Path $resolvedPath
        $existingCandidates.Add([pscustomobject]@{
            Path = $resolvedPath
            RowCount = if ($csvInfo) { $csvInfo.RowCount } else { 0 }
            LatestTime = if ($csvInfo) { $csvInfo.LatestTime } else { $null }
        })
    }

    $selectedPath = $null
    if ($inputFileExplicitlyProvided -and $RequestedPath -and (Test-Path -LiteralPath $RequestedPath)) {
        $selectedPath = (Resolve-Path -LiteralPath $RequestedPath).Path
    }
    elseif ($existingCandidates.Count -gt 0) {
        $selectedPath = ($existingCandidates |
            Sort-Object `
                @{ Expression = { [int]$_.RowCount }; Descending = $true }, `
                @{ Expression = { if ($_.LatestTime) { $_.LatestTime } else { [datetime]::MinValue } }; Descending = $true }, `
                @{ Expression = { $_.Path }; Descending = $false } |
            Select-Object -First 1).Path
    }

    if (-not $selectedPath) {
        $selectedPath = if ([System.IO.Path]::IsPathRooted($RequestedPath)) { $RequestedPath } else { (Join-Path $projectRoot $RequestedPath) }
    }

    $script:InputResolutionReport = @(
        $existingCandidates | ForEach-Object {
            [pscustomobject]@{
                Path = $_.Path
                RowCount = $_.RowCount
                LatestTime = $_.LatestTime
                Selected = ($_.Path -eq $selectedPath)
            }
        }
    )

    return $selectedPath
}

function Parse-TrainingDatetime {
    param(
        [string]$Value,
        [string]$Format
    )

    if (-not $Value) {
        return $null
    }

    $text = $Value.Trim().Trim('"')
    $culture = [System.Globalization.CultureInfo]::InvariantCulture

    try {
        return [datetime]::ParseExact($text, $Format, $culture)
    }
    catch {
        $parsed = [datetime]::MinValue
        if ([datetime]::TryParse($text, $culture, [System.Globalization.DateTimeStyles]::None, [ref]$parsed)) {
            return $parsed
        }

        return $null
    }
}

function Get-CsvInfo {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $lastLine = Get-Content -LiteralPath $Path -Tail 1
    if (-not $lastLine -or $lastLine -eq "time,open,high,low,close,volume") {
        return [pscustomobject]@{
            Path = $Path
            RowCount = 0
            EarliestTime = $null
            LatestTime = $null
        }
    }

    $firstDataLine = Get-Content -LiteralPath $Path -TotalCount 2 | Select-Object -Last 1
    $firstTimeText = if ($firstDataLine) { ($firstDataLine -split ",")[0].Trim('"') } else { $null }
    $lastTimeText = ($lastLine -split ",")[0].Trim('"')
    $lineCount = (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
    return [pscustomobject]@{
        Path = $Path
        RowCount = [Math]::Max(0, [int]$lineCount - 1)
        EarliestTime = (Parse-TrainingDatetime -Value $firstTimeText -Format "yyyy-MM-dd HH:mm:ss")
        LatestTime = (Parse-TrainingDatetime -Value $lastTimeText -Format "yyyy-MM-dd HH:mm:ss")
    }
}

function Get-TrainingDatasetInfo {
    param(
        [string]$InputPath,
        $CsvInfo
    )

    if (-not $InputPath) {
        return $null
    }

    if (-not $CsvInfo) {
        $CsvInfo = Get-CsvInfo -Path $InputPath
    }

    if (-not $CsvInfo) {
        return $null
    }

    if (
        $script:TrainingDatasetInfoCache -and
        $script:TrainingDatasetInfoCache.Path -eq $InputPath -and
        $script:TrainingDatasetInfoCache.RawRowCount -eq $CsvInfo.RowCount -and
        (Format-LogDateTime $script:TrainingDatasetInfoCache.RawLatestTime) -eq (Format-LogDateTime $CsvInfo.LatestTime) -and
        $script:TrainingDatasetInfoCache.ConfigPath -eq $resolvedConfigPath
    ) {
        return $script:TrainingDatasetInfoCache
    }

    $pythonSnippet = @"
import json
import sys
from pathlib import Path

sys.path.insert(0, r"$($projectRoot.Replace('\', '\\'))\src")

from market_regime_core import load_config, load_candles_csv, build_feature_frame

cfg = load_config(Path(r"$($resolvedConfigPath.Replace('\', '\\'))"))
candles = load_candles_csv(Path(r"$($InputPath.Replace('\', '\\'))"))
dataset, _ = build_feature_frame(candles, cfg)

payload = {
    "row_count": int(len(dataset)),
    "latest_time": None if dataset.empty else str(dataset.index.max()),
}
print(json.dumps(payload))
"@

    $output = $pythonSnippet | & $PythonExe -
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao calcular dataset efetivo para refresh."
    }

    $jsonLine = ($output | Where-Object { $_ } | Select-Object -Last 1)
    if (-not $jsonLine) {
        throw "Calculo do dataset efetivo nao retornou payload JSON."
    }

    $payload = $jsonLine | ConvertFrom-Json
    $datasetInfo = [pscustomobject]@{
        Path = $InputPath
        ConfigPath = $resolvedConfigPath
        RowCount = [int]$payload.row_count
        LatestTime = Parse-TrainingDatetime -Value $payload.latest_time -Format "yyyy-MM-dd HH:mm:ss"
        RawRowCount = [int]$CsvInfo.RowCount
        RawLatestTime = $CsvInfo.LatestTime
    }

    $script:TrainingDatasetInfoCache = $datasetInfo
    return $datasetInfo
}

function Get-LatestCsvTime {
    param(
        [string]$Path
    )

    $csvInfo = Get-CsvInfo -Path $Path
    if (-not $csvInfo) {
        return $null
    }

    return $csvInfo.LatestTime
}

function Get-ManifestInfo {
    param(
        [string]$ManifestPath
    )

    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        return $null
    }

    $rows = Import-Csv -LiteralPath $ManifestPath
    if (-not $rows -or $rows.Count -eq 0) {
        return $null
    }

    $first = $rows[0]
    $lastClosed = $rows | Where-Object { $_.active_to } | Select-Object -Last 1
    $openCount = @($rows | Where-Object { $_.status -eq "open" }).Count

    return [pscustomobject]@{
        Path = $ManifestPath
        Folds = $rows.Count
        FirstActiveFrom = Parse-TrainingDatetime -Value $first.active_from -Format "yyyy.MM.dd HH:mm:ss"
        LastActiveTo = if ($lastClosed) { Parse-TrainingDatetime -Value $lastClosed.active_to -Format "yyyy.MM.dd HH:mm:ss" } else { $null }
        OpenFolds = $openCount
        ClosedFolds = ($rows.Count - $openCount)
    }
}

function Get-RuntimeManifestInfo {
    param(
        [string]$ManifestJsonPath,
        [string]$ManifestCsvPath
    )

    $csvInfo = Get-ManifestInfo -ManifestPath $ManifestCsvPath
    if (-not (Test-Path -LiteralPath $ManifestJsonPath)) {
        return $csvInfo
    }

    $payload = Get-Content -LiteralPath $ManifestJsonPath -Raw | ConvertFrom-Json
    return [pscustomobject]@{
        Path = $ManifestJsonPath
        CsvPath = $ManifestCsvPath
        Folds = if ($csvInfo) { $csvInfo.Folds } else { 0 }
        FirstActiveFrom = if ($csvInfo) { $csvInfo.FirstActiveFrom } else { $null }
        LastActiveTo = if ($csvInfo) { $csvInfo.LastActiveTo } else { $null }
        OpenFolds = if ($csvInfo) { $csvInfo.OpenFolds } else { 0 }
        ClosedFolds = if ($csvInfo) { $csvInfo.ClosedFolds } else { 0 }
        LatestDatasetRows = if ($null -ne $payload.latest_dataset_rows) { [int]$payload.latest_dataset_rows } else { 0 }
        LatestDatasetTime = Parse-TrainingDatetime -Value $payload.latest_dataset_time -Format "yyyy-MM-dd HH:mm:ss"
        RuntimeReadyFolds = if ($null -ne $payload.runtime_ready_folds) { [int]$payload.runtime_ready_folds } else { 0 }
        RuntimeStagedFolds = if ($null -ne $payload.runtime_staged_folds) { [int]$payload.runtime_staged_folds } else { 0 }
        RuntimeOpenFolds = if ($null -ne $payload.runtime_open_folds) { [int]$payload.runtime_open_folds } else { 0 }
        RuntimeClosedFolds = if ($null -ne $payload.runtime_closed_folds) { [int]$payload.runtime_closed_folds } else { 0 }
        NextRefreshReason = $payload.next_refresh_reason
        NextRefreshStage = $payload.next_refresh_stage
        NextRefreshRowCount = if ($null -ne $payload.next_refresh_row_count) { [int]$payload.next_refresh_row_count } else { 0 }
        NextFoldId = $payload.next_fold_id
        NextFoldTrainReadyAt = Parse-TrainingDatetime -Value $payload.next_fold_train_ready_at -Format "yyyy.MM.dd HH:mm:ss"
        NextFoldActiveFrom = Parse-TrainingDatetime -Value $payload.next_fold_active_from -Format "yyyy.MM.dd HH:mm:ss"
        NextFoldActiveTo = Parse-TrainingDatetime -Value $payload.next_fold_active_to -Format "yyyy.MM.dd HH:mm:ss"
        NextFoldTrainReadyRowCount = if ($null -ne $payload.next_fold_train_ready_row_count) { [int]$payload.next_fold_train_ready_row_count } else { 0 }
        NextFoldActiveRowCount = if ($null -ne $payload.next_fold_active_row_count) { [int]$payload.next_fold_active_row_count } else { 0 }
        NextFoldCloseRowCount = if ($null -ne $payload.next_fold_close_row_count) { [int]$payload.next_fold_close_row_count } else { 0 }
    }
}

function Get-RuntimeStatusExtra {
    param(
        $CsvInfo,
        $RuntimeManifest,
        $DatasetInfo = $null
    )

    $extra = @{}
    if ($CsvInfo) {
        $extra["latest_csv_rows"] = $CsvInfo.RowCount
        $extra["latest_csv_time"] = if ($CsvInfo.LatestTime) { $CsvInfo.LatestTime.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
    }
    if ($DatasetInfo) {
        $extra["effective_dataset_rows"] = $DatasetInfo.RowCount
        $extra["effective_dataset_time"] = if ($DatasetInfo.LatestTime) { $DatasetInfo.LatestTime.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
    }
    if ($RuntimeManifest) {
        $extra["manifest_folds"] = $RuntimeManifest.Folds
        $extra["manifest_last_active_to"] = if ($RuntimeManifest.LastActiveTo) { $RuntimeManifest.LastActiveTo.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
        $extra["runtime_ready_folds"] = $RuntimeManifest.RuntimeReadyFolds
        $extra["runtime_closed_folds"] = $RuntimeManifest.RuntimeClosedFolds
        $extra["runtime_open_folds"] = $RuntimeManifest.RuntimeOpenFolds
        $extra["runtime_staged_folds"] = $RuntimeManifest.RuntimeStagedFolds
        $extra["next_refresh_reason"] = $RuntimeManifest.NextRefreshReason
        $extra["next_refresh_stage"] = $RuntimeManifest.NextRefreshStage
        $extra["next_refresh_row_count"] = $RuntimeManifest.NextRefreshRowCount
        $extra["manifest_latest_dataset_rows"] = $RuntimeManifest.LatestDatasetRows
        $extra["manifest_latest_dataset_time"] = if ($RuntimeManifest.LatestDatasetTime) { $RuntimeManifest.LatestDatasetTime.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
        $extra["next_fold_id"] = $RuntimeManifest.NextFoldId
        $extra["next_fold_train_ready_at"] = if ($RuntimeManifest.NextFoldTrainReadyAt) { $RuntimeManifest.NextFoldTrainReadyAt.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
        $extra["next_fold_active_from"] = if ($RuntimeManifest.NextFoldActiveFrom) { $RuntimeManifest.NextFoldActiveFrom.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
        $extra["next_fold_active_to"] = if ($RuntimeManifest.NextFoldActiveTo) { $RuntimeManifest.NextFoldActiveTo.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
        $extra["next_fold_train_ready_row_count"] = $RuntimeManifest.NextFoldTrainReadyRowCount
        $extra["next_fold_active_row_count"] = $RuntimeManifest.NextFoldActiveRowCount
        $extra["next_fold_close_row_count"] = $RuntimeManifest.NextFoldCloseRowCount

        $remainingRows = Get-RemainingRowsToRefresh -CsvInfo $(if ($DatasetInfo) { $DatasetInfo } else { $CsvInfo }) -RuntimeManifest $RuntimeManifest
        if ($null -ne $remainingRows) {
            $extra["effective_rows_remaining_to_refresh"] = $remainingRows
        }

        $regression = Get-InputRegressionInfo -CsvInfo $(if ($DatasetInfo) { $DatasetInfo } else { $CsvInfo }) -RuntimeManifest $RuntimeManifest
        if ($regression) {
            $extra["input_rows_behind_manifest"] = $regression.RowsBehind
            $extra["input_latest_behind_manifest_seconds"] = $regression.TimeBehindSeconds
        }
    }

    return $extra
}

function Test-LiveTailAvailable {
    param(
        [string]$PackageRoot
    )

    $liveModel = Join-Path $PackageRoot "live\model.onnx"
    $liveInfo = Join-Path $PackageRoot "live\live_tail_info.json"
    return ((Test-Path -LiteralPath $liveModel) -and (Test-Path -LiteralPath $liveInfo))
}

function Get-RefreshRequest {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $rows = Import-Csv -LiteralPath $Path
    if (-not $rows -or $rows.Count -eq 0) {
        return $null
    }

    $row = $rows[0]
    return [pscustomobject]@{
        Path = $Path
        RequestedAt = Parse-TrainingDatetime -Value $row.requested_at -Format "yyyy.MM.dd HH:mm:ss"
        Reason = $row.reason
        Symbol = $row.symbol
        Timeframe = $row.timeframe
        SignalBarTime = Parse-TrainingDatetime -Value $row.signal_bar_time -Format "yyyy.MM.dd HH:mm:ss"
        ManifestFile = $row.manifest_file
    }
}

function Get-ReferenceRuntimeManifest {
    $commonManifestPath = Join-Path $commonFilesDir (Join-Path $DeployRelativeDir "manifest.csv")
    $commonManifestJsonPath = Join-Path $commonFilesDir (Join-Path $DeployRelativeDir "manifest.json")
    $liveManifestPath = Join-Path $terminalFilesDir (Join-Path $DeployRelativeDir "manifest.csv")
    $liveManifestJsonPath = Join-Path $terminalFilesDir (Join-Path $DeployRelativeDir "manifest.json")

    $referenceManifest = Get-RuntimeManifestInfo -ManifestJsonPath $commonManifestJsonPath -ManifestCsvPath $commonManifestPath
    if (-not $referenceManifest) {
        $referenceManifest = Get-RuntimeManifestInfo -ManifestJsonPath $liveManifestJsonPath -ManifestCsvPath $liveManifestPath
    }

    return $referenceManifest
}

function Write-StatusFile {
    param(
        [string]$State,
        [string]$Reason,
        [hashtable]$Extra = @{}
    )

    $payload = @{
        updated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        state = $State
        reason = $Reason
    }

    foreach ($key in $Extra.Keys) {
        $payload[$key] = $Extra[$key]
    }

    try {
        $targetDir = Split-Path -Parent $statusPath
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        ($payload | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $statusPath -Encoding UTF8
    }
    catch {
        Write-StageLog "status" "Falha ao gravar status em $statusPath | $_"
    }
}

function Get-RefreshDecision {
    param(
        [switch]$IgnoreForce
    )

    $request = Get-RefreshRequest -Path $requestPath
    $referenceManifest = Get-ReferenceRuntimeManifest
    $csvInfo = Get-CsvInfo -Path $resolvedInput
    $datasetInfo = Get-TrainingDatasetInfo -InputPath $resolvedInput -CsvInfo $csvInfo
    $latestCsvTime = if ($csvInfo) { $csvInfo.LatestTime } else { $null }

    $reasons = New-Object System.Collections.Generic.List[string]

    if ($Force -and -not $IgnoreForce) {
        $reasons.Add("force")
    }
    if ($request) {
        $reasons.Add("ea_request:$($request.Reason)")
    }
    if (-not $referenceManifest) {
        $reasons.Add("manifest_missing")
    }
    elseif ($datasetInfo -and $referenceManifest.NextRefreshRowCount -gt 0 -and $datasetInfo.RowCount -ge $referenceManifest.NextRefreshRowCount) {
        $refreshReason = $referenceManifest.NextRefreshReason
        if (-not $refreshReason) {
            $refreshReason = "runtime_fold_boundary"
        }
        $reasons.Add($refreshReason)
    }

    $remainingRows = Get-RemainingRowsToRefresh -CsvInfo $datasetInfo -RuntimeManifest $referenceManifest

    return [pscustomobject]@{
        NeedRefresh = ($reasons.Count -gt 0)
        Reasons = $reasons
        LatestCsvTime = $latestCsvTime
        CsvInfo = $csvInfo
        DatasetInfo = $datasetInfo
        ManifestInfo = $referenceManifest
        Request = $request
        RemainingRowsToRefresh = $remainingRows
    }
}

function Invoke-DataUpdate {
    param(
        [switch]$Quiet,
        [string]$DateFrom = "",
        [string]$DateTo = "",
        [string]$Stage = "data",
        [string]$Operation = "Atualizando base de candles via MetaTrader 5"
    )

    $before = Get-CsvInfo -Path $resolvedInput
    $stageStart = Get-Date
    $updateMode = if ($DateFrom -or $DateTo) { "range" } else { "bars" }
    if (-not $Quiet) {
        Write-StageLog $Stage "${Operation} | symbol=$Symbol | timeframe=$Timeframe | mode=$updateMode | bars=$(if ($updateMode -eq 'bars') { $Bars } else { 'n/a' }) | date_from=$(if ($DateFrom) { $DateFrom } else { 'n/a' }) | date_to=$(if ($DateTo) { $DateTo } else { 'n/a' }) | csv_before_rows=$(if ($before) { $before.RowCount } else { 'n/a' }) | csv_before_earliest=$(Format-LogDateTime $(if ($before) { $before.EarliestTime } else { $null })) | csv_before_latest=$(Format-LogDateTime $(if ($before) { $before.LatestTime } else { $null }))"
    }
    $arguments = @(
        "src\extract_mt5_candles.py",
        "--symbol", $Symbol,
        "--timeframe", $Timeframe,
        "--output", $resolvedInput,
        "--append"
    )

    if ($DateFrom -or $DateTo) {
        if (-not $DateFrom) {
            throw "DateFrom e obrigatorio para atualizacao por periodo."
        }

        $arguments += @("--date-from", $DateFrom)
        if ($DateTo) {
            $arguments += @("--date-to", $DateTo)
        }
    }
    else {
        $arguments += @("--bars", $Bars)
    }

    if ($TerminalPath -ne "") {
        $arguments += @("--terminal-path", $TerminalPath)
    }

    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao atualizar a base de dados."
    }

    $after = Get-CsvInfo -Path $resolvedInput
    $elapsed = (Get-Date) - $stageStart
    $rowsBefore = if ($before) { [int]$before.RowCount } else { 0 }
    $rowsAfter = if ($after) { [int]$after.RowCount } else { 0 }
    $rowsDelta = $rowsAfter - $rowsBefore
    $changed = ($rowsDelta -ne 0)
    if (-not $changed -and $before -and $after) {
        $changed = ($before.LatestTime -ne $after.LatestTime)
    }

    if (-not $Quiet -or $changed) {
        Write-StageLog $Stage "Atualizacao de candles concluida | mode=$updateMode | rows_before=$rowsBefore | rows_after=$rowsAfter | rows_delta=$rowsDelta | earliest_after=$(Format-LogDateTime $(if ($after) { $after.EarliestTime } else { $null })) | latest_after=$(Format-LogDateTime $(if ($after) { $after.LatestTime } else { $null })) | elapsed=$(Format-Elapsed $elapsed)"
    }

    return [pscustomobject]@{
        Before = $before
        After = $after
        Elapsed = $elapsed
        RowsDelta = $rowsDelta
        Changed = $changed
    }
}

function Update-WatchInputData {
    if ($SkipDataUpdate) {
        return
    }

    $updateResult = Invoke-DataUpdate -Quiet
    $before = if ($updateResult) { $updateResult.Before } else { $null }
    $after = if ($updateResult) { $updateResult.After } else { $null }

    if (-not $after) {
        return
    }

    $changed = $false
    if (-not $before) {
        $changed = $true
    }
    elseif ($before.RowCount -ne $after.RowCount) {
        $changed = $true
    }
    elseif ($before.LatestTime -ne $after.LatestTime) {
        $changed = $true
    }

    if ($changed) {
        $latestText = if ($after.LatestTime) { $after.LatestTime.ToString("yyyy-MM-dd HH:mm:ss") } else { "n/a" }
        $rowsDelta = if ($updateResult) { $updateResult.RowsDelta } else { ([int]$after.RowCount - [int]$before.RowCount) }
        $elapsed = if ($updateResult) { Format-Elapsed $updateResult.Elapsed } else { "n/a" }
        Write-StageLog "watch" "CSV atualizado durante watch | rows=$($after.RowCount) | rows_delta=$rowsDelta | latest=$latestText | elapsed=$elapsed"
    }
}

function Invoke-Training {
    $stageStart = Get-Date
    $artifactBefore = Get-DirectoryStats -Path $ArtifactsDir
    $inputInfo = Get-CsvInfo -Path $resolvedInput
    $datasetInfo = Get-TrainingDatasetInfo -InputPath $resolvedInput -CsvInfo $inputInfo
    Write-StageLog "train" "Gerando pacote walk-forward | artifacts=$ArtifactsDir | config=$ConfigFile | input_rows=$(if ($inputInfo) { $inputInfo.RowCount } else { 'n/a' }) | input_latest=$(Format-LogDateTime $(if ($inputInfo) { $inputInfo.LatestTime } else { $null })) | dataset_rows=$(if ($datasetInfo) { $datasetInfo.RowCount } else { 'n/a' }) | dataset_latest=$(Format-LogDateTime $(if ($datasetInfo) { $datasetInfo.LatestTime } else { $null })) | artifacts_before=$(Format-DirectoryStats $artifactBefore)"
    $arguments = @(
        "src\train_market_regime_onnx.py",
        "--input", $resolvedInput,
        "--artifacts", $ArtifactsDir,
        "--config", $ConfigFile,
        "--mode", "walkforward",
        "--symbol", $Symbol,
        "--timeframe", $Timeframe,
        "--fold-progress-every", $FoldProgressEvery,
        "--incremental-export"
    )

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
    if ($ValidateOnnx) {
        $arguments += "--validate-onnx"
    }

    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao treinar/exportar o pacote walk-forward."
    }

    $artifactAfter = Get-DirectoryStats -Path $ArtifactsDir
    $manifestPath = Join-Path $ArtifactsDir "manifest.csv"
    $manifestJsonPath = Join-Path $ArtifactsDir "manifest.json"
    $manifestInfo = Get-RuntimeManifestInfo -ManifestJsonPath $manifestJsonPath -ManifestCsvPath $manifestPath
    $elapsed = (Get-Date) - $stageStart
    Write-StageLog "train" "Pacote walk-forward gerado | elapsed=$(Format-Elapsed $elapsed) | artifacts_after=$(Format-DirectoryStats $artifactAfter) | folds=$(if ($manifestInfo) { $manifestInfo.Folds } else { 'n/a' }) | open_folds=$(if ($manifestInfo) { $manifestInfo.OpenFolds } else { 'n/a' }) | next_fold=$(if ($manifestInfo) { $manifestInfo.NextFoldId } else { 'n/a' }) | next_stage=$(if ($manifestInfo) { $manifestInfo.NextRefreshStage } else { 'n/a' })"

    return [pscustomobject]@{
        Elapsed = $elapsed
        ArtifactBefore = $artifactBefore
        ArtifactAfter = $artifactAfter
        ManifestInfo = $manifestInfo
        DatasetInfo = $datasetInfo
    }
}

function Seed-ArtifactCacheFromDeploy {
    $artifactManifest = Join-Path $ArtifactsDir "manifest.csv"
    if (Test-Path -LiteralPath $artifactManifest) {
        $artifactStats = Get-DirectoryStats -Path $ArtifactsDir
        Write-StageLog "cache" "Cache local ja disponivel | path=$ArtifactsDir | $(Format-DirectoryStats $artifactStats)"
        return
    }

    $candidateSources = @(
        (Join-Path $commonFilesDir $DeployRelativeDir),
        (Join-Path $terminalFilesDir $DeployRelativeDir)
    )

    foreach ($sourceDir in $candidateSources) {
        $sourceManifest = Join-Path $sourceDir "manifest.csv"
        if (-not (Test-Path -LiteralPath $sourceManifest)) {
            continue
        }

        New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null
        $stageStart = Get-Date
        $sourceStats = Get-DirectoryStats -Path $sourceDir
        Get-ChildItem -LiteralPath $sourceDir | Copy-Item -Destination $ArtifactsDir -Recurse -Force
        $artifactStats = Get-DirectoryStats -Path $ArtifactsDir
        $elapsed = (Get-Date) - $stageStart
        Write-StageLog "cache" "Cache local inicializado | source=$sourceDir | source_stats=$(Format-DirectoryStats $sourceStats) | target=$ArtifactsDir | target_stats=$(Format-DirectoryStats $artifactStats) | elapsed=$(Format-Elapsed $elapsed)"
        return
    }

    Write-StageLog "cache" "Nenhum deploy anterior encontrado para semear cache local | target=$ArtifactsDir"
}

function Deploy-Package {
    param(
        [string]$TargetRoot,
        [string]$Label
    )

    $sourceDir = Resolve-Path $ArtifactsDir
    $targetDir = Join-Path $TargetRoot $DeployRelativeDir
    $stageStart = Get-Date
    $sourceStats = Get-DirectoryStats -Path $sourceDir
    $targetStatsBefore = Get-DirectoryStats -Path $targetDir

    Write-StageLog "deploy" "Iniciando deploy para ${Label} | source=$sourceDir | source_stats=$(Format-DirectoryStats $sourceStats) | target=$targetDir | target_before=$(Format-DirectoryStats $targetStatsBefore)"

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Get-ChildItem -LiteralPath $sourceDir | Copy-Item -Destination $targetDir -Recurse -Force
    $targetStatsAfter = Get-DirectoryStats -Path $targetDir
    $elapsed = (Get-Date) - $stageStart
    $manifestPath = Join-Path $targetDir "manifest.csv"
    Write-StageLog "deploy" "Pacote copiado para ${Label} | target=$targetDir | target_after=$(Format-DirectoryStats $targetStatsAfter) | manifest_exists=$(Test-Path -LiteralPath $manifestPath) | elapsed=$(Format-Elapsed $elapsed)"

    return [pscustomobject]@{
        TargetDir = $targetDir
        Elapsed = $elapsed
        SourceStats = $sourceStats
        TargetStatsBefore = $targetStatsBefore
        TargetStatsAfter = $targetStatsAfter
    }
}

function Invoke-Refresh {
    param(
        [string[]]$Reasons
    )

    $refreshStart = Get-Date
    $reasonText = if ($Reasons.Count -gt 0) { ($Reasons -join "|") } else { "manual" }
    $totalStageDurations = @{}
    $currentStage = "init"

    Write-StageLog "refresh" "Iniciando refresh | reason=$reasonText | input=$resolvedInput | artifacts=$ArtifactsDir | status=$statusPath"
    Write-StatusFile -State "running" -Reason $reasonText -Extra @{
        input_file = $resolvedInput
        artifacts_dir = $ArtifactsDir
        config_file = $ConfigFile
        started_at = $refreshStart.ToString("yyyy-MM-dd HH:mm:ss")
    }

    try {
        $dataUpdateResult = $null
        if (-not $SkipDataUpdate) {
            $currentStage = "data_update"
            $dataUpdateResult = Invoke-DataUpdate
            if ($dataUpdateResult) {
                $totalStageDurations["data_update_seconds"] = Get-DurationSeconds $dataUpdateResult.Elapsed
            }
        }
        else {
            Write-StageLog "data" "Atualizacao de dados ignorada por parametro"
            $totalStageDurations["data_update_seconds"] = 0
        }

        $latestBeforeTrainInfo = Get-CsvInfo -Path $resolvedInput
        $latestBeforeTrainDatasetInfo = Get-TrainingDatasetInfo -InputPath $resolvedInput -CsvInfo $latestBeforeTrainInfo
        $latestBeforeTrain = if ($latestBeforeTrainInfo) { $latestBeforeTrainInfo.LatestTime } else { $null }
        if ($latestBeforeTrainInfo -and $latestBeforeTrain) {
            Write-StageLog "data" "Ultimo candle do CSV: $($latestBeforeTrain.ToString('yyyy-MM-dd HH:mm:ss')) | rows=$($latestBeforeTrainInfo.RowCount)"
        }
        if ($latestBeforeTrainDatasetInfo) {
            Write-StageLog "data" "Dataset efetivo para treino: rows=$($latestBeforeTrainDatasetInfo.RowCount) | latest=$(Format-LogDateTime $latestBeforeTrainDatasetInfo.LatestTime)"
        }

        $referenceManifest = Get-ReferenceRuntimeManifest
        $historySync = Ensure-InputHistorySynchronized -CsvInfo $latestBeforeTrainInfo -DatasetInfo $latestBeforeTrainDatasetInfo -RuntimeManifest $referenceManifest -Context "refresh"
        if ($historySync.CsvInfo) {
            $latestBeforeTrainInfo = $historySync.CsvInfo
        }
        if ($historySync.DatasetInfo) {
            $latestBeforeTrainDatasetInfo = $historySync.DatasetInfo
        }
        $inputRegression = $historySync.InputRegression
        if ($inputRegression) {
            $regressionMessage = "Dataset efetivo de entrada esta atras do manifesto implantado | input_rows=$($inputRegression.InputRows) | manifest_rows=$($inputRegression.ManifestRows) | rows_behind=$($inputRegression.RowsBehind) | input_latest=$(Format-LogDateTime $inputRegression.InputLatestTime) | manifest_latest=$(Format-LogDateTime $inputRegression.ManifestLatestTime)"
            if ($Force) {
                Write-StageLog "warning" "$regressionMessage | continuando por causa do -Force"
            }
            else {
                throw $regressionMessage
            }
        }

        $currentStage = "seed_cache"
        Seed-ArtifactCacheFromDeploy
        $currentStage = "training"
        $trainingResult = Invoke-Training
        if ($trainingResult) {
            $totalStageDurations["training_seconds"] = Get-DurationSeconds $trainingResult.Elapsed
        }

        $liveDeployResult = $null
        if (-not $SkipDeployLive) {
            $currentStage = "deploy_live"
            $liveDeployResult = Deploy-Package -TargetRoot $terminalFilesDir -Label "MQL5\Files"
            if ($liveDeployResult) {
                $totalStageDurations["deploy_live_seconds"] = Get-DurationSeconds $liveDeployResult.Elapsed
            }
        }
        else {
            Write-StageLog "deploy" "Deploy live ignorado por parametro"
            $totalStageDurations["deploy_live_seconds"] = 0
        }

        $commonDeployResult = $null
        if (-not $SkipDeployCommon) {
            $currentStage = "deploy_common"
            $commonDeployResult = Deploy-Package -TargetRoot $commonFilesDir -Label "Terminal\Common\Files"
            if ($commonDeployResult) {
                $totalStageDurations["deploy_common_seconds"] = Get-DurationSeconds $commonDeployResult.Elapsed
            }
        }
        else {
            Write-StageLog "deploy" "Deploy common ignorado por parametro"
            $totalStageDurations["deploy_common_seconds"] = 0
        }

        $currentStage = "cleanup_request"
        if (Test-Path -LiteralPath $requestPath) {
            $requestCleanupStart = Get-Date
            Remove-Item -LiteralPath $requestPath -Force
            $cleanupElapsed = (Get-Date) - $requestCleanupStart
            $totalStageDurations["cleanup_request_seconds"] = Get-DurationSeconds $cleanupElapsed
            Write-StageLog "request" "Refresh request removido | path=$requestPath | elapsed=$(Format-Elapsed $cleanupElapsed)"
        }
        else {
            $totalStageDurations["cleanup_request_seconds"] = 0
            Write-StageLog "request" "Nenhum refresh request pendente para remover | path=$requestPath"
        }

        $artifactRoot = (Resolve-Path $ArtifactsDir).Path
        $artifactManifestPath = Join-Path $artifactRoot "manifest.csv"
        $artifactManifestJsonPath = Join-Path $artifactRoot "manifest.json"
        $artifactManifestInfo = Get-RuntimeManifestInfo -ManifestJsonPath $artifactManifestJsonPath -ManifestCsvPath $artifactManifestPath
        $latestAfterTrainInfo = Get-CsvInfo -Path $resolvedInput
        $latestAfterTrainDatasetInfo = Get-TrainingDatasetInfo -InputPath $resolvedInput -CsvInfo $latestAfterTrainInfo
        $latestAfterTrain = if ($latestAfterTrainInfo) { $latestAfterTrainInfo.LatestTime } else { $null }
        $liveTailAvailable = Test-LiveTailAvailable -PackageRoot $artifactRoot
        $coverageOk = $false
        if ($artifactManifestInfo -and ($artifactManifestInfo.OpenFolds -gt 0 -or $artifactManifestInfo.ClosedFolds -gt 0)) {
            $coverageOk = $true
        }
        elseif ($liveTailAvailable) {
            $coverageOk = $true
        }

        if ($artifactManifestInfo -and $artifactManifestInfo.NextFoldId) {
            $remainingRows = Get-RemainingRowsToRefresh -CsvInfo $latestAfterTrainDatasetInfo -RuntimeManifest $artifactManifestInfo
            Write-StageLog "runtime" "Proxima virada | fold=$($artifactManifestInfo.NextFoldId) | stage=$($artifactManifestInfo.NextRefreshStage) | row_count=$($artifactManifestInfo.NextRefreshRowCount) | rows_remaining=$(if ($null -ne $remainingRows) { $remainingRows } else { 'n/a' }) | dataset_rows=$(if ($latestAfterTrainDatasetInfo) { $latestAfterTrainDatasetInfo.RowCount } else { 'n/a' }) | open_folds=$($artifactManifestInfo.OpenFolds) | staged_folds=$($artifactManifestInfo.RuntimeStagedFolds)"
        }

        if (-not $coverageOk) {
            Write-StageLog "warning" "Pacote gerado sem fold de runtime utilizavel e sem live tail."
        }
        elseif ($liveTailAvailable -and $artifactManifestInfo -and $artifactManifestInfo.OpenFolds -le 0) {
            Write-StageLog "live" "Live tail mantido apenas como contingencia enquanto nao houver fold runtime aberto."
        }

        $refreshElapsed = (Get-Date) - $refreshStart
        $successExtra = @{
            started_at = $refreshStart.ToString("yyyy-MM-dd HH:mm:ss")
            finished_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            live_tail_available = $liveTailAvailable
            coverage_ok = $coverageOk
            refresh_elapsed_seconds = Get-DurationSeconds $refreshElapsed
        }
        $runtimeExtra = Get-RuntimeStatusExtra -CsvInfo $latestAfterTrainInfo -RuntimeManifest $artifactManifestInfo -DatasetInfo $latestAfterTrainDatasetInfo
        foreach ($key in $runtimeExtra.Keys) {
            $successExtra[$key] = $runtimeExtra[$key]
        }
        foreach ($key in $totalStageDurations.Keys) {
            $successExtra[$key] = $totalStageDurations[$key]
        }
        Write-StatusFile -State "success" -Reason $reasonText -Extra $successExtra

        $dataSeconds = if ($successExtra.ContainsKey("data_update_seconds")) { $successExtra["data_update_seconds"] } else { 0 }
        $trainSeconds = if ($successExtra.ContainsKey("training_seconds")) { $successExtra["training_seconds"] } else { 0 }
        $deployLiveSeconds = if ($successExtra.ContainsKey("deploy_live_seconds")) { $successExtra["deploy_live_seconds"] } else { 0 }
        $deployCommonSeconds = if ($successExtra.ContainsKey("deploy_common_seconds")) { $successExtra["deploy_common_seconds"] } else { 0 }
        Write-StageLog "done" "Refresh concluido com sucesso | reason=$reasonText | elapsed=$(Format-Elapsed $refreshElapsed) | data=$(('{0:N3}s' -f $dataSeconds)) | train=$(('{0:N3}s' -f $trainSeconds)) | deploy_live=$(('{0:N3}s' -f $deployLiveSeconds)) | deploy_common=$(('{0:N3}s' -f $deployCommonSeconds))"
    }
    catch {
        $refreshElapsed = (Get-Date) - $refreshStart
        Write-StatusFile -State "error" -Reason $reasonText -Extra @{
            started_at = $refreshStart.ToString("yyyy-MM-dd HH:mm:ss")
            finished_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            current_stage = $currentStage
            refresh_elapsed_seconds = Get-DurationSeconds $refreshElapsed
            error = $_.Exception.Message
        }
        Write-StageLog "error" "Refresh falhou | stage=$currentStage | reason=$reasonText | elapsed=$(Format-Elapsed $refreshElapsed) | error=$($_.Exception.Message)"
        throw
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

if (-not (Test-Path $PythonExe)) {
    Write-StageLog "python" "Interpretador informado nao encontrado: $PythonExe"
}

$selectedPythonStatus = Get-PythonDependencyStatus -Exe $PythonExe
if (-not $selectedPythonStatus.Ok) {
    Write-StageLog "python" "Falha nas dependencias do interpretador selecionado: $PythonExe | $($selectedPythonStatus.Error)"

    $resolvedPython = $null
    if ((Test-Path $defaultGlobalPython) -and ($PythonExe -ne $defaultGlobalPython)) {
        $globalPythonStatus = Get-PythonDependencyStatus -Exe $defaultGlobalPython
        if ($globalPythonStatus.Ok) {
            Write-StageLog "python" "Interpretador sem dependencias. Usando Python global: $defaultGlobalPython"
            $resolvedPython = $defaultGlobalPython
        }
        else {
            throw "Nao foi possivel encontrar um Python com as dependencias do projeto. selecionado='$($selectedPythonStatus.Error)' | global='$($globalPythonStatus.Error)'"
        }
    }
    else {
        throw "Nao foi possivel encontrar um Python com as dependencias do projeto. selecionado='$($selectedPythonStatus.Error)'"
    }

    $PythonExe = $resolvedPython
}

$resolvedInput = Resolve-InputFilePath -RequestedPath $InputFile
Write-StageLog "init" "Python=$PythonExe"
Write-StageLog "init" "Input=$resolvedInput"
foreach ($candidate in $script:InputResolutionReport) {
    $label = if ($candidate.Selected) { "selected" } else { "candidate" }
    Write-StageLog "input" "$label | path=$($candidate.Path) | rows=$($candidate.RowCount) | latest=$(Format-LogDateTime $candidate.LatestTime)"
}
$resolvedConfigPath = if ((Test-Path -LiteralPath $ConfigFile)) { (Resolve-Path -LiteralPath $ConfigFile).Path } else { (Join-Path $projectRoot $ConfigFile) }
Write-StageLog "init" "Config=$ConfigFile"
Write-StageLog "init" "ResolvedConfig=$resolvedConfigPath"
Write-StageLog "init" "Artifacts=$ArtifactsDir"
Write-StageLog "init" "RequestFile=$requestPath"
Write-StageLog "init" "StatusFile=$statusPath"
Write-StageLog "init" "DeployLive=$(Join-Path $terminalFilesDir $DeployRelativeDir)"
Write-StageLog "init" "DeployCommon=$(Join-Path $commonFilesDir $DeployRelativeDir)"

if (-not $Watch) {
    Invoke-Refresh -Reasons @("run_once")
    exit 0
}

Write-StageLog "watch" "Watcher iniciado | poll_seconds=$PollSeconds | refresh_cooldown_minutes=$RefreshCooldownMinutes"
$lastRefreshTime = Get-Date "2000-01-01"

while ($true) {
    try {
        Update-WatchInputData
        $decision = Get-RefreshDecision
        $inputRegression = Get-InputRegressionInfo -CsvInfo $decision.DatasetInfo -RuntimeManifest $decision.ManifestInfo
        if ($inputRegression) {
            $historySync = Ensure-InputHistorySynchronized -CsvInfo $decision.CsvInfo -DatasetInfo $decision.DatasetInfo -RuntimeManifest $decision.ManifestInfo -Context "watch"
            if ($historySync.Attempted) {
                $decision = Get-RefreshDecision
            }
            $inputRegression = if ($historySync.Attempted -or $historySync.InputRegression -eq $null) {
                Get-InputRegressionInfo -CsvInfo $decision.DatasetInfo -RuntimeManifest $decision.ManifestInfo
            }
            else {
                $historySync.InputRegression
            }
        }
        $statusExtra = Get-RuntimeStatusExtra -CsvInfo $decision.CsvInfo -RuntimeManifest $decision.ManifestInfo -DatasetInfo $decision.DatasetInfo
        $watchSummary = @(
            "need_refresh=$($decision.NeedRefresh)",
            "csv_rows=$(if ($decision.CsvInfo) { $decision.CsvInfo.RowCount } else { 'n/a' })",
            "csv_latest=$(Format-LogDateTime $(if ($decision.CsvInfo) { $decision.CsvInfo.LatestTime } else { $null }))",
            "dataset_rows=$(if ($decision.DatasetInfo) { $decision.DatasetInfo.RowCount } else { 'n/a' })",
            "dataset_latest=$(Format-LogDateTime $(if ($decision.DatasetInfo) { $decision.DatasetInfo.LatestTime } else { $null }))",
            "next_row_count=$(if ($decision.ManifestInfo) { $decision.ManifestInfo.NextRefreshRowCount } else { 'n/a' })",
            "rows_remaining=$(if ($null -ne $decision.RemainingRowsToRefresh) { $decision.RemainingRowsToRefresh } else { 'n/a' })",
            "next_stage=$(if ($decision.ManifestInfo) { $decision.ManifestInfo.NextRefreshStage } else { 'n/a' })",
            "next_fold=$(if ($decision.ManifestInfo) { $decision.ManifestInfo.NextFoldId } else { 'n/a' })",
            "open_folds=$(if ($decision.ManifestInfo) { $decision.ManifestInfo.OpenFolds } else { 'n/a' })",
            "request=$(if ($decision.Request) { $decision.Request.Reason } else { 'none' })"
        ) -join " | "
        Write-StageLog "watch" "Decision | $watchSummary"
        if ($inputRegression) {
            Write-StageLog "warning" "Dataset efetivo de entrada atras do manifesto | input_rows=$($inputRegression.InputRows) | manifest_rows=$($inputRegression.ManifestRows) | rows_behind=$($inputRegression.RowsBehind) | input_latest=$(Format-LogDateTime $inputRegression.InputLatestTime) | manifest_latest=$(Format-LogDateTime $inputRegression.ManifestLatestTime)"
            Write-StatusFile -State "input_regressed" -Reason "input_behind_manifest" -Extra $statusExtra
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ($decision.NeedRefresh) {
            $minutesSinceLastRun = ((Get-Date) - $lastRefreshTime).TotalMinutes
            if ($minutesSinceLastRun -lt $RefreshCooldownMinutes) {
                Write-StageLog "watch" "Refresh necessario, mas aguardando cooldown | reasons=$($decision.Reasons -join '|')"
                Write-StatusFile -State "cooldown" -Reason ($decision.Reasons -join "|") -Extra $statusExtra
            }
            else {
                Write-StageLog "watch" "Refresh solicitado | reasons=$($decision.Reasons -join '|')"
                Invoke-Refresh -Reasons $decision.Reasons
                $lastRefreshTime = Get-Date
                $Force = $false
            }
        }
        else {
            Write-StatusFile -State "idle" -Reason "watch_wait" -Extra $statusExtra
        }
    }
    catch {
        Write-StageLog "error" $_.Exception.Message
        Write-StatusFile -State "watch_error" -Reason "watch_loop" -Extra @{
            error = $_.Exception.Message
        }
    }

    Start-Sleep -Seconds $PollSeconds
}
