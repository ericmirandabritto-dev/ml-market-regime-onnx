param(
    [switch]$ForceRecreate,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ($ForceRecreate -and (Test-Path $venvPath)) {
    Write-Host "Removendo .venv anterior..."
    Remove-Item -LiteralPath $venvPath -Recurse -Force
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Criando ambiente virtual em $venvPath"
    & $PythonExe -m venv $venvPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "Falha ao criar o ambiente virtual em $venvPath"
    }
}

Write-Host "Instalando dependencias no .venv..."
$requirements = Join-Path $projectRoot "requirements.txt"
if (Test-Path -LiteralPath $requirements -PathType Leaf) {
    & $venvPython -m pip install -r $requirements
}
else {
    # Keep the public checkout self-contained with the tested CPU baseline.
    & $venvPython -m pip install numpy==2.4.4 pandas==3.0.2 scikit-learn==1.8.0 PyYAML==6.0.3 onnx==1.21.0 skl2onnx==1.20.0 onnxruntime==1.24.4
}
$pipExitCode = $LASTEXITCODE

if ($pipExitCode -ne 0) {
    throw @"
Falha ao instalar dependencias no .venv (exit code $pipExitCode).

Se a maquina estiver sem acesso ao indice do pip, o ambiente pode ficar parcialmente instalado.
Use um espelho interno, habilite o acesso de rede para o pip ou instale manualmente os pacotes faltantes no .venv.
"@
}

Write-Host "Ambiente pronto."
Write-Host "Python do projeto: $venvPython"
