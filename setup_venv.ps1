param(
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ($ForceRecreate -and (Test-Path $venvPath)) {
    Write-Host "Removendo .venv anterior..."
    Remove-Item -LiteralPath $venvPath -Recurse -Force
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Criando ambiente virtual em $venvPath"
    python -m venv $venvPath
}

Write-Host "Instalando dependencias no .venv..."
& $venvPython -m pip install -r requirements.txt
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
