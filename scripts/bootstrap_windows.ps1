[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $rootDir

$venvDir = Join-Path $rootDir ".venv"
$venvPython = Join-Path $venvDir "Scripts/python.exe"
$activateScript = Join-Path $venvDir "Scripts/Activate.ps1"

function Show-RunCommand {
    param(
        [bool]$EditableInstalled
    )

    Write-Host ""
    Write-Host "==== Commande à utiliser ====" -ForegroundColor Cyan

    if ($EditableInstalled) {
        Write-Host "CLI installée :" -ForegroundColor Green
        Write-Host "  mobilesfrdth --help"
    } else {
        Write-Host "Mode fallback sans installation editable :" -ForegroundColor Yellow
        Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/mobilesfrdth.ps1 --help"
        Write-Host "  # (équivalent direct)"
        Write-Host "  `$env:PYTHONPATH='src'; python -m mobilesfrdth --help"
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Le lanceur Python 'py' est introuvable. Installez Python 3.11+ puis relancez ce script."
    exit 1
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Création de l'environnement virtuel .venv avec py -3.11..."
    py -3.11 -m venv .venv
}

if (-not (Test-Path $activateScript)) {
    Write-Error "Environnement virtuel introuvable ou incomplet : '$venvDir'. Vérifiez la création de .venv puis relancez ce script."
    exit 1
}

Write-Host "Activation de .venv..."
. $activateScript

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python n'est pas accessible après activation de .venv. Vérifiez '$venvDir'."
    exit 1
}

Write-Host "Version Python active :" -ForegroundColor Cyan
python --version

Write-Host "Vérification de l'import setuptools..." -ForegroundColor Cyan
$setuptoolsOk = $true
python -c "import setuptools" 2>$null
if ($LASTEXITCODE -ne 0) {
    $setuptoolsOk = $false
}

if (-not $setuptoolsOk) {
    Write-Warning "Import setuptools KO. Tentative d'installation de setuptools..."
    python -m pip install setuptools
    python -c "import setuptools"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "setuptools reste indisponible : passage en mode fallback PYTHONPATH=src."
        Show-RunCommand -EditableInstalled $false
        exit 0
    }
}

Write-Host "Installation du projet en mode editable (sans build isolation)..." -ForegroundColor Cyan
python -m pip install -e . --no-build-isolation
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Échec de 'pip install -e . --no-build-isolation'."
    Write-Warning "Basculer automatiquement en mode PYTHONPATH=src."
    Show-RunCommand -EditableInstalled $false
    exit 0
}

Write-Host "Bootstrap Windows terminé." -ForegroundColor Green
Show-RunCommand -EditableInstalled $true
