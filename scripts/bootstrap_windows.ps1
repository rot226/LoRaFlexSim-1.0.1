[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $rootDir

$venvDir = Join-Path $rootDir ".venv"
$venvPython = Join-Path $venvDir "Scripts/python.exe"
$activateScript = Join-Path $venvDir "Scripts/Activate.ps1"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Le lanceur Python 'py' est introuvable. Installez Python 3.11 puis relancez ce script."
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

Write-Host "Installation du projet en mode editable (sans build isolation)..."
python -m pip install -e . --no-build-isolation

Write-Host "Bootstrap Windows terminé."
