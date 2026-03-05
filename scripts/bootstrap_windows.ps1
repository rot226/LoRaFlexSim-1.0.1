[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $rootDir

$venvDir = Join-Path $rootDir ".venv"
$venvPython = Join-Path $venvDir "Scripts/python.exe"
$activateScript = Join-Path $venvDir "Scripts/Activate.ps1"

if (-not (Test-Path $venvPython)) {
    Write-Host "Création de l'environnement virtuel .venv..."
    python -m venv .venv
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

Write-Host "Mise à jour de pip..."
python -m pip install --upgrade pip

Write-Host "Installation du projet en mode editable..."
python -m pip install -e .

$mobilesfrdth = Get-Command mobilesfrdth -ErrorAction SilentlyContinue
if ($null -ne $mobilesfrdth) {
    Write-Host "CLI détectée : mobilesfrdth ($($mobilesfrdth.Source))"
} else {
    Write-Warning "La commande 'mobilesfrdth' est introuvable dans le PATH du venv. Fallback vers le mode module Python 'sfrd'."
    Write-Host "Exemple fallback : python -m sfrd.cli.run_campaign --help"
    python -m sfrd.cli.run_campaign --help
}

Write-Host "Bootstrap Windows terminé."
