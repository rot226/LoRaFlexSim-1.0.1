<#
Exécution guidée de la matrice d'essais Step 1 sous Windows.

Ce script :
- se place à la racine du dépôt ;
- active l'environnement virtuel local (``.\\venv`` prioritaire, sinon ``.\\env``) ;
- lance ``python scripts/run_step1_matrix.py`` avec les paramètres
  recommandés pour générer les CSV sous ``results/step1/<snir_state>/seed_<seed>/``.
#>

[CmdletBinding()]
param(
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"

# Localise la racine du dépôt depuis le dossier scripts/
$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $rootDir

if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    if (Test-Path ".\\.venv\\Scripts\\Activate.ps1") {
        $VenvPath = ".\\.venv"
    } elseif (Test-Path ".\\env\\Scripts\\Activate.ps1") {
        $VenvPath = ".\\env"
    } else {
        throw "Aucun venv détecté. Créez un environnement avec 'python -m venv .venv' ou 'python -m venv env', ou fournissez -VenvPath."
    }
}

$activateScript = Join-Path $VenvPath "Scripts/Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "Fichier d'activation introuvable : $activateScript. Créez l'environnement avec 'python -m venv $VenvPath'."
}

$pythonExe = Join-Path $VenvPath "Scripts/python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Interpréteur Python introuvable dans le venv : $pythonExe"
}

Write-Host "Activation de l'environnement virtuel '$VenvPath'..."
. $activateScript

$arguments = @(
    "scripts/run_step1_matrix.py"
    "--algos" "adr" "apra" "mixra_h" "mixra_opt"
    "--with-snir" "true" "false"
    "--seeds" "1" "2" "3"
    "--nodes" "1000" "5000"
    "--packet-intervals" "300" "600"
)

Write-Host "Lancement de la matrice Step 1..."
& $pythonExe @arguments
