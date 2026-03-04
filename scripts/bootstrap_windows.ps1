Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fail([string]$Message) {
    Write-Error $Message
    exit 1
}

function Get-PythonCandidate {
    $candidates = @(
        @{ Exe = "py"; Args = @("-3.12"); Label = "Python 3.12" },
        @{ Exe = "py"; Args = @("-3.11"); Label = "Python 3.11" },
        @{ Exe = "python"; Args = @(); Label = "python" }
    )

    foreach ($candidate in $candidates) {
        try {
            $versionText = & $candidate.Exe @($candidate.Args + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")) 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $versionText) {
                continue
            }

            $versionText = $versionText.Trim()
            if ($versionText -in @("3.11", "3.12")) {
                return [PSCustomObject]@{
                    Exe = $candidate.Exe
                    Args = $candidate.Args
                    Version = $versionText
                    Label = $candidate.Label
                }
            }
        } catch {
            continue
        }
    }

    try {
        $detected = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null).Trim()
        if ($detected) {
            Fail "Python détecté ($detected) mais version non supportée. Utilisez Python 3.11 ou 3.12."
        }
    } catch {
        # Python introuvable, message géré ci-dessous.
    }

    Fail "Python introuvable. Installez Python 3.11 ou 3.12 puis relancez ce script."
}

$python = Get-PythonCandidate
Write-Host "Interpréteur retenu : $($python.Label) ($($python.Version))"

if (-not (Test-Path ".venv")) {
    Write-Host "Création de l'environnement virtuel .venv..."
    & $python.Exe @($python.Args + @("-m", "venv", ".venv"))
    if ($LASTEXITCODE -ne 0) {
        Fail "Impossible de créer l'environnement virtuel .venv."
    }
} else {
    Write-Host ".venv déjà présent."
}

$activateScript = Join-Path ".venv" "Scripts/Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Fail "Script d'activation introuvable : $activateScript"
}

Write-Host "Activation de .venv..."
. $activateScript

Write-Host "Mise à jour de pip..."
python -m pip install -U pip
if ($LASTEXITCODE -ne 0) {
    Fail "Échec de la mise à jour de pip."
}

Write-Host "Installation du projet en mode editable..."
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Fail "Échec de l'installation editable (pip install -e .)."
}

Write-Host "Bootstrap terminé avec succès."
