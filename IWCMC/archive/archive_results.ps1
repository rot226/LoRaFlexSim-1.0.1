param(
    [string]$RepoRoot = $(Resolve-Path (Join-Path $PSScriptRoot ".." ".."))
)

$iwcmcDir = Join-Path $RepoRoot "IWCMC"
$archiveDir = Join-Path $iwcmcDir "archive"
New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archivePath = Join-Path $archiveDir "iwcmc_results_$stamp.tar.gz"

$targets = @(
    "IWCMC/snir_static/data",
    "IWCMC/snir_static/figures",
    "IWCMC/rl_static/figures",
    "IWCMC/rl_mobile/figures",
    "results/iwcmc"
) | Where-Object { Test-Path (Join-Path $RepoRoot $_) }

if ($targets.Count -eq 0) {
    Write-Error "Aucun dossier de résultats à archiver."
    exit 1
}

$targetArgs = $targets -join " "
Push-Location $RepoRoot
try {
    tar -czf $archivePath $targetArgs
} finally {
    Pop-Location
}

Write-Host "Archive créée : $archivePath"
