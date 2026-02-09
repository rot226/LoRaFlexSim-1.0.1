$ErrorActionPreference = "Stop"

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $rootDir

python IWCMC/rl_mobile/scenarios/run_rl_mobile.py @Args
python IWCMC/rl_mobile/plots/plot_rlm_figures.py
