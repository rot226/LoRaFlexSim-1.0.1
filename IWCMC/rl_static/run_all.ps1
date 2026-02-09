$ErrorActionPreference = "Stop"

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $rootDir

python IWCMC/rl_static/scenarios/run_ucb1_vs_qos.py @Args
python IWCMC/rl_static/plots/plot_rls_figures.py
