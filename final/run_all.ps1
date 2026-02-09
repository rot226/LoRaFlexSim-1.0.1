<# 
  Script d'orchestration (Windows 11, PowerShell) :
  - Enchaîne les simulations QoS baselines, SNIR et UCB1.
  - Ajustez les paramètres ci-dessous selon vos besoins.

  Exemple :
    .\final\run_all.ps1
    $env:CELL_RADIUS=3000; $env:RUNS=5; .\final\run_all.ps1
#>

$DataDir = $env:DATA_DIR
if ([string]::IsNullOrWhiteSpace($DataDir)) { $DataDir = "final/data" }

$CellRadius = $env:CELL_RADIUS
if ([string]::IsNullOrWhiteSpace($CellRadius)) { $CellRadius = 2500 }

$Runs = $env:RUNS
if ([string]::IsNullOrWhiteSpace($Runs)) { $Runs = 10 }

$Period = $env:PERIOD
if ([string]::IsNullOrWhiteSpace($Period)) { $Period = 300 }

$Duration = $env:DURATION
if ([string]::IsNullOrWhiteSpace($Duration)) { $Duration = 86400 }

# Chemin optionnel vers un INI (laissez vide si non utilisé).
$ConfigPath = $env:CONFIG_PATH

# Tailles de réseau (N ∈ {1000,...,15000}).
$NodesList = @(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000)

foreach ($Nodes in $NodesList) {
  $qosOut = Join-Path $DataDir "qos_baselines/N$Nodes"
  $snirOut = Join-Path $DataDir "snir/N$Nodes"
  $ucb1Out = Join-Path $DataDir "ucb1/N$Nodes"

  $commonArgs = @("--cell-radius", $CellRadius, "--nodes", $Nodes, "--runs", $Runs, "--period", $Period, "--duration", $Duration)
  if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $commonArgs += @("--config", $ConfigPath)
  }

  python final/scenarios/run_qos_baselines.py @commonArgs --output-dir $qosOut
  python final/scenarios/run_snir.py @commonArgs --output-dir $snirOut
  python final/scenarios/run_ucb1.py @commonArgs --output-dir $ucb1Out
}
