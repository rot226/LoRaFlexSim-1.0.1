<#
  Script de tracé (Windows 11, PowerShell) :
  - Génère toutes les figures à partir des CSV déjà présents.
  - Ajustez DATA_DIR / FIGURES_DIR si besoin.

  Exemple :
    .\final\plot_all.ps1
#>

$DataDir = $env:DATA_DIR
if ([string]::IsNullOrWhiteSpace($DataDir)) { $DataDir = "final/data" }

$FiguresDir = $env:FIGURES_DIR
if ([string]::IsNullOrWhiteSpace($FiguresDir)) { $FiguresDir = "final/figures" }

python final/plots/plot_der_vs_nodes.py --data-dir $DataDir --output-dir $FiguresDir
python final/plots/plot_throughput.py --data-dir $DataDir --output-dir $FiguresDir
python final/plots/plot_snir_distribution.py --data-dir $DataDir --output-dir $FiguresDir
