# Runbook hors-ligne

Ce guide décrit les commandes **exactes** à exécuter depuis la racine du dépôt pour
reproduire les CSV Step 1, générer les figures étendues et lancer les tests
SNIR/QoS. Aucune connexion réseau n’est requise.

> **Note IEEE** : les figures **IEEE-ready** sont celles de l’étape 1, situées
> dans `figures/step1/extended/`. Les figures de l’étape 2 (comparaison) ne sont
> pas IEEE-ready.

## 1) Step 1 — Générer les CSV

```bash
python scripts/run_step1_matrix.py --algos adr apra mixra_h mixra_opt --with-snir true false --seeds 1 2 3 --nodes 1000 5000 --packet-intervals 300 600
python scripts/aggregate_step1_results.py --strict-snir-detection
```

Résultats attendus :
- CSV bruts : `results/step1/<snir_state>/seed_<seed>/`.
- CSV agrégés : `results/step1/summary.csv` et `results/step1/raw_index.csv`.

## 2) Step 2 — Comparaison et figures Step 2

```bash
python scripts/run_step2_scenarios.py
python scripts/plot_step1_results.py --official --use-summary --plot-cdf
python scripts/plot_step2_comparison.py
```

Résultats attendus :
- Normalisation Step 2 : `results/step2/raw` et `results/step2/agg`.
- Figures Step 1 (IEEE-ready) : `figures/step1/extended/`.
- Figures Step 2 (comparaison) : `figures/step2/*.png` et `figures/step2/*.pdf`.

## 3) Exécuter les tests SNIR/QoS

> Les tests **rapides** doivent inclure **plusieurs tailles** de réseau ;
> utilisez par exemple `--network-sizes 80 160 320 640 1280` pour couvrir
> rapidement plusieurs échelles.

### SNIR

```bash
python scripts/validate_snir_plots.py --nodes 8 --duration 120 --packet-interval 60
pytest tests/qos/test_snir_window_effect.py
```

### QoS

```bash
pytest tests/test_qos_clusters.py
pytest tests/test_qos_validation_script.py
```

## Section Windows PowerShell

> Exécuter ces commandes dans un terminal PowerShell (Windows 11).

### 1) Step 1 — Générer les CSV

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_step1_matrix_windows.ps1
python scripts/aggregate_step1_results.py --strict-snir-detection
```

### 2) Step 2 — Comparaison et figures Step 2

```powershell
python scripts/run_step2_scenarios.py
python scripts/plot_step1_results.py --official --use-summary --plot-cdf
python scripts/plot_step2_comparison.py
```

### 3) Exécuter les tests SNIR/QoS

```powershell
python scripts/validate_snir_plots.py --nodes 8 --duration 120 --packet-interval 60
pytest tests/qos/test_snir_window_effect.py
pytest tests/test_qos_clusters.py
pytest tests/test_qos_validation_script.py
```
