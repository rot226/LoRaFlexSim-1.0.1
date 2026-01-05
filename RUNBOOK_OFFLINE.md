# Runbook hors-ligne

Ce guide décrit les commandes **exactes** à exécuter depuis la racine du dépôt pour
reproduire les CSV Step 1, générer les figures étendues et lancer les tests
SNIR/QoS. Aucune connexion réseau n’est requise.

> **Note IEEE** : seules les figures situées dans `figures/step1/extended/` sont
> considérées comme validées IEEE.

## 1) Générer les CSV Step 1

```bash
python scripts/run_step1_matrix.py --algos adr apra mixra_h mixra_opt --with-snir true false --seeds 1 2 3 --nodes 1000 5000 --packet-intervals 300 600
python scripts/aggregate_step1_results.py --strict-snir-detection
```

Résultats attendus :
- CSV bruts : `results/step1/<snir_state>/seed_<seed>/`.
- CSV agrégés : `results/step1/summary.csv` et `results/step1/raw_index.csv`.

## 2) Générer les figures « extended »

```bash
python scripts/plot_step1_results.py --official --use-summary --plot-cdf
```

Résultats attendus :
- Figures officielles : `figures/step1/extended/`.

## 3) Exécuter les tests SNIR/QoS

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

### 1) Générer les CSV Step 1

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_step1_matrix_windows.ps1
python scripts/aggregate_step1_results.py --strict-snir-detection
```

### 2) Générer les figures « extended »

```powershell
python scripts/plot_step1_results.py --official --use-summary --plot-cdf
```

### 3) Exécuter les tests SNIR/QoS

```powershell
python scripts/validate_snir_plots.py --nodes 8 --duration 120 --packet-interval 60
pytest tests/qos/test_snir_window_effect.py
pytest tests/test_qos_clusters.py
pytest tests/test_qos_validation_script.py
```
