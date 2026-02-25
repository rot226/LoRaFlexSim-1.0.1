# mobile-sfrd_th

Structure initiale d'un package Python 3.11 indépendant.

## Génération des figures depuis `aggregates/*.csv`

La commande de tracé lit **uniquement** les fichiers CSV produits par l'étape `aggregate`:

- `metric_by_factor.csv`
- `distribution_sf.csv`
- `convergence_tc.csv`
- `sinr_cdf.csv`
- `fairness_airtime_switching.csv`

Elle génère dans le dossier de sortie (`--out`) les figures minimales:

1. `fig01_pdr_vs_n_snir_off.png`
2. `fig02_pdr_vs_n_snir_on.png`
3. `fig03_der_vs_n_snir_off.png`
4. `fig04_der_vs_n_snir_on.png`
5. `fig05_throughput_vs_n_snir_off.png`
6. `fig06_throughput_vs_n_snir_on.png`
7. `fig07_tc_vs_speed.png`
8. `fig08_fairness_vs_n.png`
9. `fig09_sf_distribution.png`
10. `fig10_sinr_cdf.png`

Bonus (si données disponibles):

11. `fig11_airtime_vs_n.png`
12. `fig12_switch_count_vs_n.png`

### Exemple Windows 11 (PowerShell)

```powershell
# Depuis la racine du dépôt
mobilesfrdth plots `
  --aggregates-dir .\runs\aggregates `
  --out .\runs\plots
```

### Exemple avec filtres de scénario (PowerShell)

```powershell
mobilesfrdth plots `
  --aggregates-dir .\runs\aggregates `
  --out .\runs\plots_filtered `
  --scenario-filter mode=snir_on `
  --scenario-filter algo=ucb,legacy `
  --scenario-filter mobility_model=rwp
```

### Désactiver les figures bonus

```powershell
mobilesfrdth plots `
  --aggregates-dir .\runs\aggregates `
  --out .\runs\plots_minimal `
  --no-bonus
```

> En cas de données manquantes (fichier absent, colonne absente, lignes non numériques), la commande émet un warning explicite et ignore uniquement la/les figure(s) concernée(s).
