# Étape 2 — Normalisation et figures

Ce mode d’emploi décrit la **normalisation** des sorties Step 2 (CSV de
métriques et de décisions) vers `results/step2/raw|agg`, puis la génération des
figures comparatives.

Les figures Step 2 intègrent désormais la **récompense moyenne** ainsi que ses
composantes (succès, SNIR, énergie, collisions, équité). Cette décomposition
reflète l’agrégation utilisée côté bandit multi-bras (MAB/UCB1) : une moyenne
pondérée normalisée des composantes configurées dans `loraflexsim/learning/ucb1.py`.

## Préparer les entrées Step 2

Les scripts attendent des CSV similaires à ceux générés par les campagnes UCB1
(`experiments/ucb1/*decision_log*.csv` et `*metrics*.csv`). Pour bénéficier des
colonnes **SNIR OFF/ON**, placez vos exports dans des dossiers nommés
`snir_off`/`snir_on` (ou incluez `snir_off`/`snir_on` dans le chemin).

Exemple de structure conseillée :

```
results/step2/input/
  snir_off/
    ucb1_baseline_decision_log.csv
    ucb1_baseline_metrics.csv
  snir_on/
    ucb1_baseline_decision_log.csv
    ucb1_baseline_metrics.csv
```

## Démo rapide

1. Copiez vos CSV (métriques + décisions) dans `results/step2/input/` en
   conservant un sous-dossier `snir_off` et/ou `snir_on`.
2. Normalisez les sorties :

```powershell
python scripts/run_step2_scenarios.py --input-dir results/step2/input
```

3. Générez les figures :

```powershell
python scripts/plot_step2_comparison.py --results-dir results/step2
```

Les PNG sont produits dans `figures/step2`. **`plot_step2_comparison.py` est
le seul générateur de figures Step 2** : évitez d’appeler d’autres scripts de
plots Step 2 (en CI ou dans vos wrappers) en parallèle.

## Paper runs (campagnes complètes)

1. Lancez vos campagnes Step 2 (par exemple les scripts UCB1 ou MixRA)
   **deux fois** : une fois avec SNIR désactivé, une fois avec SNIR activé.
2. Rangez les exports dans des sous-dossiers `snir_off`/`snir_on` comme
   illustré ci-dessus.
3. Normalisez et tracez :

```powershell
python scripts/run_step2_scenarios.py --input-dir results/step2/input --output-dir results/step2
python scripts/plot_step2_comparison.py --results-dir results/step2 --output-dir figures/step2
```

Pour ne produire que les figures principales (performance + convergence), ajoutez
`--only-core-figures` :

```powershell
python scripts/plot_step2_comparison.py --results-dir results/step2 --only-core-figures
```

### Détails des sorties

- `results/step2/raw/decisions.csv` : décisions normalisées (rounds/épisodes).
- `results/step2/raw/metrics.csv` : métriques agrégées par cluster.
- `results/step2/agg/performance_rounds.csv` : performance vs rounds.
- `results/step2/agg/convergence.csv` : convergence moyenne + CI95.
- `results/step2/agg/sf_tp_distribution.csv` : distribution SF/TP (optionnelle).
- `figures/step2/step2_reward_components_ci95.png` : récompense moyenne et
  composantes (succès, SNIR, énergie, collisions = 1 − PDR, équité) par scénario.
