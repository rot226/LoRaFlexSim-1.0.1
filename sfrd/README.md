# sfrd

Structure initiale pour les scripts CLI et les parseurs SFRD.

## Reward UCB (normalisée)

- La pondération énergétique `lambda_E` est exposée dans `sfrd/parse/reward_ucb.py` via la constante `LAMBDA_E` (valeur par défaut: `0.5`).
- Cette pondération est appliquée dans le flux d'exécution UCB (instrumentation de `LoRaSFSelectorUCB1` + `Simulator`) afin de journaliser directement `reward_raw` et `reward_normalized` pendant la simulation.
- Chaque run UCB exporte un log dédié `ucb_history.csv` avec: `episode`, `reward_raw`, `reward_normalized`, `chosen_sf`, `success_rate`, `bitrate_norm`, `energy_norm`.

## Agrégation `learning_curve_ucb.csv`

Stratégie d'alignement des épisodes en cas de multi-runs UCB:

1. chaque run produit sa courbe locale avec des épisodes démarrant à 1;
2. l'agrégateur aligne les runs par numéro d'épisode;
3. la valeur `reward` finale est la **moyenne simple** des `reward_normalized` disponibles pour cet épisode (sans interpolation des épisodes absents).
