# sfrd

Structure initiale pour les scripts CLI et les parseurs SFRD.

## Prérequis

- Python 3.10+ recommandé.
- Dépendances du projet installées (depuis la racine du dépôt), par exemple:
  - `python -m pip install -r requirements.txt`
- Exécuter les commandes depuis la racine du projet pour conserver les chemins `sfrd/...`.

## Campagne principale (complète)

```bash
python -m sfrd.cli.run_campaign --network-sizes 80 160 320 640 1280 --replications 5 --seeds-base 1 --snir OFF,ON --algos UCB ADR MixRA-H MixRA-Opt --warmup-s 0
```

## Validation des sorties

```bash
python -m sfrd.cli.validate_outputs --root sfrd/output
```

## Agrégation optionnelle

```bash
python -m sfrd.parse.aggregate --logs-root sfrd/logs --out-root sfrd/output
```

## Définition des métriques

- **PDR (Packet Delivery Ratio)**: proportion de paquets correctement reçus sur le total envoyé.
  - Formule type: `PDR = paquets_reçus / paquets_envoyés`.
- **Throughput**: volume utile livré par unité de temps (souvent en bps/kbps).
  - Formule type: `throughput = bits_reçus_utiles / durée_observation`.
- **Energy/packet**: énergie moyenne consommée par paquet transmis (ou livré selon la convention d'analyse).
  - Formule type: `energy_per_packet = énergie_totale / nb_paquets`.
- **SF distribution**: répartition des transmissions par Spreading Factor (SF7..SF12, etc.), utile pour observer l'équilibre charge/robustesse radio.
- **Warm-up**: fenêtre initiale de simulation exclue des métriques finales pour éviter les biais de démarrage.
  - Ici, `--warmup-s 0` signifie qu'aucune fenêtre de chauffe n'est retranchée.
- **Agrégation des réplications**: combinaison des résultats de plusieurs runs (seeds différentes) pour obtenir des statistiques plus robustes (moyenne, dispersion, intervalles éventuels).

## Reward UCB

### Formule (principe)

La récompense UCB combine performance de livraison et coût énergétique:

- `reward_raw` = combinaison pondérée de composantes normalisées (succès, débit, énergie).
- `reward_normalized` = version bornée/normalisée de `reward_raw` pour stabiliser la comparaison entre épisodes.

Les colonnes exportées dans `ucb_history.csv` sont: `episode`, `reward_raw`, `reward_normalized`, `chosen_sf`, `success_rate`, `bitrate_norm`, `energy_norm`.

### Normalisation

- Les composantes (`success_rate`, `bitrate_norm`, `energy_norm`) sont ramenées sur des échelles comparables.
- La récompense normalisée est utilisée pour les courbes d'apprentissage et l'agrégation inter-réplications.

### `lambda_E`

- La pondération énergétique `lambda_E` est exposée dans `sfrd/parse/reward_ucb.py` via la constante `LAMBDA_E` (valeur par défaut: `0.5`).
- Cette pondération contrôle le compromis performance énergétique vs performance de livraison/débit.

### Définition d'un épisode

- Un **épisode** correspond à une unité de décision/apprentissage UCB pour laquelle une action (ex. choix de SF) est évaluée puis journalisée avec sa récompense.
- Les épisodes sont indexés à partir de 1 dans chaque run UCB.

## Agrégation `learning_curve_ucb.csv`

Stratégie d'alignement des épisodes en cas de multi-runs UCB:

1. chaque run produit sa courbe locale avec des épisodes démarrant à 1;
2. l'agrégateur aligne les runs par numéro d'épisode;
3. la valeur `reward` finale est la **moyenne simple** des `reward_normalized` disponibles pour cet épisode (sans interpolation des épisodes absents).


## Configuration UCB externalisée

Le fichier `sfrd/config/ucb_config.json` contient désormais:

- `lambda_E` (poids énergie, injecté directement dans `energy_penalty_weight` du sélecteur UCB);
- `exploration_coefficient` (bonus UCB: `sqrt(c * log(t) / n)`);
- `reward_window` (fenêtre glissante interne);
- `episode` (`mode=packets|time`, `packet_window`, `time_window_s`).

Aucun recalcul a posteriori de reward n'est effectué: la courbe d'apprentissage
`learning_curve_ucb.csv` est dérivée directement de `reward_normalized` journalisé par épisode.

## Mini-campagne de calibration UCB

Exemple (sous-ensemble tailles/seeds):

```bash
python -m sfrd.cli.calibrate_ucb --network-sizes 40 80 --replications 2 --seeds-base 101 --warmup-s 0.0
```

Cette commande exécute plusieurs configs candidates, compare et archive:

- `pdr_results.csv`
- `throughput_results.csv`
- `learning_curve_ucb.csv`

par candidate sous `sfrd/output/ucb_calibration/<candidate>/...`, puis choisit une
config finale selon un critère documenté (`maximize(pdr_mean - 0.2 * energy_mean)`) et
la fige dans `sfrd/config/ucb_config.json` pour les campagnes “preuves”.
