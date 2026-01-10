# Hypothèses et approximations du calcul SNIR

## Chaîne de calcul

- Le RSSI/SNR « instantané » est évalué dans `loraflexsim/launcher/channel.py` via
  `compute_rssi`, puis le SNIR combine le bruit thermique et les puissances
  concurrentes dans `compute_snir`. La valeur `snir_fading_std` introduit un
  fading aléatoire propre au SNIR (appliqué au signal **et** aux interférences)
  pour éviter une vision trop déterministe du canal.
- Les collisions et le capture effect sont résolus dans
  `loraflexsim/launcher/gateway.py` (`start_reception`). Les RSSI peuvent être
  perturbés par un fading rapide (gaussien) avant le test de capture; lorsque le
  SNIR ne dépasse le seuil que de quelques dB, une perte marginale peut être
  déclenchée avec une probabilité configurable.
- `loraflexsim/launcher/simulator.py` regroupe les interférences dans
  `InterferenceTracker`, désormais bruitées par un fading léger pour refléter la
  variabilité des collisions au lieu d'une moyenne strictement optimiste.

## Hypothèses/approximations identifiées

- La propagation reste log-distance par défaut, sans modèle multi-trajet
  explicite (hors options `multipath_taps` et `fast_fading_std`). Les valeurs
  très faibles de fading ou de shadowing peuvent donc conduire à des SNIR plus
  optimistes que sur le terrain.
- Le capture effect repose sur des seuils fixes (table SNIR ou `capture_threshold`).
  Même avec le bruitage marginal, un paquet juste au-dessus du seuil reste
  susceptible d'être favorisé si la matrice non-orthogonale est trop permissive.
- Les interférences sont intégrées comme puissances moyennes sur la durée
  d'overlap. Des pics impulsifs restent modélisés via `impulsive_noise_prob`,
  mais il n'existe pas encore de profil temporel fin (par symbole) pour les
  collisions partielles.

## Paramètres clefs pour durcir le réalisme

- `snir_fading_std` (Channel) : écart-type du fading injecté dans les calculs
  SNIR et dans la capture (`1.5 dB` par défaut lorsque `use_snir=True`).
- `marginal_snir_margin_db` et `marginal_snir_drop_prob` (Channel → Gateway) :
  contrôlent le seuil de marge et la probabilité maximale de perte aléatoire en
  cas de SNIR à peine supérieur au seuil.
- `InterferenceTracker` applique désormais un fading léger sur la puissance
  cumulée pour éviter d'estimer une interférence constamment moyenne.
- Les générateurs pseudo-aléatoires sont découplés (canaux, passerelles,
  collisions) et dérivés de `seed` via `RngManager`, limitant les corrélations
  fortuites entre placement, fading et captures. Les canaux héritent
  explicitement d'un flux `channel` pour rendre reproductibles les fades SNIR
  injectés dans le calcul d'interférence.

## Configuration rapide

- La section `[channel]` d'un fichier INI (par ex. `config.ini`) peut fournir
  `snir_fading_std`, `noise_floor_std`, `interference_dB`,
  `capture_threshold_dB`, `sensitivity_margin_dB`,
  `marginal_snir_margin_db`, `marginal_snir_drop_prob`,
  `baseline_loss_rate`, `baseline_collision_rate`,
  `residual_collision_prob` et `snir_off_noise_prob`. Passer
  `channel_config=<chemin>` au constructeur `Simulator` applique ces valeurs à
  tous les canaux créés par défaut.
- Les mêmes clés restent surchargeables à l'initialisation du `Simulator`
  (arguments nommés), ce qui permet d'ajuster rapidement le niveau de bruit de
  fond ou la sévérité du capture effect sans modifier le code.

## Plages raisonnables (article-friendly)

Les valeurs ci-dessous constituent des ordres de grandeur raisonnables pour
documenter un scénario dans un article sans surcharger la simulation :

- `baseline_loss_rate` : 0.001 à 0.01
- `baseline_collision_rate` : 0.005 à 0.03
- `residual_collision_prob` : 0.0 à 0.05
- `snir_off_noise_prob` : 0.0 à 0.02
