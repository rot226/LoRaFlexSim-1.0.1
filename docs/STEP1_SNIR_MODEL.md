# Modèle SNIR – Étape 1

Ce document résume les hypothèses de fading utilisées par les scripts de
l'étape 1 et leurs valeurs par défaut.

## Valeurs par défaut (scénario QoS multi-canaux)

Le scénario `loraflexsim/scenarios/qos_cluster_bench.py` configure un fading
rapide et un bruitage SNIR non nuls par défaut :

- `fast_fading_std = 1.0` dB (fading Rayleigh sur le RSSI).
- `snir_fading_std = 1.5` dB (fading Rayleigh appliqué au signal et aux
  interférences lors du calcul SNIR).
- `shadowing_std = 6.0` dB (valeur par défaut du canal si aucun override n'est
  appliqué).

Ces valeurs peuvent être remplacées via les overrides passés au scénario (voir
ci-dessous).

## Options CLI pour `scripts/run_step1_experiments.py`

Le script expose une option de sélection du modèle de fading :

- `--rayleigh` (par défaut) : conserve le fading rapide (`fast_fading_std`,
  `snir_fading_std`) et désactive le shadowing (`shadowing_std = 0`).
- `--shadowing` : active uniquement le shadowing log-normal et force
  `fast_fading_std = 0` et `snir_fading_std = 0`.

Les paramètres `--fading-std-db` et `--noise-floor-std-db` restent disponibles
pour affiner respectivement le fading SNIR et le bruit de fond.
