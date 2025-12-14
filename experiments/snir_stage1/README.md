# Expérimentations SNIR – Étape 1

## Pré-requis
- Python 3.10 ou plus récent.
- Dépendances installées en mode développement :
  ```bash
  pip install -e .
  ```

## Scripts de simulation
Chaque scénario écrit un CSV dans `data/` à la racine du dépôt.

- Densité (PDR/DER/SNIR par cluster) :
  ```bash
  python experiments/snir_stage1/scenarios/der_density.py
  # -> data/der_density.csv
  ```
- Charge (variation de l'intervalle d'émission) :
  ```bash
  python experiments/snir_stage1/scenarios/der_load.py
  # -> data/der_load.csv
  ```
- Trafic offert et taux d'utilisation radio :
  ```bash
  python experiments/snir_stage1/scenarios/offered_traffic.py
  # -> data/offered_traffic.csv
  ```
- Débit agrégé et collisions :
  ```bash
  python experiments/snir_stage1/scenarios/throughput.py
  # -> data/throughput.csv
  ```

## Génération des figures
- PDR/DER en fonction de la densité :
  ```bash
  python scripts/plot_der_density.py --input data/der_density.csv --output-dir plots/snir_stage1 --pdr-target 0.9
  ```
  Les figures PNG et PDF sont produites dans `plots/snir_stage1/`.

## Rappels sur la configuration radio
- Activer `flora_mode=True` pour appliquer les seuils et captures FLoRa (collisions inter-SF incluses via les presets).
- Utiliser `snir_model=True` et `interference_model=True` lorsque vous comparez baseline vs SNIR afin de prendre en compte le modèle d'interférences croisés.
- Les scénarios fournis forcent déjà le delta de non-orthogonalité FLoRa (`DEFAULT_NON_ORTH_DELTA`) sur l'ensemble des canaux et marquent chaque canal pour utiliser le SNIR.

## Préparation de l'Étape 2 (ne pas exécuter encore)
- Mettre en place les tracés pour `der_load`, `offered_traffic` et `throughput` afin d'exploiter les CSV générés.
- Préparer une campagne SNIR avec variations simultanées de `flora_mode`, `snir_model` et `interference_model` pour mesurer leur impact individuel.
- Étendre la documentation pour détailler la méthodologie de validation et les indicateurs à surveiller lors de l'étape suivante.
