# Expérimentations SNIR – Comparaison Étape 1

Ce dossier rassemble les scénarios, données et aides aux tracés utilisés pour comparer une configuration de référence FLoRa avec l'activation progressive du modèle SNIR.

## Structure
- `scenarios/` : scripts de simulation (baseline et variantes SNIR).
- `data/` : CSV générés par les scénarios.
- `plots_helpers/` : fonctions utilitaires pour les graphiques.
- `plots/` : sorties PNG/PDF produites par les scripts de tracé.

## Pré-requis
- Python 3.10 ou plus récent.
- Dépendances du projet installées en mode développement :
  ```bash
  pip install -e .
  ```

## Utilisation attendue
1. Ajouter les scripts de simulation dans `scenarios/` pour chaque variante (baseline, SNIR seul, SNIR + interférences).
2. Exécuter chaque script pour remplir `data/` avec des métriques DER/PDR et SNIR.
3. Centraliser les helpers matplotlib ou seaborn dans `plots_helpers/` pour harmoniser les figures.
4. Générer les graphiques comparatifs dans `plots/` (PNG et PDF) afin de préparer la présentation de l'étape 1.

## Points de contrôle
- Conserver la même graine et les mêmes paramètres de trafic entre baseline et variantes SNIR pour garantir la reproductibilité.
- Vérifier que `flora_mode`, `snir_model` et `interference_model` sont documentés dans chaque scénario.
- Inclure dans les légendes des figures la configuration exacte pour faciliter la lecture.
