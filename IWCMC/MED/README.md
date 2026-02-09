# SNIR‑Aware Resource Allocation in LoRaWAN with Reinforcement Learning under Mobility

Ce dossier regroupe les artefacts liés à la section MED pour IWCMC.

## Structure

- `simulations/` : configurations et sorties brutes des simulations MED.
- `figures/` : figures finales utilisées dans le manuscrit (MED1 à MED8).
- `scripts/` : scripts de génération, post-traitement et export des figures MED.
- `data/` : jeux de données intermédiaires utilisés par les scripts.

## Scripts (conventions)

Chaque script déposé dans `scripts/` doit préciser en en‑tête :

- l’objectif du script,
- les entrées attendues (dossier ou fichiers dans `data/` ou `simulations/`),
- les sorties générées (fichiers écrits dans `figures/`).

Nommer les scripts selon l’action principale et la figure cible, par exemple :

- `build_med1_traffic_profile.py`
- `plot_med4_snir_cdf.py`

## Nomenclature des figures MED

Les figures finales suivent la convention `MED<n>.svg` où `<n>` est un numéro de 1 à 8.

| Fichier | Contenu prévu |
| --- | --- |
| `MED1.svg` | Profil de mobilité / scénario (à préciser). |
| `MED2.svg` | Allocation des ressources vs mobilité (à préciser). |
| `MED3.svg` | SNIR vs temps / distance (à préciser). |
| `MED4.svg` | CDF SNIR / PDR (à préciser). |
| `MED5.svg` | Récompense RL vs itérations (à préciser). |
| `MED6.svg` | Répartition des SF / canaux (à préciser). |
| `MED7.svg` | Comparaison baselines (à préciser). |
| `MED8.svg` | Sensibilité paramètres (à préciser). |

Mettre à jour la colonne « Contenu prévu » dès qu’un intitulé définitif est connu.
