# Figures Step 1

Ce document décrit les figures générées par `scripts/plot_step1_comparison.py`.
Les titres, axes et légendes sont en anglais afin d'être directement publiables.

## Figure 1 — PDR and DER vs Nodes

- **Structure** : grille 3×2 (lignes = clusters, colonnes = SNIR OFF/ON).
- **Contenu** : pour chaque cluster, les courbes montrent les moyennes PDR (plein)
  et DER (pointillé) en fonction du nombre de nœuds.
- **Incertitude** : les barres d'erreur représentent l'intervalle de confiance à 95 %.

Fichiers exportés :
- `figures/step1/step1_pdr_der_comparison.png`
- `figures/step1/step1_pdr_der_comparison.pdf`

## Figure 2 — Jain Index vs Nodes

- **Structure** : 1×2 (SNIR OFF/ON).
- **Contenu** : indice de Jain en fonction du nombre de nœuds, pour 4 algorithmes.
- **Incertitude** : intervalle de confiance à 95 %.

Fichiers exportés :
- `figures/step1/step1_jain_comparison.png`
- `figures/step1/step1_jain_comparison.pdf`

## Figure 3 — Throughput vs Nodes

- **Structure** : 1×2 (SNIR OFF/ON).
- **Contenu** : débit (bps) en fonction du nombre de nœuds, pour 4 algorithmes.
- **Incertitude** : intervalle de confiance à 95 %.

Fichiers exportés :
- `figures/step1/step1_throughput_comparison.png`
- `figures/step1/step1_throughput_comparison.pdf`
