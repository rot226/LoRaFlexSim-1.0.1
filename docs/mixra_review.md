# Revue des implémentations MixRA

## `_solve_mixra_greedy`
- Respecte les bornes supérieures en limitant chaque variable à `bounds[idx][1]` avant attribution.
- Applique les limites spécifiques SF/cluster via `sf_limits[(cluster_id, sf)]`.
- Tient compte du duty-cycle par canal en réduisant la part disponible selon `channel_remaining[channel_index]` lorsqu'un gestionnaire duty-cycle est actif.
- Considère la capacité propre au cluster via `cluster_remaining[cluster_id]`.

## `_apply_mixra_opt`
- Construit les contraintes d'égalité par cluster pour garantir que la somme des parts est égale à 1.
- Ajoute des contraintes d'inégalité pour les limites de capacité de cluster, les limites SF/cluster et le duty-cycle par canal.
- Normalise la solution par cluster pour éviter les dérives numériques.

## `_apply_mixra_h`
- Trie les clusters par `pdr_target` décroissant afin d'allouer en priorité ceux ayant les exigences les plus strictes.
- Promeut les équipements vers les SF accessibles supérieurs lorsque le SF courant n'est pas disponible.
- Applique `cluster_capacity_limits` en limitant la charge cumulée par canal lors de l'affectation.

Ces vérifications confirment la prise en compte des contraintes de bornes, de SF/canaux et de duty-cycle dans les implémentations MixRA.
