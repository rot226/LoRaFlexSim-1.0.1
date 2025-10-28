# Rapport de validation QoS

## Revue des tests existants

La suite `tests/test_qos_clusters.py` couvre désormais la tolérance aux petites erreurs d'arrondi lors de la définition des proportions de clusters. Un test dédié vérifie qu'une somme flottante légèrement imprécise est acceptée et normalisée avant d'alimenter le gestionnaire QoS.【F:tests/test_qos_clusters.py†L16-L37】

## Matrice de topologies supplémentaires

Une matrice paramétrée évalue trois topologies contrastées : un réseau compact mono-cluster, un déploiement bi-cluster équilibré et un scénario dense tri-cluster multi-canaux. Chaque configuration exerce la répartition des nœuds, la génération des trafics offerts et le calcul des airtimes afin de garantir une couverture multi-cluster et des variations de taux d'arrivée (λ).【F:tests/test_qos_clusters.py†L732-L828】

## Comparaison QoS vs ADR

Le script `scripts/run_qos_comparison.py` a été exécuté avec 40 nœuds, six paquets par nœud et un intervalle moyen de 30 s afin de confronter le profil QoS (MixRA-Opt) à une ligne de base sans QoS. Les métriques agrégées montrent un PDR global de 0,74 pour la configuration QoS contre 0,82 pour la référence ADR désactivée. Le débit moyen passe de 74,4 bps à 67,9 bps et l'indice de Gini de débit augmente de 0 à 0,13, indiquant que cette configuration QoS ne respecte pas encore les objectifs PDR ni les attentes de gains de débit/équité, ce qui motive un recalibrage des paramètres de cluster.【F:results/qos_comparison/summary.json†L1-L20】【F:results/qos_comparison/qos_enabled_metrics.json†L1-L119】

## Lignes de base ADR, APRA et AIMI

`QoSManager` expose désormais trois clés lisibles : `"ADR-Pure"`, `"APRA-like"` et `"Aimi-like"`. `ADR-Pure` délègue à l'implémentation ADR historique (ADR1 ou ADR Max) après avoir remis à zéro les structures internes, tandis que `APRA-like` et `Aimi-like` appliquent respectivement un ordonnancement sans découpage des canaux et une partition prioritaire de canaux avec promotion de SF en cas de saturation locale.【F:loraflexsim/launcher/qos.py†L31-L37】【F:loraflexsim/launcher/qos.py†L748-L1032】

Ces scénarios sont couverts par de nouveaux tests unitaires qui valident l'absence de collisions de configuration, le respect du duty-cycle et la répartition par priorité des canaux. Les cas de test introduits documentent également le comportement attendu pour chaque stratégie.【F:tests/test_qos_clusters.py†L876-L969】

Depuis le banc d'essai ou les scripts d'expérience, il suffit d'instancier un `QoSManager` puis d'appeler `manager.apply(simulateur, "ADR-Pure")`, `manager.apply(simulateur, "APRA-like")` ou `manager.apply(simulateur, "Aimi-like")` après la configuration des clusters pour sélectionner la ligne de base désirée.【F:loraflexsim/launcher/qos.py†L149-L174】【F:loraflexsim/launcher/qos.py†L748-L1032】

