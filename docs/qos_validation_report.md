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

## Rafraîchissement automatique du contexte QoS

Le simulateur déclenche désormais automatiquement la réallocation QoS dès qu'une dérive mesurée dépasse les seuils configurés ou qu'un nouveau nœud rejoint le réseau. Un événement périodique (`EventType.QOS_RECONFIG`) est planifié après chaque application de l'algorithme, selon l'intervalle `reconfig_interval_s`; lorsque le délai est atteint, `Simulator.request_qos_refresh` réévalue la nécessité de recalculer le contexte en respectant `_should_refresh_context`.【F:loraflexsim/launcher/simulator.py†L40-L48】【F:loraflexsim/launcher/simulator.py†L1014-L1073】【F:loraflexsim/launcher/qos.py†L160-L192】

Les mises à jour de PDR récentes déclenchent aussi cette routine : chaque traitement d'uplink compare la variation de `recent_pdr` aux seuils `pdr_drift_threshold` et `traffic_drift_threshold` avant de rappeler `QoSManager.apply` si nécessaire.【F:loraflexsim/launcher/simulator.py†L1495-L1503】【F:loraflexsim/launcher/qos.py†L479-L563】 Enfin, le serveur réseau notifie le simulateur lors des acceptations OTAA ou activations internes afin de recalculer immédiatement les ressources après l'arrivée d'un nœud.【F:loraflexsim/launcher/server.py†L522-L532】【F:loraflexsim/launcher/server.py†L713-L731】【F:loraflexsim/launcher/server.py†L880-L890】

