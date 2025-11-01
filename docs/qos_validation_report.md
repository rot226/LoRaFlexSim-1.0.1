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

## Mode « validation » du banc QoS

Le banc `loraflexsim.scenarios.qos_cluster_bench` expose un mode `validation` qui force l'utilisation des couples de charges et de périodes décrits dans les figures de référence. Ce mode calcule et exporte un fichier `validation_normalized_metrics.json` contenant les DER, débits et écarts aux cibles normalisés par rapport aux maxima observés ainsi que les ratios PDR/objectif pour chaque cluster.【F:loraflexsim/scenarios/qos_cluster_bench.py†L18-L647】 Pour lancer la campagne complète :

```
python -m scripts.run_qos_cluster_bench --mode validation --seed 1 --quiet
```

Le chemin du fichier normalisé est rappelé à la fin de l'exécution.【F:scripts/run_qos_cluster_bench.py†L13-L120】 Les références officielles sont stockées dans `docs/qos_validation_reference.json` et peuvent être remplacées lorsque de nouvelles métriques sont validées publiquement.【F:docs/qos_validation_reference.json†L1-L38】

## Comparaison automatique avec les références

Le script `scripts/validate_qos_against_reference.py` charge les séries normalisées produites par le mode `validation`, les confronte aux valeurs de référence et échoue si un écart dépasse la tolérance configurée ou si un cluster passe sous sa cible PDR (après prise en compte de la tolérance).【F:scripts/validate_qos_against_reference.py†L1-L173】 Un test automatisé (`tests/test_qos_validation_script.py`) couvre le chemin nominal et vérifie qu'une dérive volontaire des ratios de PDR déclenche bien une erreur.【F:tests/test_qos_validation_script.py†L1-L54】

Exemple d'exécution :

```
python -m scripts.validate_qos_against_reference \
    --series results/qos_clusters/validation/validation_normalized_metrics.json \
    --reference docs/qos_validation_reference.json \
    --tolerance 0.05
```

## Mise à jour des valeurs de référence

1. Exécuter le banc en mode `validation` avec les seeds et solveurs retenus pour la publication et vérifier que les objectifs de clusters sont atteints.
2. Inspecter `validation_normalized_metrics.json`, copier les métriques normalisées souhaitées (par exemple celles utilisées dans les figures) puis mettre à jour `docs/qos_validation_reference.json` en conservant l'ordre des algorithmes pour limiter les conflits de fusion.
3. Documenter le changement et valider la suite de tests ; le test `test_validate_qos_against_reference_detects_cluster_gap` garantit que la tolérance ne masque pas une baisse de DER cluster.【F:tests/test_qos_validation_script.py†L24-L54】
4. Lorsque la tolérance doit évoluer (nouvelle figure, métriques plus volatiles), ajuster le paramètre `--tolerance` des jobs CI ou du script manuel, puis mettre à jour la valeur indicative `tolerance_hint` dans le fichier de référence.【F:scripts/validate_qos_against_reference.py†L129-L173】【F:docs/qos_validation_reference.json†L1-L38】

