# Vérification de la configuration des clusters QoS

## 1. Validation des paramètres dans le code source
- La dataclass `Cluster` définit un conteneur immuable pour l'identifiant, la proportion d'équipements, le taux d'arrivée et la cible PDR. 【F:loraflexsim/launcher/qos.py†L38-L105】
- `build_clusters` vérifie :
  - que `cluster_count` est strictement positif ;
  - que les séquences `proportions`, `arrival_rates` et `pdr_targets` ont toutes la longueur attendue ;
  - que chaque proportion et chaque taux d'arrivée est strictement positif ;
  - que la somme des proportions est ≈ 1 (tolérance relative/absolue `1e-6`) ;
  - que chaque cible PDR est dans l'intervalle ]0, 1].
  Les instances construites sont normalisées en `float`. 【F:loraflexsim/launcher/qos.py†L78-L106】
- `QoSManager.configure_clusters` délègue la validation à `build_clusters` puis réinitialise l'état interne lié aux reconfigurations. Les clusters validés sont exposés via `self.clusters`. 【F:loraflexsim/launcher/qos.py†L170-L190】

## 2. Couverture des tests unitaires
- `test_build_clusters_returns_normalised_instances` couvre le cas nominal avec deux clusters et vérifie la normalisation. 【F:tests/test_qos_clusters.py†L16-L26】
- `test_build_clusters_validates_lengths_and_values` couvre plusieurs scénarios d'échec :
  - longueurs incohérentes (proportions vs taux) ;
  - somme des proportions ≠ 1 ;
  - taux d'arrivée nul ;
  - cible PDR hors bornes. 【F:tests/test_qos_clusters.py†L29-L57】
- `test_qos_manager_configure_clusters` vérifie que `configure_clusters` renvoie les instances validées avec incrémentation des identifiants. 【F:tests/test_qos_clusters.py†L60-L69】
- Les tests fonctionnels (`test_qos_manager_computes_sf_limits_and_accessible_sets`, `test_mixra_opt_respects_duty_cycle_and_capacity`, `test_qos_reconfig_triggers_on_node_change_and_pdr_drift`) exploitent `configure_clusters` avec des données valides pour vérifier le comportement complet du gestionnaire QoS. 【F:tests/test_qos_clusters.py†L143-L389】
- Le module vérifie aussi l'export des métriques QoS (`test_simulator_metrics_expose_qos_statistics`), confirmant que la configuration des clusters se propage jusqu'au simulateur. 【F:tests/test_qos_clusters.py†L392-L487】

## 3. Points d'entrée pour les paramètres utilisateur
- **Interface graphique (`dashboard.py`)** : la zone QoS propose désormais le toggle principal, le choix de l'algorithme et les champs radio (activation SNIR, `α` inter-SF, seuils de capture). Ces contrôles sont masqués et ignorés tant que `qos_toggle` reste désactivé afin de conserver le profil ADR historique ; ils ne deviennent visibles/actifs qu'après activation du QoS. Les widgets cluster (proportions, taux, PDR) peuvent être ajoutés autour de la section « QoS » (après la déclaration de `qos_toggle` et `qos_algorithm_select`) et utilisés lors de l'appel à `qos_manager.apply`. 【F:loraflexsim/launcher/dashboard.py†L229-L764】【F:loraflexsim/launcher/dashboard.py†L1467-L1537】
- **Fichiers de configuration** : le mécanisme `setup_simulation` accepte un `config_file`, mais `config.ini` ne contient pas encore de section dédiée au QoS. Une extension de ce fichier ou d'un profil JSON/YAML permettrait d'injecter les paramètres dans `QoSManager.configure_clusters` avant l'appel à `qos_manager.apply`. 【F:loraflexsim/launcher/dashboard.py†L700-L763】
- **Scripts / CLI** : aucune invocation actuelle de `configure_clusters` en dehors des tests. Les scripts de lancement (par exemple dans `scripts/` ou via une future ligne de commande) devront collecter les paramètres utilisateur puis appeler explicitement `QoSManager.configure_clusters` avant de démarrer la simulation.
