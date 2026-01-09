# README_DEBUG

Ce document résume les constats et correctifs liés aux métriques et aux modes de collision.

## Où PDR/DER sont calculés
- **Simulator** : calculs côté simulateur (agrégation des paquets transmis/reçus).
- **qos_cluster_bench** : agrégation des sorties d’exécution pour les campagnes QoS.
- **metrics.py** : post-traitement final et calcul des métriques exportées.

## Pourquoi les collisions étaient désactivées
- **pure_poisson_mode** : les émissions sont générées de manière indépendante, ce qui peut masquer des interactions réalistes.
- **min_interference_time** : seuil trop permissif qui écarte des collisions si la fenêtre d’interférence est jugée trop courte.

## Pourquoi SNIR_ON/OFF était identique
- La **capture avancée** était activée sans condition, appliquant le même traitement quel que soit l’état de SNIR (ON/OFF).

## Actions correctives et paramètres proposés
1. **Réactiver les collisions** en désactivant le mode poisson pur pour les expériences réalistes.
2. **Ajuster la fenêtre d’interférence** : définir `min_interference_time` à une valeur cohérente avec la durée de symbole/paquet.
3. **Conditionner la capture avancée** : n’appliquer la capture que lorsque SNIR est explicitement activé.
4. **Harmoniser les scripts QoS** : vérifier que `qos_cluster_bench` et `metrics.py` utilisent les mêmes conventions de comptage.

### Paramètres proposés (point de départ)
- `pure_poisson_mode = false`
- `min_interference_time = 1 * symbol_time`
- `enable_advanced_capture = (SNIR == ON)`

## Checklist rapide des sanity checks
- [ ] PDR/DER identiques entre simulator et post-traitement (écart < 1%).
- [ ] Différence observable entre SNIR_ON et SNIR_OFF sur un scénario chargé.
- [ ] Collisions non nulles sur un scénario dense.
- [ ] Résultats stables sur 3 exécutions consécutives (variance limitée).
- [ ] Scripts QoS et metrics exportent les mêmes colonnes de sortie.
