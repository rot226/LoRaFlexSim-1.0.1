# Interface CLI QoS

Ce répertoire centralise les fichiers nécessaires pour piloter les scénarios QoS via la future CLI.
Les scripts fournis (préfixés `lfs_`) orchestrent la préparation des données et l'analyse des résultats.
La commande `python -m qos_cli.lfs_run` exécute directement une simulation LoRaFlexSim pour un scénario donné.

## Flux de travail proposé

1. **Créer ou mettre à jour les scénarios**
   ```bash
   python qos_cli/lfs_make_scenarios.py --new
   ```
   Ce script prépare le fichier `qos_cli/scenarios.yaml`, mais **ne** génère **pas** `commands.txt`.
   Pour obtenir les commandes prêtes à l'emploi, il faut poursuivre avec l'étape suivante.

2. **Générer et examiner `commands.txt`**
   ```bash
   python qos_cli/lfs_print_commands.py --config qos_cli/scenarios.yaml > commands.txt
   ```
   Le fichier `commands.txt` contient désormais les appels prêts à l'emploi vers la CLI dédiée :
   `python -m qos_cli.lfs_run --scenario … --method … --out …`. Vérifiez la liste, adaptez les
   paramètres au besoin puis exécutez chaque ligne pour lancer automatiquement LoRaFlexSim avec la
   configuration correspondante.

3. **Collecter les métriques**
   Une fois toutes les simulations terminées et les résultats disponibles dans `results/`, exécutez :
   ```bash
   python qos_cli/lfs_metrics.py --in results/ --config qos_cli/scenarios.yaml
   ```
   Le script agrège les sorties de simulation et met à jour les fichiers de synthèse.

4. **Produire les graphiques**
   ```bash
   python qos_cli/lfs_plots.py --in results/ --config qos_cli/scenarios.yaml
   ```
   Cette étape exploite les métriques précédemment calculées pour générer les figures.

5. **Générer le rapport**
   ```bash
   python qos_cli/lfs_report.py --in results/ --summary qos_cli/SUMMARY.txt
   ```
   Le rapport final assemble métriques et visualisations en un document synthétique.

> ⚠️ Les scripts ci-dessus supposent que les résultats bruts de LoRaFlexSim sont accessibles et
> correctement renseignés. Ajustez les paramètres si nécessaire avant la première utilisation.

## Récapitulatif rapide

- Création/maintenance des scénarios : `python qos_cli/lfs_make_scenarios.py --new`
- Génération des commandes à exécuter : `python qos_cli/lfs_print_commands.py --config qos_cli/scenarios.yaml > commands.txt`
- Lancement des simulations : exécuter chaque ligne `python -m qos_cli.lfs_run …` listée dans `commands.txt`
- Agrégation des métriques : `python qos_cli/lfs_metrics.py --in results/ --config qos_cli/scenarios.yaml`
- Production des graphiques : `python qos_cli/lfs_plots.py --in results/ --config qos_cli/scenarios.yaml`
- Génération du rapport : `python qos_cli/lfs_report.py --in results/ --summary qos_cli/SUMMARY.txt`
