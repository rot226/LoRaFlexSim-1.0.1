#!/usr/bin/env bash
# À exécuter une fois toutes les commandes listées dans commands.txt terminées.
# Sous Windows PowerShell, exécuter : python qos_cli/lfs_metrics.py --in results/ --config qos_cli/scenarios_small.yaml
# Si le fichier n'est pas exécutable, lancer : bash extract_metrics.sh
python qos_cli/lfs_metrics.py --in results/ --config qos_cli/scenarios_small.yaml
