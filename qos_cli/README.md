# Interface CLI QoS

Ce répertoire centralise les fichiers nécessaires pour piloter les scénarios QoS via la future CLI.
Les scripts fournis (préfixés `lfs_`) ne lancent **jamais** LoRaFlexSim automatiquement :
il est de la responsabilité de l'utilisateur d'exécuter la simulation manuellement entre les étapes.

## Flux de travail proposé

1. **Créer ou mettre à jour les scénarios**
   ```bash
   python scripts/lfs_make_scenarios.py --new
   ```
   Cette commande génère le fichier `commands.txt` contenant les instructions à exécuter manuellement.

2. **Examiner `commands.txt`**
   Vérifiez les commandes générées, ajustez-les si nécessaire puis exécutez LoRaFlexSim vous-même
   en vous référant aux instructions documentées par le projet.

3. **Collecter les métriques**
   Après exécution de LoRaFlexSim, lancez :
   ```bash
   python scripts/lfs_metrics.py
   ```
   Le script mettra à jour `SUMMARY.txt` et les métriques associées.

4. **Produire les graphiques**
   ```bash
   python scripts/lfs_plots.py
   ```
   Cette étape exploite les données produites par `lfs_metrics.py` pour générer les figures.

5. **Générer le rapport**
   ```bash
   python scripts/lfs_report.py
   ```
   Le rapport final assemble les métriques et visualisations collectées aux étapes précédentes.

> ⚠️ Les scripts ci-dessus supposent que les résultats bruts de LoRaFlexSim sont accessibles et
> correctement renseignés. Ajustez les paramètres si nécessaire avant la première utilisation.
