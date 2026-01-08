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

   ### Balayer automatiquement les valeurs (`--sweep`)

   Il est désormais possible de générer une grille complète de scénarios supplémentaires sans modifier le
   fichier principal. Les options `--sweep` décrivent chaque dimension à explorer et la combinaison
   cartésienne est automatiquement transformée en un nouveau fichier `qos_cli/scenarios_sweep.yaml`
   (modifiable via `--sweep-out`).

   ```bash
   python qos_cli/lfs_make_scenarios.py \
       --sweep N=60,120,180 \
       --sweep period=120,300,600
   ```

   Chaque combinaison produit un identifiant unique (`S_N60_T600`, etc.) issu du gabarit interne
   `_make_scenario`. Les valeurs de `evaluation.cluster_targets` sont recalculées automatiquement de sorte
   à rester cohérentes avec les cibles de chaque cluster. La console récapitule systématiquement les
   paramètres retenus ainsi que le chemin d'écriture.

   Pour injecter une table de combinaisons personnalisée (par exemple issue d'Excel), utilisez un CSV dont
   les en-têtes correspondent aux champs du scénario :

   ```bash
   python qos_cli/lfs_make_scenarios.py --sweep-csv docs/mes_scenarios.csv
   ```

   Les valeurs sont analysées via YAML, ce qui permet d'indiquer des booléens ou des nombres flottants. Les
   fichiers générés peuvent ensuite être utilisés directement avec les scripts de post-traitement (`lfs_print_commands.py`,
   `lfs_metrics.py`, `lfs_plots.py`, `lfs_plots_surfaces.py`, `lfs_plots_scatter.py`, etc.) en spécifiant le
   chemin du YAML de balayage via l'option `--config`.

2. **Générer et examiner `commands.txt`**
   ```bash
   python qos_cli/lfs_print_commands.py --config qos_cli/scenarios.yaml > commands.txt
   ```
   Le fichier `commands.txt` contient désormais les appels prêts à l'emploi vers la CLI dédiée :
   `python -m qos_cli.lfs_run --scenario … --method … --out …`. Vérifiez la liste, adaptez les
   paramètres au besoin puis exécutez chaque ligne pour lancer automatiquement LoRaFlexSim avec la
   configuration correspondante.

   ### Forcer l'état SNIR (CLI ou YAML)

   Chaque scénario peut forcer l'état SNIR via `snir_enabled: true|false` ou `propagation.phy_profile`
   dans `qos_cli/scenarios.yaml`. Les options CLI surchargent ensuite la valeur YAML.

   Exemple pour le même scénario et la même graine :
   ```bash
   python -m qos_cli.lfs_run --scenario S0 --method MixRA_Opt --seed 42 \
       --snir on --out results/S0/MixRA_Opt/snir_on_seed42

   python -m qos_cli.lfs_run --scenario S0 --method MixRA_Opt --seed 42 \
       --disable-snir --out results/S0/MixRA_Opt/snir_off_seed42
   ```

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
   Cette étape exploite les métriques précédemment calculées pour générer les figures suivantes (dans `qos_cli/figures/` par défaut) :
   - `pdr_clusters_vs_scenarios.png` : PDR détaillé par cluster pour chaque méthode.
   - `pdr_global_vs_scenarios.png` : PDR global (livraisons ÷ tentatives) par scénario et par méthode.
   - `der_global_vs_scenarios.png` : taux de livraison descendant (DER) par scénario.
   - `pdr_global_vs_nodes.png` : PDR global en fonction du nombre de nœuds (par méthode).
   - `der_global_vs_nodes.png` : DER global en fonction du nombre de nœuds (par méthode).
   - `pdr_clusters_vs_nodes.png` : PDR par cluster en fonction du nombre de nœuds (par méthode).
   - `collisions_vs_scenarios.png` : nombre total de collisions montantes observées par scénario.
   - `collision_rates_vs_nodes.png` : taux de collisions destructives/capture en fonction du nombre de nœuds.
   - `energy_total_vs_scenarios.png` : énergie totale consommée par les nœuds (somme en joules) par scénario.
   - `energy_per_node_vs_pdr.png` : énergie moyenne par nœud vs PDR global.
   - `energy_per_node_vs_snir.png` : énergie moyenne par nœud vs SNIR moyen.
   - `jain_index_vs_scenarios.png` : indice de Jain pour qualifier l’équité de la distribution de trafic.
   - `min_sf_share_vs_scenarios.png` : part de nœuds utilisant le facteur d’étalement minimal.
   - `snir_mean_by_sf_vs_nodes.png` : SNIR moyen par SF en fonction du nombre de nœuds (par méthode).
   - `snir_mean_by_cluster_vs_nodes.png` : SNIR moyen par cluster en fonction du nombre de nœuds (par méthode).
   - `snir_cdf_<scénario>.png` : courbes CDF du SNIR pour chaque scénario (une image par scénario).
   - `pdr_vs_snir_<scénario>.png` : PDR vs SNIR (bins SNIR), une courbe par méthode.
   - `pdr_vs_snir_<scénario>_cluster_<cluster>.png` : PDR vs SNIR par cluster, une courbe par méthode.
   - `effective_sf_vs_distance.png` : SF effectif moyen en fonction de la distance au GW (ADR vs SNIR-aware).

   Pour compléter ces courbes par une visualisation surfacique des performances (nombre de nœuds × période), utilisez la CLI dédiée :
   ```bash
   python -m qos_cli.lfs_plots_surfaces --in results/ --config qos_cli/scenarios.yaml --out qos_cli/figures
   ```
   Le script crée, pour chaque méthode, des heatmaps du PDR global, du DER et de l’écart à la cible dans le dossier de sortie (fichiers `pdr_heatmap_<méthode>.png`, `der_heatmap_<méthode>.png`, `target_gap_heatmap_<méthode>.png`).

   Pour explorer les corrélations entre métriques (ex. énergie vs PDR, collisions vs indice de Jain), un nouveau module permet de générer des nuages de points paramétrables :
   ```bash
   python -m qos_cli.lfs_plots_scatter --in results/ --config qos_cli/scenarios.yaml \
       --x energy_per_delivery --y pdr_global --color collision_rate --connect --annotate
   ```
   Les options `--x`, `--y` et `--color` acceptent toute métrique numérique (y compris `cluster_pdr:<id>` ou `pdr_gap_by_cluster:<id>`). Les lignes de tolérance sont automatiquement ajoutées pour les axes PDR et collision, et des avertissements `[WARN]` signalent les données manquantes.

5. **Générer le rapport**
   ```bash
   python qos_cli/lfs_report.py --in results/ --summary qos_cli/SUMMARY.txt
   ```
   Le rapport final assemble métriques et visualisations en un document synthétique.

> ⚠️ Les scripts ci-dessus supposent que les résultats bruts de LoRaFlexSim sont accessibles et
> correctement renseignés. Ajustez les paramètres si nécessaire avant la première utilisation.

## Récapitulatif rapide

- Création/maintenance des scénarios : `python qos_cli/lfs_make_scenarios.py --new`
- Génération d'une grille de balayage : `python qos_cli/lfs_make_scenarios.py --sweep … --sweep-out …`
- Génération des commandes à exécuter : `python qos_cli/lfs_print_commands.py --config qos_cli/scenarios.yaml > commands.txt`
- Lancement des simulations : exécuter chaque ligne `python -m qos_cli.lfs_run …` listée dans `commands.txt`
- Agrégation des métriques : `python qos_cli/lfs_metrics.py --in results/ --config qos_cli/scenarios.yaml`
- Production des graphiques : `python qos_cli/lfs_plots.py --in results/ --config qos_cli/scenarios.yaml`
- Cartes de chaleur des performances : `python -m qos_cli.lfs_plots_surfaces --in results/ --config qos_cli/scenarios.yaml --out qos_cli/figures`
- Nuages de points corrélés : `python -m qos_cli.lfs_plots_scatter --in results/ --config qos_cli/scenarios.yaml --x … --y …`
- Génération du rapport : `python qos_cli/lfs_report.py --in results/ --summary qos_cli/SUMMARY.txt`
