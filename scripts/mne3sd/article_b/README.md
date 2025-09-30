# Expériences de mobilité Article B

Ce dossier rassemble les scripts utilisés pour reproduire les expériences axées sur la mobilité de l'étude « Article B » de la campagne MNE3SD. La suite est organisée autour d'un petit ensemble de scénarios de mobilité canoniques et des utilitaires de post-traitement nécessaires pour transformer les sorties brutes du simulateur en figures prêtes à être publiées.

## Scénarios de mobilité

Tous les scénarios simulent des liens LoRaWAN de bout en bout avec des paramètres PHY identiques (SF10, bande passante 125 kHz, taux de codage 4/5) et un intervalle d'émission adaptatif recalculé après chaque livraison réussie. Les scénarios diffèrent par la configuration spatiale et les profils de mouvement de la flotte de nœuds mobiles :

### `urban_canyon`
- **Environnement** : îlots urbains denses avec des rues de 30 m de large et des stations de base installées à 25 m du sol.
- **Population de nœuds** : 120 traceurs piétons.
- **Plage de distances** : de 0,2 km à 2,5 km de la passerelle, échantillonnée uniformément le long du graphe de rues.
- **Profil de vitesse** : vitesses piétonnes tirées d'une loi gaussienne de moyenne 1,4 m/s et d'écart-type 0,4 m/s (tronquée dans l'intervalle \[0,5 ; 2,5\] m/s).
- **Gestion de l'effet Doppler** : cohérence de canal mise à jour toutes les 3 s pour capturer le fading rapide provoqué par les intersections.

### `rural_highway`
- **Environnement** : tronçon autoroutier de 8 km avec des passerelles en bord de route espacées de 2 km.
- **Population de nœuds** : 40 traceurs de véhicules circulant sur des voies prédéfinies.
- **Plage de distances** : trajets en ligne de vue de 0,5 km à 6 km avec des sections occasionnellement obstruées (20 % de probabilité NLOS).
- **Profil de vitesse** : distribution triangulaire culminant à 27 m/s et support \[15 ; 35\] m/s pour refléter la variabilité du trafic.
- **Modèle de handover** : les nœuds se réassocient à la passerelle la plus puissante dès que la puissance reçue reste sous –115 dBm pendant plus de 5 s.

### `industrial_campus`
- **Environnement** : site industriel de 1,5 km × 1,0 km avec obstacles métalliques et trois passerelles intérieures.
- **Population de nœuds** : 75 traceurs d'actifs alternant entre zones intérieures (60 %) et extérieures (40 %).
- **Plage de distances** : de 50 m à 1,8 km avec une distribution bimodale favorisant les sauts courts en intérieur.
- **Profil de vitesse** : vitesses par morceaux constantes tirées de \[0,1 ; 1,8\] m/s pour modéliser chariots élévateurs et personnel.
- **Ombrage** : shadowing log-normal d'écart-type 8 dB en intérieur et 4 dB en extérieur.

Chaque script de scénario stocke les paramètres dérivés (exposant de propagation, backoff de retransmission, etc.) dans des constantes de module afin de concentrer l'interface CLI sur des entrées reproductibles.

## Paramètres de simulation

Les points d'entrée des scénarios proposent les options suivantes :

- `--config` : chemin optionnel vers un fichier de configuration personnalisé pour le simulateur. Valeur par défaut : `config.ini`.
- `--seed` : graine aléatoire de base appliquée au simulateur. Chaque run répété incrémente la graine en interne pour garantir des lots Monte Carlo reproductibles.
- `--runs` : nombre de répétitions Monte Carlo. Par défaut : `20` pour les piétons (`urban_canyon`, `industrial_campus`), `10` pour les véhicules (`rural_highway`).
- `--duration` : horizon de simulation en secondes. Valeur par défaut : `7200` s.
- `--distance-min` / `--distance-max` : redéfinit les distances minimales et maximales du lien. En l'absence de ces options, les plages ci-dessus sont utilisées.
- `--speed-min` / `--speed-max` : redéfinit l'intervalle de vitesses autorisé par les générateurs aléatoires propres au scénario.
- `--output` : fichier CSV de destination dans `results/mne3sd/article_b/`.

Les modules de scénario peuvent proposer d'autres options (activation de la diversité passerelle, ajustement de la charge, etc.). Documentez-les dans les docstrings correspondantes.

## Utilitaires de tracé

Les modules de tracé sont de fines surcouches qui agrègent les CSV produits par les scénarios et génèrent les figures de l'Article B. L'interface partagée accepte :

- `--input` : un ou plusieurs fichiers CSV générés par les scripts de scénario. Répétez l'option pour traiter plusieurs jeux de données.
- `--figures-dir` : répertoire de destination des figures exportées. Valeur par défaut : `figures/mne3sd/article_b/`.
- `--format` : format de sortie (`png`, `pdf`, `svg`, …). Par défaut : `pdf`.
- `--style` : feuille de style Matplotlib optionnelle appliquée avant le rendu (par défaut `figures/matplotlib-paper.mplstyle` lorsqu'elle est disponible).

## Figures

### `plot_mobility_gateway_metrics`
- **Répartition du PDR par passerelle** (`figures/mne3sd/article_b/mobility_gateway/pdr_distribution_by_gateway/`) : graphique en barres empilées montrant la part de trafic collectée par chaque passerelle selon le modèle de mobilité et le nombre total de passerelles. Une répartition homogène signale une couverture bien équilibrée tandis qu'un segment dominant met en évidence un goulot d'étranglement.
- **Délai moyen downlink vs passerelles** (`figures/mne3sd/article_b/mobility_gateway/downlink_delay_vs_gateways/`) : courbes avec barres d'erreur reliant le nombre de passerelles au délai moyen des accusés de réception. La pente renseigne sur l'intérêt d'ajouter des passerelles supplémentaires pour réduire la latence downlink.
- **Comparaison RandomWaypoint/Smooth** (`figures/mne3sd/article_b/mobility_gateway/model_comparison/`) : nuage de points positionnant chaque configuration selon son PDR agrégé et son délai downlink moyen. Les annotations `n GW` rappellent le nombre de passerelles associé, ce qui aide à visualiser le compromis portée/latence entre les profils de mobilité.

### Profils d'exécution
Les lanceurs de scénarios respectent l'option `--profile` partagée ainsi que la variable d'environnement `MNE3SD_PROFILE` :

- `full` *(par défaut)* – exécute l'intégralité de la grille de scénarios avec les paramètres de recherche documentés.
- `ci` – réduit le nombre de nœuds, les plages de mobilité, les permutations de passerelles et les répétitions Monte Carlo afin d'accélérer les tests automatisés tout en produisant des résultats représentatifs.

### Parallélisation des réplicats
Les scripts `run_mobility_range_sweep.py`, `run_mobility_speed_sweep.py` et `run_mobility_gateway_sweep.py` acceptent un paramètre commun `--workers` (par défaut `1`) pour répartir les réplicats Monte Carlo sur plusieurs processus. Les résultats agrégés restent triés de manière déterministe quel que soit le nombre de workers, ce qui facilite la comparaison entre exécutions. En dehors des traitements lourds, conservez la valeur par défaut pour éviter un surcoût d'initialisation. Pour des vérifications rapides sous Windows 11 ou dans un pipeline CI, combinez `--workers 1` avec `--profile ci` afin de bénéficier des paramètres allégés documentés ci-dessus.

## Structure du répertoire

```
scripts/mne3sd/article_b/
├── README.md                # Ce guide
├── __init__.py              # Marqueur de package pour les utilitaires partagés
├── scenarios/               # Points d'entrée des scénarios
│   └── __init__.py
└── plots/                   # Points d'entrée pour les figures
    └── __init__.py
```

## Sorties attendues

### Artefacts CSV
Toutes les données brutes et métriques agrégées doivent résider dans `results/mne3sd/article_b/`. Utilisez des noms explicites comme `urban_canyon_runs.csv` ou `rural_highway_summary.csv`. Lorsqu'un scénario produit plusieurs fichiers (par exemple métriques par passerelle et traces par nœud), créez un sous-dossier : `results/mne3sd/article_b/industrial_campus/gateway_load.csv`.

Chaque CSV doit contenir au minimum les colonnes suivantes pour alimenter la chaîne de génération de figures :

- `scenario` : nom du module de scénario (ex. `urban_canyon`).
- `run` : indice de répétition Monte Carlo à partir de 0.
- `distance_m` : distance instantanée émetteur-récepteur.
- `speed_mps` : vitesse du nœud à l'instant échantillonné.
- `snr_db`, `rssi_dbm` : indicateurs de qualité de canal.
- `latency_s`, `delivery_ratio` : indicateurs clés utilisés dans les figures de l'Article B.

### Artefacts graphiques
Exportez les figures dans `figures/mne3sd/article_b/`, en gardant des noms alignés sur la numérotation du manuscrit (ex. `figure_3_latency_vs_speed.pdf`). Placez les graphiques temporaires de débogage dans un sous-dossier dédié tel que `figures/mne3sd/article_b/debug/` pour ne pas les mélanger aux figures finales.

## Exécution de la chaîne

Toutes les commandes doivent être lancées depuis la racine du dépôt. Remplacez `<scenario_module>` par le nom du fichier de scénario (sans l'extension `.py`) et `<figure_module>` par le module de tracé souhaité.

### Générer les données de simulation

```
python -m scripts.mne3sd.article_b.scenarios.<scenario_module> \
    --runs 20 \
    --duration 7200 \
    --seed 123 \
    --output results/mne3sd/article_b/<scenario_name>.csv
```

Pour balayer différentes plages de distance ou de vitesse, utilisez `--distance-min`, `--distance-max`, `--speed-min` et `--speed-max`. Les modules de scénario peuvent proposer des options additionnelles (par exemple `--handover-threshold` dans `rural_highway`). Consultez la docstring pour les détails.

### Générer les figures

```
python -m scripts.mne3sd.article_b.plots.<figure_module> \
    --input results/mne3sd/article_b/<scenario_name>.csv \
    --figures-dir figures/mne3sd/article_b/ \
    --format pdf
```

Répétez l'option `--input` pour combiner plusieurs jeux de données dans une seule figure. Utilisez `--style` afin d'assurer une mise en page homogène entre les figures.

### Chaîne complète

1. Exécutez chaque script de scénario requis jusqu'à remplir `results/mne3sd/article_b/` avec les CSV attendus.
2. Inspectez les sorties pour vérifier que les plages de distance et de vitesse correspondent à la configuration souhaitée.
3. Lancez les modules de tracé pour générer les figures finales dans `figures/mne3sd/article_b/`.
4. Relisez les figures exportées localement avant de les intégrer au manuscrit.

### Lanceur de batch

Vous pouvez exécuter toute la chaîne Article B en une seule commande grâce à `scripts/mne3sd/run_all_article_outputs.py` :

```
python -m scripts.mne3sd.run_all_article_outputs --article b
```

Cette commande orchestre tous les scénarios `run_mobility_*`, puis les modules `plot_*`, et se termine en affichant la liste des CSV et figures générés. Combinez-la avec `--skip-scenarios` ou `--skip-plots` lorsque seule une partie du workflow doit être régénérée.

Pour accélérer les itérations successives (par exemple lors de séries de tracés), ajoutez `--reuse` : chaque tâche vérifiera que toutes ses sorties existent et sont plus récentes que le script exécuté avant de lancer un nouveau calcul.

Maintenez ce README synchronisé avec les scripts disponibles lorsque de nouveaux scénarios ou graphiques sont ajoutés.
