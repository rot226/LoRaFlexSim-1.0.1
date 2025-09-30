# Suite d'expériences Article A

## Objectifs
- Reproduire les simulations nécessaires à l'étude « Article A » de la campagne MNE3SD.
- Regrouper en un seul endroit les définitions de scénarios et les utilitaires réutilisables partagés dans l'étude.
- Collecter les métriques propres à chaque scénario sous forme de fichiers CSV et les post-traiter en figures prêtes pour la publication.

## Paramètres de simulation communs
Chaque script de scénario expose un ensemble cohérent d'options en ligne de commande pour faciliter la reproductibilité :

- `--config` : chemin d'un fichier de configuration optionnel pour la simulation, qui remplace les valeurs par défaut fournies avec le dépôt.
- `--seed` : graine aléatoire de base appliquée au simulateur. Les scripts peuvent en dériver d'autres graines.
- `--runs` : nombre de répétitions Monte Carlo à exécuter pour chaque configuration de scénario.
- `--duration` : durée de la simulation en secondes. Si l'option est absente, chaque script revient à sa valeur par défaut documentée.
- `--output` : fichier CSV de destination. Par convention il est placé dans `results/mne3sd/article_a/`.

Les scripts de `plots/` suivent la même logique :

- `--input` : un ou plusieurs fichiers CSV produits par les scripts de scénario.
- `--figures-dir` : dossier où les figures générées seront écrites. Valeur par défaut : `figures/mne3sd/article_a/`.
- `--format` : format d'image pour les graphiques exportés (par ex. `png`, `pdf`, `svg`).

### Profils d'exécution
Tous les lanceurs de scénarios acceptent l'option commune `--profile` (ou la variable d'environnement `MNE3SD_PROFILE`) pour basculer entre des presets :

- `full` *(valeur par défaut)* – conserve les paramètres de publication décrits dans chaque script.
- `fast` – limite le nombre de nœuds à 150 et réduit le volume de paquets/répétitions pour accélérer les itérations locales. C'est le réglage conseillé pour des itérations rapides sous Windows 11.
- `ci` – réduit le nombre de nœuds, de répétitions et l'étendue des paramètres explorés afin d'accélérer les tests automatisés et les vérifications rapides, tout en exerçant l'intégralité de la chaîne.

## Arborescence et artefacts
```
scripts/mne3sd/article_a/
├── README.md                # Ce guide
├── __init__.py              # Marqueur de package pour les utilitaires partagés
├── scenarios/               # Points d'entrée pour la génération de données
│   └── __init__.py
└── plots/                   # Points d'entrée pour la génération de figures
    └── __init__.py
```

### Sorties CSV
Toutes les métriques brutes ou agrégées produites par les expériences doivent être stockées dans `results/mne3sd/article_a/`. Chaque script de scénario devrait créer un sous-dossier dédié lorsqu'il écrit plusieurs fichiers, par exemple `results/mne3sd/article_a/urban/summary.csv`. Les utilitaires de prétraitement partagés peuvent également conserver des CSV intermédiaires dans la même arborescence.

### Figures
Utilisez `figures/mne3sd/article_a/` pour stocker toute figure exportée pour l'Article A. Préférez des noms de fichiers explicites correspondant au manuscrit, par exemple `figure_2_packet_delivery.pdf`. Les artefacts intermédiaires (tels que les graphiques de débogage) peuvent être placés dans un sous-dossier dédié ignoré lors de la rédaction finale.

## Exécution des scripts
Toutes les commandes ci-dessous doivent être lancées depuis la racine du dépôt. Remplacez les éléments entre chevrons par des valeurs propres au scénario.

### Générer les données de simulation
```
python -m scripts.mne3sd.article_a.scenarios.<scenario_module> \
    --runs 10 \
    --duration 3600 \
    --seed 42 \
    --output results/mne3sd/article_a/<scenario_name>.csv
```

Chaque module de scénario peut accepter des options supplémentaires (par exemple pour ajuster la topologie, les profils de trafic ou les paramètres PHY). Documentez les options spécifiques directement dans la docstring du script.

### Générer les figures
```
python -m scripts.mne3sd.article_a.plots.<figure_module> \
    --input results/mne3sd/article_a/<scenario_name>.csv \
    --figures-dir figures/mne3sd/article_a/ \
    --format pdf
```

Les modules de tracé peuvent agréger plusieurs fichiers CSV en répétant l'option `--input`. Adoptez des noms de module descriptifs tels que `throughput_breakdown` ou `sensitivity_overview` pour rester aligné avec la structure de l'article.

## Rejouer toute la chaîne
Pour exécuter le flux complet de bout en bout :

1. Lancez tous les modules de scénario requis afin d'alimenter `results/mne3sd/article_a/`.
2. Vérifiez les fichiers CSV générés et, si nécessaire, validez-les dans une branche séparée pour assurer leur traçabilité.
3. Exécutez chaque module de tracé pour remplir `figures/mne3sd/article_a/`.
4. Passez en revue les figures localement avant de les exporter vers le dépôt du manuscrit.

### Lanceur de batch

Pour réaliser toute la chaîne Article A en une seule étape, utilisez le lanceur partagé `scripts/mne3sd/run_all_article_outputs.py` :

```
python -m scripts.mne3sd.run_all_article_outputs --article a
```

Ce script enchaîne toutes les commandes `run_class_*`, puis les modules `plot_*`, et affiche un résumé des CSV et figures générés. Utilisez `--skip-scenarios` ou `--skip-plots` pour limiter l'exécution à une seule étape, par exemple lorsque seules les figures doivent être régénérées à partir de données existantes.

Lorsque vous répétez des séries de tracés, pensez à ajouter `--reuse` : les tâches dont toutes les sorties sont déjà présentes et plus récentes que le script associé seront alors ignorées, ce qui accélère significativement les itérations successives.

Gardez ce README à jour au fur et à mesure que de nouveaux scénarios ou graphiques sont ajoutés afin de garantir une utilisation homogène entre collaborateur·rice·s.
