# Article C

Ce dossier contient une structure minimale pour les scripts de l'article C.

## Organisation

- `common/` : modules utilitaires partagés.
- `step1/` : scripts de la première étape.
- `step2/` : scripts de la seconde étape.
- `run_all.py` : exécute toutes les étapes.
- `make_all_plots.py` : génère tous les graphes disponibles.

## Modèle radio et SNIR

- **Modèle radio (proxy)** : l'étape 1 génère des nœuds avec des niveaux SNR/RSSI aléatoires et applique des seuils par SF pour estimer la QoS, puis approxime les collisions via une capacité par SF (proxy de charge). Les algorithmes ADR/MixRA sont des heuristiques simplifiées pour produire des valeurs reproductibles.
- **Modèle d'interférences** : le calcul considère les transmissions **co‑SF** sur le **même canal**; l'interférence agrégée est la somme des puissances reçues des transmissions simultanées, à laquelle on ajoute le bruit thermique pour former le dénominateur du SNIR. Il n'y a pas d'interférences inter‑SF ni de canaux adjacents dans ce proxy.
- **SNIR OFF** : la réception est validée uniquement si le RSSI est au-dessus du seuil de sensibilité (pas d'impact des interférences dans la décision).
- **SNIR ON** : le SNIR est calculé à partir de la somme des interférences co‑SF sur le même canal (interférence + bruit) et la réception dépend à la fois du RSSI et du seuil de capture SNIR.

## UCB1 et fonction de récompense

- **UCB1 (UCB1‑SF)** : l'agent sélectionne un SF via un warm‑up initial puis le score UCB1 classique (moyenne + terme d'exploration), avec mise à jour incrémentale de la moyenne des récompenses.
- **Récompense** : pour chaque fenêtre, on calcule une récompense bornée dans \[0, 1\] selon

  ```text
  reward = success_rate * bitrate_norm - lambda_energy * energy_norm
  ```

  où `success_rate` est le taux de succès dans la fenêtre, `bitrate_norm` la normalisation du débit, et `energy_norm` une normalisation de l'énergie.

## Exécution (Windows 11)

> Les scripts sont **100 % offline** : ils ne téléchargent rien et n'appellent aucun service réseau.

### Installation minimale

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r article_c/requirements.txt
```

### Commandes CLI exactes

Exécuter toutes les étapes :

```powershell
python -m article_c.run_all
```

Exécuter toutes les étapes en sortie **flat** + générer les figures (exemple Windows 11) :

```powershell
python -m article_c.run_all --flat-output
python -m article_c.make_all_plots
```

Exécuter toutes les étapes en sautant l'étape 1 :

```powershell
python -m article_c.run_all --skip-step1
```

Exécuter toutes les étapes en sautant l'étape 2 :

```powershell
python -m article_c.run_all --skip-step2
```

> Si l'étape 1 bloque, tester `--skip-step1` pour lancer directement l'étape 2.
> **Note** : `make_all_plots` nécessite les résultats de l'étape 1 ; utiliser `--skip-step1` empêchera donc la génération des figures.

Désactiver le fallback d'optimisation MixRA (option CLI de `run_all`) :

```powershell
python -m article_c.run_all --mixra-opt-no-fallback
```

Exécuter uniquement l'étape 1 :

```powershell
python article_c/step1/run_step1.py --network-sizes 50 100 150 --replications 5 --seeds_base 1000 --snir_modes snir_on,snir_off
```

Exécuter uniquement l'étape 2 :

```powershell
python article_c/step2/run_step2.py --network-sizes 50 100 150 --replications 5 --seeds_base 1000
```

### Diagnostic d'import (package `article_c`)

Si vous avez un doute sur la résolution du package `article_c`, vous pouvez lancer
le script de diagnostic suivant pour vérifier l'import et afficher le chemin résolu :

```powershell
python article_c/diagnose_import.py
```

Le script affiche également un extrait de `sys.path` pour aider à comprendre
la résolution des modules sur Windows 11.

### Jitter (décalage temporel)

Le **jitter** ajoute un décalage aléatoire (uniforme) à chaque instant de transmission généré. L'amplitude est contrôlée par `--jitter-range-s` (secondes) et s'applique aux modèles de trafic périodique ou poisson, en conservant uniquement les transmissions qui restent dans la fenêtre de simulation. La valeur par défaut est **30 s**.

Exemple CLI avec un jitter explicite :

```powershell
python article_c/step2/run_step2.py --network-sizes 50 100 150 --replications 5 --seeds_base 1000 --jitter-range-s 30
```

### Paramètres avancés de collisions et congestion (étape 2)

Ces options permettent d'ajuster finement les pertes dues aux collisions/congestion :

- `--capture-probability` : probabilité qu'une collision laisse un émetteur survivre. Valeur conseillée **0.08–0.15** (défaut 0.12).
- `--congestion-coeff-base` : coefficient de base de la probabilité de congestion. Valeur conseillée **0.25–0.40** (défaut 0.32).
- `--congestion-coeff-growth` : vitesse de croissance avec la surcharge. Valeur conseillée **0.25–0.50** (défaut 0.35).
- `--congestion-coeff-max` : plafond de la probabilité de congestion. Valeur conseillée **0.25–0.40** (défaut 0.35).
- `--collision-size-factor` : facteur de taille appliqué aux collisions (si non défini, calcul automatique). Valeur conseillée **0.8–1.6** selon la densité.

Exemple CLI avec ajustement des coefficients :

```powershell
python article_c/step2/run_step2.py --network-sizes 50 100 150 --replications 5 --seeds_base 1000 --capture-probability 0.12 --congestion-coeff-base 0.32 --congestion-coeff-growth 0.35 --congestion-coeff-max 0.35
```

Générer toutes les figures :

```powershell
python -m article_c.make_all_plots
```

Contrôler les formats d'export (PNG/PDF/EPS) :

```powershell
python -m article_c.make_all_plots --formats png,pdf,eps
```

Régénérer toutes les figures (Windows 11) :

```powershell
Remove-Item -Recurse -Force article_c/step1/plots/output, article_c/step2/plots/output
python -m article_c.make_all_plots
```

> **Windows (py -m recommandé)** : si `article_c` n’est **pas** reconnu comme un package
> (absence de `__init__.py` ou appel depuis un répertoire inadéquat), la commande
> `python article_c/run_all.py` peut échouer. Préférez toujours l’appel en module
> `python -m article_c.run_all` pour garantir la résolution correcte des imports.

## Seeds et réplications

- **Seeds** : l'étape 1 et l'étape 2 utilisent `--seeds_base` (ex: `--network-sizes 100 200 --replications 2 --seeds_base 123`) pour initialiser un seed de base déterministe, puis incrémentent en interne.
- **Réplications** : `--replications` définit le nombre de répétitions par configuration (taille de réseau/algorithme/mode SNIR).

> **Network size = number of nodes (integer)**.

Les résultats sont écrits dans `article_c/step*/results/` et les figures dans `article_c/step*/plots/output/`.

## Résultats

Les scripts écrivent les CSV dans deux formats :

- **Format imbriqué (nested)** : chaque taille/réplication écrit dans
  `article_c/step*/results/size_<N>/rep_<R>/` (`raw_metrics.csv` ou
  `raw_results.csv`, plus un `aggregated_results.csv` local par réplication).
- **Format flat** : les CSV sont directement au niveau de
  `article_c/step*/results/` (`raw_metrics.csv` ou `raw_results.csv` + un
  `aggregated_results.csv` global).

Les **plots de synthèse** (ex. courbes globales) et `validate_results.py`
attendent des CSV **flat** (présence de `aggregated_results.csv` dans
`article_c/step*/results/`). Pensez à activer `--flat-output` sur `run_all.py`,
`run_step1.py` et `run_step2.py` pour écrire ces fichiers directement.

Si vous avez uniquement un format imbriqué, `make_all_plots.py` peut servir de
**fallback d'agrégation** : il reconstitue les `aggregated_results.csv` flat à
partir des sous-dossiers avant de lancer les figures.

## Légendes IEEE‑ready

### Tailles recommandées (IEEE)

Pour éviter tout redimensionnement destructif lors de la mise en page IEEE, privilégier des tailles de figure proches des largeurs finales :

- **Colonne simple** : ~**3.5 in** (≈ 8.9 cm) de large.
- **Double colonne** : ~**7.16 in** (≈ 18.2 cm) de large.
- **Hauteur** : typiquement **2.2–3.5 in** (≈ 5.6–8.9 cm) selon le contenu.

Ces tailles permettent de conserver des polices lisibles et des épaisseurs de traits cohérentes dans le PDF final.

### Gestion dynamique des légendes

- **Légende en haut** : positionner systématiquement la légende en partie haute de la figure.
- **Marges réservées** : laisser une marge supérieure dédiée à la légende pour éviter le chevauchement avec le tracé.
- **Toujours visible** : afficher la légende même si une métrique est constante (pas d'auto‑masquage).
- **Placement adaptatif** : ajuster automatiquement le nombre de colonnes et l'espacement pour conserver une légende lisible quand le nombre de séries varie.

### Export EPS

- **Format EPS** : activer l'export EPS pour les soumissions IEEE qui exigent des figures vectorielles.
- **CLI** : utiliser `--formats png,pdf,eps` avec `make_all_plots.py`.

## Figures disponibles

Les scripts listés ci‑dessous génèrent les figures de chaque étape. Les courbes par cluster sont produites par les scripts « cluster_* » et s'appuient sur des CSV contenant une colonne `cluster` pour filtrer/agréger les séries.

### Étape 1

- `plot_S1.py`
- `plot_S2.py`
- `plot_S3.py`
- `plot_S4.py`
- `plot_S5.py`
- `plot_S6.py`
- `plot_S6_cluster_pdr_vs_density.py` (nouvelle)
- `plot_S6_cluster_pdr_vs_network_size.py` (nouvelle)
- `plot_S7_cluster_outage_vs_density.py` (nouvelle)
- `plot_S7_cluster_outage_vs_network_size.py` (nouvelle)

### Étape 2

- `plot_RL1.py`
- `plot_RL2.py`
- `plot_RL3.py`
- `plot_RL4.py`
- `plot_RL5.py`
- `plot_RL6_cluster_outage_vs_density.py` (nouvelle)
- `plot_RL7_reward_vs_density.py` (nouvelle)
