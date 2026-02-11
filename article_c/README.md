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
- **Assouplissement SNIR** : le seuil effectif est **clampé** entre une borne basse et une borne haute (par défaut **3–6 dB**). Pour un réglage “doux” recommandé, utilisez par exemple `--snir-threshold-db 4.0` et ajustez les bornes si besoin via `--snir-threshold-min-db` / `--snir-threshold-max-db`.

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

### Pipeline de comparaison (fig. 4/5/7/8 + SNIR + DER par cluster)

Pour centraliser les sorties des scripts de comparaison (figures 4/5/7/8,
comparaison SNIR et DER par cluster), utilisez :

```powershell
python -m article_c.all_plot_compare --output-dir article_c/plots/output/compare_all
```

Exporter également des **CSV structurés** (points des figures 4/5/7/8) :

```powershell
python -m article_c.all_plot_compare --export-csv --output-dir article_c/plots/output/compare_all
```

Les CSV sont écrits dans `article_c/plots/output/compare_all/csv` et peuvent être
chargés dans Excel/Power BI pour vérification ou post-traitement.

Exécuter toutes les étapes en sautant l'étape 1 :

```powershell
python -m article_c.run_all --skip-step1
```

Exécuter toutes les étapes en sautant l'étape 2 :

```powershell
python -m article_c.run_all --skip-step2
```

Exécuter toutes les étapes en ajustant collisions/congestion :

```powershell
python -m article_c.run_all --capture-probability 0.28 --congestion-coeff 1.0 --collision-size-factor 1.1
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

### Profil standard adouci (par défaut)

Depuis cette version, l'étape 2 utilise un **profil standard adouci** par défaut
pour réduire les risques de congestion extrême. Les valeurs par défaut suivantes
servent de base lorsque `--safe-profile` n'est pas activé :

- `capture_probability=0.28` (tolérance légèrement plus élevée aux collisions).
- `network_load_min=0.60` et `network_load_max=1.65` (clamp de charge plus modéré).
- `collision_size_min=0.60`, `collision_size_under_max=1.10`,
  `collision_size_over_max=1.90` (facteur de taille des collisions adouci).

Le profil sécurisé reste disponible pour des scénarios plus difficiles ou pour
stabiliser rapidement des runs instables.


### Calibration Step2 (profil standard + safe)

Objectif de calibration : conserver un PDR non nul aux faibles tailles, tout en gardant une décroissance visible quand la densité augmente (jusqu'à ~1280 nœuds).

Ajustements retenus :

- **Profil standard (`DEFAULT_CONFIG.step2`)**
  - `capture_probability=0.28`
  - `network_load_min/max=0.60/1.65`
  - `collision_size_min/under/over=0.60/1.10/1.90`
- **Profil safe (`STEP2_SAFE_CONFIG`)**
  - `capture_probability=0.32`
  - `network_load_min/max=0.65/1.45`
  - `collision_size_min/under/over=0.65/1.05/1.60`
- **Profil super-safe (`STEP2_SUPER_SAFE_CONFIG`)**
  - `capture_probability=0.36`
  - `network_load_min/max=0.75/1.30`
  - `collision_size_min/under/over=0.75/1.00/1.40`

Validation qualitative (exemple) : exécuter une passe rapide sur `80, 160, 320, 640, 960, 1280` et vérifier que le `success_rate` moyen reste **> 0** pour les petites tailles puis diminue globalement vers 1280 nœuds.

Exemple de tendance observée (moyenne agrégée rapide, 6 rounds, seed fixe) :

- `n=80`: ~0.0152
- `n=160`: ~0.0065
- `n=320`: ~0.0022
- `n=640`: ~0.0010
- `n=960`: ~0.0005
- `n=1280`: ~0.0005

Commande type (Windows 11) :

```powershell
python article_c/step2/run_step2.py --network-sizes 80 160 320 640 960 1280 --replications 1 --seeds_base 123 --allow-low-success-rate --workers 1
```

### Mode sécurisé (--safe-profile)

Le flag `--safe-profile` active un **preset modéré** pour l'étape 2, pensé pour
stabiliser la charge, les collisions et le plancher de récompense. Il applique
automatiquement des valeurs plus douces :

- **Charge** (clamp du facteur de charge réseau) : `network_load_min=0.65` et `network_load_max=1.45`.
- **Collisions** (bornes du facteur de taille) : `collision_size_min=0.65`,
  `collision_size_under_max=1.05`, `collision_size_over_max=1.60`.
- **Reward floor** : `reward_floor=0.05` (plancher appliqué dès que `success_rate > 0`).

Exemples :

```powershell
python article_c/step2/run_step2.py --safe-profile --network-sizes 50 100 150 --replications 5 --seeds_base 1000
python -m article_c.run_all --safe-profile
```

### Auto-safe-profile

L'option `--auto-safe-profile` déclenche un basculement automatique vers
`STEP2_SAFE_CONFIG` si la première taille simulée passe sous le seuil de succès
(`success_rate` moyen < 0.2). En mode multi-process, l'exécution devient
séquentielle pour pouvoir détecter le premier échec.

### Réduire la verbosité des alertes (étape 2)

L'alerte "reward uniforme" peut être émise fréquemment selon les scénarios. Utilisez
`--reward-alert-level` pour la basculer en `INFO` et réduire la verbosité.

```powershell
python article_c/step2/run_step2.py --network-sizes 50 100 150 --replications 5 --seeds_base 1000 --reward-alert-level INFO
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

- `--capture-probability` : probabilité qu'une collision laisse un émetteur survivre. Valeur conseillée **0.22–0.38** (défaut 0.28).
- `--congestion-coeff` : coefficient multiplicatif appliqué à la probabilité de congestion. Valeur conseillée **0.8–1.2** (défaut 1.0).
- `--congestion-coeff-base` : coefficient de base de la probabilité de congestion. Valeur conseillée **0.25–0.40** (défaut 0.28).
- `--congestion-coeff-growth` : vitesse de croissance avec la surcharge. Valeur conseillée **0.25–0.50** (défaut 0.30).
- `--congestion-coeff-max` : plafond de la probabilité de congestion. Valeur conseillée **0.25–0.40** (défaut 0.30).
- `--network-load-min` / `--network-load-max` : bornes du facteur de charge réseau (clamp).
- `--collision-size-min` / `--collision-size-under-max` / `--collision-size-over-max` : bornes du facteur de taille des collisions.
- `--collision-size-factor` : facteur de taille appliqué aux collisions (si non défini, calcul automatique). Valeur conseillée **0.8–1.6** selon la densité.

Ces options sont disponibles via `article_c/step2/run_step2.py` et `article_c/run_all.py`.

Exemple CLI avec ajustement des coefficients :

```powershell
python article_c/step2/run_step2.py --network-sizes 50 100 150 --replications 5 --seeds_base 1000 --capture-probability 0.28 --congestion-coeff 1.0 --congestion-coeff-base 0.28 --congestion-coeff-growth 0.30 --congestion-coeff-max 0.30 --collision-size-factor 1.1
```

### Causes fréquentes d’un `success_rate` faible et valeurs de départ recommandées

Un `success_rate` bas vient généralement d’un **triptyque** : congestion excessive,
collisions élevées et SNIR trop strict. Avant d’ajuster la récompense, stabilisez
d’abord ces paramètres :

- **Congestion** : trop de charge effective augmente la probabilité d’échec même
  sans collisions directes.
  - Symptôme : chute globale du succès, même à faible densité.
  - Paramètres clés : `--congestion-coeff`, `--congestion-coeff-base`,
    `--congestion-coeff-growth`, `--congestion-coeff-max`,
    `--network-load-min/--network-load-max`.
- **Collisions** : la charge radio entraîne des pertes co‑SF, surtout si la
  probabilité de capture est faible.
  - Symptôme : pertes en rafale quand la densité augmente.
  - Paramètres clés : `--capture-probability`, `--collision-size-factor`,
    `--collision-size-min/--collision-size-under-max/--collision-size-over-max`.
- **SNIR** : un seuil trop exigeant peut invalider des réceptions pourtant
  “proches” de la sensibilité.
  - Symptôme : `success_rate` bas même en mode peu chargé.
  - Paramètres clés : `--snir-threshold-db`,
    `--snir-threshold-min-db/--snir-threshold-max-db`.

**Valeurs de départ recommandées (profil standard adouci)**

- **Congestion** : `--congestion-coeff 1.0`, `--congestion-coeff-base 0.28`,
  `--congestion-coeff-growth 0.30`, `--congestion-coeff-max 0.30`,
  `--network-load-min 0.60`, `--network-load-max 1.65`.  
  *Justification* : garde une croissance modérée de la congestion sans écraser
  les cas moyens.
- **Collisions** : `--capture-probability 0.28`,
  `--collision-size-factor 1.1`,
  `--collision-size-min 0.60`,
  `--collision-size-under-max 1.10`,
  `--collision-size-over-max 1.90`.  
  *Justification* : tolérance réaliste aux collisions et montée progressive
  avec la densité.
- **SNIR** : `--snir-threshold-db 4.0` avec clamp
  `--snir-threshold-min-db 3.0` / `--snir-threshold-max-db 6.0`.  
  *Justification* : seuil “doux” évitant des refus excessifs tout en respectant
  l’impact des interférences.

Si le `success_rate` reste trop bas, appliquez `--safe-profile` pour stabiliser
rapidement les runs, puis augmentez progressivement `--capture-probability` ou
relâchez le clamp de charge (`--network-load-max`) par petites touches.

### Plancher de récompense en absence de succès (étape 2)

Quand les conditions sont extrêmes (ex. congestion forte), `success_rate` peut tomber à **0** et produire des rewards uniformes. L'option `--floor-on-zero-success` (config Step2 `floor_on_zero_success`) force l'application du plancher d'exploration (`reward_floor` effectif) **avant** le clip lorsque `success_rate == 0`, afin de préserver un signal d'exploration.

Exemple CLI :

```powershell
python article_c/step2/run_step2.py --network-sizes 50 100 150 --replications 5 --seeds_base 1000 --floor-on-zero-success
```

Générer toutes les figures :

```powershell
python -m article_c.make_all_plots
```

Contrôler les formats d'export (PNG/EPS, PDF optionnel) :

```powershell
python -m article_c.make_all_plots --formats png,eps
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

## Reproduction article QoS / Comparaison SNIR

Cette section documente les scripts dédiés à la reproduction des figures QoS et
à la comparaison SNIR. Ils reposent **exclusivement** sur des CSV agrégés en
format **flat** dans `article_c/step*/results/`.

### Entrées attendues (CSV agrégés flat)

- `article_c/step1/results/aggregated_results.csv`
- `article_c/step2/results/aggregated_results.csv`
- (optionnel) `article_c/common/data/author_curves.csv` pour les courbes auteurs QoS.
- (optionnel) `article_c/common/data/author_curves_snir.csv` pour la comparaison SNIR.

### Sorties générées (PNG/EPS, PDF optionnel)

Les scripts produisent des fichiers dans les répertoires ci‑dessous, avec les
extensions demandées (par défaut PNG/EPS). Pour inclure le PDF, ajouter
`--formats png,eps,pdf`.

- `article_c/plots/output/` :
  - `fig4_der_by_cluster.*`
  - `fig5_der_by_load.*`
  - `fig7_traffic_sacrifice.*`
  - `fig8_throughput_clusters.*`
- `article_c/plots/output/compare_with_snir/` :
  - `compare_pdr_snir.*`
  - `compare_der_snir.*`
  - `compare_throughput_snir.*`
- `article_c/plots/output/` :
  - `plot_cluster_der.*`

### Commandes Windows 11

> Toutes les commandes ci‑dessous utilisent `python -m` pour garantir la
> résolution correcte des imports sous Windows 11.

Reproduire les figures QoS (figures 4/5/7/8) :

```powershell
python -m article_c.reproduce_author_results --formats png,eps
```

Comparer SNIR ON/OFF (PDR/DER/Throughput) :

```powershell
python -m article_c.compare_with_snir --formats png,eps
```

Tracer le DER par cluster :

```powershell
python -m article_c.plot_cluster_der --formats png,eps
```

### Style IEEE et option `--formats`

- **Taille/Légende** : les scripts appliquent les recommandations IEEE
  (dimensions et légende en haut) via les helpers de style.
- **Export PDF** : pour inclure le PDF en plus du PNG/EPS, ajouter
  `--formats png,eps,pdf`.
- **Formats multiples** : `--formats` accepte une liste séparée par des virgules
  (ex. `png,eps,pdf`).

### Script d'orchestration (optionnel)

Si un script d'orchestration est ajouté (ex. `all_plot_compare.py`), documentez
ici :

- **Commande Windows 11** (ex. `python -m article_c.all_plot_compare --formats png,eps`).
- **Entrées attendues** (CSV agrégés flat dans `article_c/step*/results/`).
- **Sorties** (liste des fichiers générés et répertoires de sortie).

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
- **CLI** : ajouter `pdf` à `--formats` si nécessaire (ex. `png,eps,pdf`).

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
