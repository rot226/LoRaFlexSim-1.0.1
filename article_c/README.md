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
python article_c/run_all.py
```

Exécuter uniquement l'étape 1 :

```powershell
python article_c/step1/run_step1.py --densities 0.1,0.5,1.0 --replications 5 --seeds_base 1000 --snir_modes snir_on,snir_off
```

Exécuter uniquement l'étape 2 :

```powershell
python article_c/step2/run_step2.py --densities 0.1,0.5,1.0 --replications 5 --seed 1000
```

Générer toutes les figures :

```powershell
python article_c/make_all_plots.py
```

## Seeds et réplications

- **Seeds** : l'étape 1 utilise `--seeds_base` et incrémente le seed à chaque exécution; l'étape 2 utilise `--seed` pour initialiser un seed de base déterministe.
- **Réplications** : `--replications` définit le nombre de répétitions par configuration (densité/algorithme/mode SNIR).

Les résultats sont écrits dans `article_c/step*/results/` et les figures dans `article_c/step*/plots/output/`.

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
- `plot_S7_cluster_outage_vs_density.py` (nouvelle)

### Étape 2

- `plot_RL1.py`
- `plot_RL2.py`
- `plot_RL3.py`
- `plot_RL4.py`
- `plot_RL5.py`
- `plot_RL6_cluster_outage_vs_density.py` (nouvelle)
- `plot_RL7_reward_vs_density.py` (nouvelle)
