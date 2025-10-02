# Scénarios Article B (B1–B10)

Ce guide décrit l’intégralité des dix scénarios utilisés pour l’article B de la campagne MNE3SD. Chaque section récapitule les paramètres appliqués, la commande CLI de référence, le fichier CSV produit et le module de tracé associé. Toutes les commandes sont à exécuter depuis la racine du dépôt.

## Vue d’ensemble

| ID | Script de scénario | Paramètres clés | CSV attendu | Module de tracé |
|----|-------------------|-----------------|-------------|-----------------|
| B1 | `run_mobility_range_sweep` | RandomWaypoint, portée 5 km, 100 nœuds, 50 paquets, 5 réplicats | `results/mne3sd/article_b/B1_B4_range_5km.csv` | `plot_mobility_range_metrics` |
| B2 | `run_mobility_range_sweep` | RandomWaypoint, portée 10 km, 100 nœuds, 50 paquets, 5 réplicats | `results/mne3sd/article_b/B2_B5_range_10km.csv` | `plot_mobility_range_metrics` |
| B3 | `run_mobility_range_sweep` | RandomWaypoint, portée 15 km, 100 nœuds, 50 paquets, 5 réplicats | `results/mne3sd/article_b/B3_B6_range_15km.csv` | `plot_mobility_range_metrics` |
| B4 | `run_mobility_range_sweep` | Smooth, portée 5 km, 100 nœuds, 50 paquets, 5 réplicats | `results/mne3sd/article_b/B1_B4_range_5km.csv` | `plot_mobility_range_metrics` |
| B5 | `run_mobility_range_sweep` | Smooth, portée 10 km, 100 nœuds, 50 paquets, 5 réplicats | `results/mne3sd/article_b/B2_B5_range_10km.csv` | `plot_mobility_range_metrics` |
| B6 | `run_mobility_range_sweep` | Smooth, portée 15 km, 100 nœuds, 50 paquets, 5 réplicats | `results/mne3sd/article_b/B3_B6_range_15km.csv` | `plot_mobility_range_metrics` |
| B7 | `run_mobility_speed_sweep` | RandomWaypoint, profil « pedestrian » (0,5–1,5 m/s), portée 10 km | `results/mne3sd/article_b/B7_B8_speed_pedestrian.csv` | `plot_mobility_speed_metrics` |
| B8 | `run_mobility_speed_sweep` | Smooth, profil « pedestrian » (0,5–1,5 m/s), portée 10 km | `results/mne3sd/article_b/B7_B8_speed_pedestrian.csv` | `plot_mobility_speed_metrics` |
| B9 | `run_mobility_gateway_sweep` | RandomWaypoint, 1/2/4 passerelles, 100 nœuds | `results/mne3sd/article_b/B9_B10_gateway.csv` | `plot_mobility_gateway_metrics` |
| B10 | `run_mobility_gateway_sweep` | Smooth, 1/2/4 passerelles, 100 nœuds | `results/mne3sd/article_b/B9_B10_gateway.csv` | `plot_mobility_gateway_metrics` |

> **Remarque :** les paires (B1,B4), (B2,B5), (B3,B6) et (B7,B8) partagent la même exécution. Le fichier CSV contient alors deux lignes agrégées (`replicate=aggregate`) distinguées par la colonne `model`.

## Détails par scénario

### B1 – RandomWaypoint, portée 5 km
- **Modèle de mobilité :** RandomWaypoint parmi les deux modèles évalués par le script.【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L320-L344】  
- **Topologie radio :** une passerelle, zone carrée de 10 km × 10 km dérivée de la portée (aire `range_km × 2000`).【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L333-L347】  
- **Charge :** 100 nœuds, 50 paquets chacun, réplicats Monte Carlo : 5 (valeurs par défaut).【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L239-L351】  
- **Intervalle moyen entre paquets :** 300 s.【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L254-L257】

```bash
python -m scripts.mne3sd.article_b.scenarios.run_mobility_range_sweep \
  --range-km 5 \
  --nodes 100 --packets 50 --replicates 5 --seed 1 \
  --results results/mne3sd/article_b/B1_B4_range_5km.csv
```

- **Extraction des métriques :** conservez la ligne agrégée (`replicate=aggregate`) dont `model=random_waypoint` pour constituer le jeu B1.  
- **Tracé associé :**

```bash
python -m scripts.mne3sd.article_b.plots.plot_mobility_range_metrics \
  --results results/mne3sd/article_b/B1_B4_range_5km.csv
```

### B2 – RandomWaypoint, portée 10 km
- Même configuration que B1, portée réglée sur 10 km (aire 20 km × 20 km).【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L333-L347】

```bash
python -m scripts.mne3sd.article_b.scenarios.run_mobility_range_sweep \
  --range-km 10 \
  --nodes 100 --packets 50 --replicates 5 --seed 1 \
  --results results/mne3sd/article_b/B2_B5_range_10km.csv
```

- Extraire la ligne agrégée avec `model=random_waypoint`.  
- Utiliser le même module `plot_mobility_range_metrics` sur ce fichier.

### B3 – RandomWaypoint, portée 15 km
- Identique à B1 avec `--range-km 15` (aire 30 km × 30 km, borne maximale permise par le script).【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L333-L347】【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L54-L63】

```bash
python -m scripts.mne3sd.article_b.scenarios.run_mobility_range_sweep \
  --range-km 15 \
  --nodes 100 --packets 50 --replicates 5 --seed 1 \
  --results results/mne3sd/article_b/B3_B6_range_15km.csv
```

- Extraire `model=random_waypoint`, `replicate=aggregate`.  
- Tracé : `plot_mobility_range_metrics`.

### B4 – Smooth, portée 5 km
- **Modèle de mobilité :** SmoothMobility (deuxième modèle évalué).【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L320-L344】  
- Partage la même exécution que B1, conservez la ligne agrégée `model=smooth` dans `B1_B4_range_5km.csv`.  
- Tracé : `plot_mobility_range_metrics`.

### B5 – Smooth, portée 10 km
- Même exécution que B2. Conserver la ligne agrégée `model=smooth` dans `B2_B5_range_10km.csv`.  
- Tracé : `plot_mobility_range_metrics`.

### B6 – Smooth, portée 15 km
- Même exécution que B3. Conserver la ligne agrégée `model=smooth` dans `B3_B6_range_15km.csv`.  
- Tracé : `plot_mobility_range_metrics`.

### B7 – RandomWaypoint, profil « pedestrian »
- **Profil de vitesse :** 0,5–1,5 m/s (profil `pedestrian` par défaut).【F:scripts/mne3sd/article_b/scenarios/run_mobility_speed_sweep.py†L55-L156】  
- **Portée :** 10 km (aire 20 km × 20 km).【F:scripts/mne3sd/article_b/scenarios/run_mobility_speed_sweep.py†L327-L334】  
- **Charge :** 100 nœuds, 50 paquets, 5 réplicats.【F:scripts/mne3sd/article_b/scenarios/run_mobility_speed_sweep.py†L251-L346】

```bash
python -m scripts.mne3sd.article_b.scenarios.run_mobility_speed_sweep \
  --speed-profiles "pedestrian: (0.5, 1.5)" \
  --range-km 10 \
  --nodes 100 --packets 50 --replicates 5 --seed 1 \
  --results results/mne3sd/article_b/B7_B8_speed_pedestrian.csv
```

- Conserver la ligne agrégée `model=random_waypoint`, `speed_profile=pedestrian`.  
- Tracé :

```bash
python -m scripts.mne3sd.article_b.plots.plot_mobility_speed_metrics \
  --results results/mne3sd/article_b/B7_B8_speed_pedestrian.csv
```

### B8 – Smooth, profil « pedestrian »
- Extraire du fichier précédent la ligne agrégée `model=smooth`.  
- Tracé : `plot_mobility_speed_metrics`.

### B9 – RandomWaypoint, 1/2/4 passerelles
- **Passerelles explorées :** 1, 2 et 4 (valeurs par défaut du script).【F:scripts/mne3sd/article_b/scenarios/run_mobility_gateway_sweep.py†L60-L151】  
- **Canaux LoRaWAN :** 868,1/868,3/868,5 MHz (plan par défaut).【F:scripts/mne3sd/article_b/scenarios/run_mobility_gateway_sweep.py†L60-L184】  
- **Portée :** 10 km (aire 20 km × 20 km).【F:scripts/mne3sd/article_b/scenarios/run_mobility_gateway_sweep.py†L419-L439】  
- **Charge :** 100 nœuds, 50 paquets, 5 réplicats.【F:scripts/mne3sd/article_b/scenarios/run_mobility_gateway_sweep.py†L353-L474】

```bash
python -m scripts.mne3sd.article_b.scenarios.run_mobility_gateway_sweep \
  --gateways-list 1,2,4 \
  --range-km 10 \
  --nodes 100 --packets 50 --replicates 5 --seed 1 \
  --results results/mne3sd/article_b/B9_B10_gateway.csv
```

- Conserver les lignes agrégées `model=random_waypoint` pour chacune des valeurs de `gateways`.  
- Tracé :

```bash
python -m scripts.mne3sd.article_b.plots.plot_mobility_gateway_metrics \
  --results results/mne3sd/article_b/B9_B10_gateway.csv
```

### B10 – Smooth, 1/2/4 passerelles
- Même exécution que B9. Conserver les lignes agrégées `model=smooth`.  
- Tracé : `plot_mobility_gateway_metrics`.

## Profils d’exécution (`--profile`)

Chaque script accepte l’argument commun `--profile` (ou la variable d’environnement `MNE3SD_PROFILE`). Utilisez :

- `full` (par défaut) pour reproduire exactement les paramètres ci-dessus.  
- `fast` pour limiter la portée à 5–10 km, 80 nœuds, 25 paquets et 3 réplicats, idéal pour des itérations rapides sous Windows 11.【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L13-L63】【F:scripts/mne3sd/article_b/scenarios/run_mobility_speed_sweep.py†L14-L69】【F:scripts/mne3sd/article_b/scenarios/run_mobility_gateway_sweep.py†L14-L77】【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L281-L318】【F:scripts/mne3sd/article_b/scenarios/run_mobility_speed_sweep.py†L293-L347】【F:scripts/mne3sd/article_b/scenarios/run_mobility_gateway_sweep.py†L395-L440】  
- `ci` pour restreindre l’exécution aux paramètres minimum (40 nœuds, 10 paquets, un seul réplicat, portée de 5 km) et accélérer les vérifications automatisées.【F:scripts/mne3sd/article_b/scenarios/run_mobility_range_sweep.py†L54-L313】【F:scripts/mne3sd/article_b/scenarios/run_mobility_speed_sweep.py†L55-L343】【F:scripts/mne3sd/article_b/scenarios/run_mobility_gateway_sweep.py†L60-L435】

Ajoutez simplement `--profile fast` ou `--profile ci` aux commandes précédentes. Les scripts recadrent automatiquement les paramètres excédentaires pour respecter le profil demandé.

## Astuces pratiques (Windows 11)

### Copier ou renommer les CSV

PowerShell permet de dupliquer rapidement les jeux de résultats :

```powershell
Copy-Item results\mne3sd\article_b\B1_B4_range_5km.csv `
  results\mne3sd\article_b\B1_random_waypoint_5km.csv
```

Pour renommer en place :

```powershell
Rename-Item results\mne3sd\article_b\B1_B4_range_5km.csv `
  B1_B4_range_5km_backup.csv
```

### Fusionner plusieurs résultats pour superposer les portées

Utilisez PowerShell pour agréger les lignes `replicate=aggregate` issues de plusieurs scénarios (pratique pour juxtaposer les portées 5/10/15 km dans un même graphique) :

```powershell
$files = Get-ChildItem results\mne3sd\article_b\B*_range_*.csv
$rows = foreach ($file in $files) {
  Import-Csv $file | Where-Object { $_.replicate -eq 'aggregate' }
}
$rows | Export-Csv results\mne3sd\article_b\mobility_range_overlay.csv -NoTypeInformation
```

Le fichier `mobility_range_overlay.csv` peut ensuite être utilisé comme entrée unique du module `plot_mobility_range_metrics` pour tracer les portées côte à côte.

### Fusionner des CSV hétérogènes avec Python

Lorsque vous mélangez des profils de vitesse, un mini-script Python exécutable sous Windows 11 permet de concaténer les agrégats tout en sélectionnant les colonnes utiles :

```powershell
python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path('results/mne3sd/article_b')
inputs = [root / 'B7_B8_speed_pedestrian.csv']
df = pd.concat(
    pd.read_csv(path) for path in inputs
)
subset = df[df['replicate'] == 'aggregate'][
    ['model', 'speed_profile', 'pdr_mean', 'avg_delay_s_mean']
]
subset.to_csv(root / 'mobility_speed_overlay.csv', index=False)
PY
```

### Superposer les portées dans une figure existante

Après avoir généré `mobility_range_overlay.csv`, relancez le module de tracé en lui fournissant plusieurs fichiers :

```powershell
python -m scripts.mne3sd.article_b.plots.plot_mobility_range_metrics \
  --results results/mne3sd/article_b/mobility_range_overlay.csv
```

Le graphique `figures/mne3sd/article_b/mobility_range/pdr_vs_range/pdr_vs_communication_range.*` présentera alors les courbes 5/10/15 km sur un seul jeu.【F:scripts/mne3sd/article_b/plots/plot_mobility_range_metrics.py†L22-L175】

Mettez régulièrement à jour ce README lors de l’ajout de nouveaux scénarios ou figures afin de conserver une traçabilité parfaite des expériences Article B.
