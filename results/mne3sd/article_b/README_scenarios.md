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

## Graphiques Article B

Les trois modules de tracé créent automatiquement des paires PNG/EPS suivant la nomenclature
`figures/mne3sd/<article>/<scénario>/<métrique>/<nom>.(png|eps)`.【F:scripts/mne3sd/common.py†L109-L154】

### `plot_mobility_range_metrics`

- **PDR agrégé vs portée :** `figures/mne3sd/article_b/mobility_range/pdr_vs_range/pdr_vs_communication_range.*`. Les points
  mettant en évidence une PDR inférieure à un seuil optionnel peuvent être activés via
  `--highlight-threshold 90`.【F:scripts/mne3sd/article_b/plots/plot_mobility_range_metrics.py†L40-L111】【F:scripts/mne3sd/article_b/plots/plot_mobility_range_metrics.py†L120-L146】
- **Délai moyen vs portée :** `figures/mne3sd/article_b/mobility_range/average_delay_vs_range/average_delay_vs_communication_range.*`,
  généré automatiquement lors du même appel.【F:scripts/mne3sd/article_b/plots/plot_mobility_range_metrics.py†L148-L182】
- **Style :** applique le preset IEEE (`apply_ieee_style`). Vous pouvez fournir un fichier `.mplstyle` personnalisé avec
  `--style chemin/vers/style.mplstyle` pour ajuster les polices ou la palette.【F:scripts/mne3sd/article_b/plots/plot_mobility_range_metrics.py†L16-L37】【F:scripts/mne3sd/article_b/plots/plot_mobility_range_metrics.py†L187-L201】

### `plot_mobility_speed_metrics`

Le script synthétise toutes les métriques agrégées issues de B7/B8 et de futurs scénarios de balayage des vitesses :

- **Barres groupées PDR :** `figures/mne3sd/article_b/mobility_speed/pdr_by_speed_profile/pdr_by_speed_profile.*` (pourcentages,
  arrondis selon les données).【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L27-L161】【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L205-L244】
- **Barres groupées délai moyen :** `figures/mne3sd/article_b/mobility_speed/average_delay_by_speed_profile/average_delay_by_speed_profile.*`.【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L205-L244】【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L260-L279】
- **Jitter (si présent dans le CSV) :** `figures/mne3sd/article_b/mobility_speed/latency_jitter_by_speed_profile/latency_jitter_by_speed_profile.*`.【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L281-L353】
- **Énergie moyenne par nœud :** `figures/mne3sd/article_b/mobility_speed/energy_by_speed_profile/energy_by_speed_profile.*`.【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L355-L409】
- **Empilement énergétique :** `figures/mne3sd/article_b/mobility_speed/energy_stack_by_speed_profile/energy_stack_by_speed_profile.*`. Utiles pour comparer visuellement la contribution de chaque modèle.【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L411-L457】
- **Carte thermique PDR (si plusieurs portées) :** `figures/mne3sd/article_b/mobility_speed/pdr_heatmap_speed_profile_range/pdr_heatmap_speed_profile_range.*`. Les annotations indiquent la PDR (%) et utilisent des couleurs adaptées aux faibles valeurs.【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L459-L533】
- **Options :** `--dpi` contrôle la résolution de tous les exports (300 dpi par défaut) et `--style` permet de charger un thème Matplotlib supplémentaire.【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L38-L69】【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L535-L573】

### `plot_mobility_gateway_metrics`

Exploite les agrégats produits pour B9/B10 afin de documenter l’impact du nombre de passerelles :

- **Répartition du trafic par passerelle :** `figures/mne3sd/article_b/mobility_gateway/pdr_distribution_by_gateway/pdr_distribution_by_gateway.*`. Chaque barre empilée indique la part de PDR captée par passerelle (JSON décodé automatiquement depuis `pdr_by_gateway_mean`).【F:scripts/mne3sd/article_b/plots/plot_mobility_gateway_metrics.py†L26-L145】【F:scripts/mne3sd/article_b/plots/plot_mobility_gateway_metrics.py†L155-L203】
- **Délai downlink vs nombre de passerelles :** `figures/mne3sd/article_b/mobility_gateway/downlink_delay_vs_gateways/average_downlink_delay_vs_gateways.*`.【F:scripts/mne3sd/article_b/plots/plot_mobility_gateway_metrics.py†L205-L238】
- **Comparaison des modèles :** `figures/mne3sd/article_b/mobility_gateway/model_comparison/pdr_vs_delay_model_comparison.*` résume PDR (%) et délai downlink avec annotations `N GW`.【F:scripts/mne3sd/article_b/plots/plot_mobility_gateway_metrics.py†L240-L278】
- **Options :** `--style` suit la même logique que les autres scripts et `--show` permet d’afficher les figures au lieu de fermer Matplotlib en mode batch.【F:scripts/mne3sd/article_b/plots/plot_mobility_gateway_metrics.py†L30-L87】【F:scripts/mne3sd/article_b/plots/plot_mobility_gateway_metrics.py†L280-L304】

> Astuce : les trois scripts prennent en charge `--show` pour examiner les figures avant export (pratique sous Windows 11 avec une session interactive), sans modifier les fichiers produits.【F:scripts/mne3sd/article_b/plots/plot_mobility_range_metrics.py†L37-L39】【F:scripts/mne3sd/article_b/plots/plot_mobility_range_metrics.py†L199-L201】【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L65-L69】【F:scripts/mne3sd/article_b/plots/plot_mobility_speed_metrics.py†L571-L573】【F:scripts/mne3sd/article_b/plots/plot_mobility_gateway_metrics.py†L45-L87】【F:scripts/mne3sd/article_b/plots/plot_mobility_gateway_metrics.py†L280-L304】

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
