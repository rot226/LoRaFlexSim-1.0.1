# Scénarios d'utilisation

Ces scripts illustrent différentes utilisations de LoRaFlexSim et servent de point de départ pour l'exploration du simulateur.

Ce document décrit les scripts fournis pour lancer et visualiser des simulations
LoRa. Chaque utilitaire indique ses paramètres, leurs valeurs par défaut et la
sortie attendue.

## Scripts de simulation (`run_*`)

### `run_mobility_multichannel.py`
- **Description** : exécute plusieurs scénarios combinant mobilité et mono/tri‑canal.
- **Paramètres principaux**
  - `--nodes` (50) : nombre de nœuds simulés.
  - `--packets` (100) : paquets à émettre par nœud.
  - `--seed` (1) : graine aléatoire.
  - `--adr-node`, `--adr-server` : active l'ADR sur les nœuds/serveur.
  - `--area-size` (1000.0) : côté du carré de simulation (m).
  - `--interval` (60.0) : intervalle moyen d'émission (s).
  - `--replicates` (1) : répétitions du scénario.
- **Sortie** : `results/mobility_multichannel.csv` contenant moyenne et écart‑type de
  la PDR, du taux de collision, du délai moyen et de l'énergie par nœud.

### `run_mobility_latency_energy.py`
- **Description** : mêmes scénarios que ci‑dessus en enregistrant PDR, délai,
  énergie et collisions.
- **Paramètres** : identiques à `run_mobility_multichannel.py`.
- **Sortie** : `results/mobility_latency_energy.csv` avec moyenne et écart‑type des
  quatre métriques.

### `run_battery_tracking.py`
- **Description** : suit l'énergie restante de chaque nœud après chaque événement.
- **Paramètres**
  - `--nodes` (5) : nombre de nœuds.
  - `--packets` (3) : paquets par nœud.
  - `--seed` (1) : graine aléatoire.
- **Sortie** : `results/battery_tracking.csv` avec `time`, `node_id`, `energy_j` et
  `capacity_j`.

## Scripts de visualisation (`plot_*`)

### `plot_mobility_multichannel.py`
- **Paramètres** : chemin du CSV agrégé, `--output-dir` ("figures") et options
  `--allowed N,C` pour limiter les couples nœuds/canaux, `--scenarios nom` pour
  filtrer les scénarios.
- **Sortie** : graphiques PNG de PDR, taux de collision, délai moyen et énergie
  moyenne par nœud.

### `plot_mobility_latency_energy.py`
- **Paramètres** : CSV d'entrée et `--output-dir` ("figures").
- **Sortie** : fichiers SVG pour la PDR, le délai moyen, l'énergie moyenne par
  nœud et le taux de collision.

### `plot_battery_tracking.py`
- **Paramètres** : aucun ; lit `results/battery_tracking.csv`.
- **Sortie** : `figures/battery_tracking.png` montrant l'énergie résiduelle.

### `plot_node_positions.py`
- **Paramètres** : `--num-nodes` (100), `--area-size` (1000.0), `--seed` (42),
  `--output` ("figures/node_positions.png").
- **Sortie** : carte de dispersion des positions initiales.

## Scénarios prédéfinis

Les scripts `run_mobility_multichannel.py` et
`run_mobility_latency_energy.py` exécutent huit scénarios prédéfinis :

| Scénario | N | C | Mobilité | Vitesse (m/s) |
|---------|---|---|----------|---------------|
| `n50_c1_static` | 50 | 1 | Non | 0 |
| `n50_c1_mobile` | 50 | 1 | Oui | 5 |
| `n50_c3_mobile` | 50 | 3 | Oui | 5 |
| `n50_c6_static` | 50 | 6 | Non | 0 |
| `n200_c1_static` | 200 | 1 | Non | 0 |
| `n200_c1_mobile` | 200 | 1 | Oui | 5 |
| `n200_c3_mobile` | 200 | 3 | Oui | 5 |
| `n200_c6_static` | 200 | 6 | Non | 0 |

## Paramètres essentiels pour reproduire FLoRa

Pour obtenir des résultats alignés sur les scénarios publiés par FLoRa, appliquez
les réglages suivants dans vos scripts (ainsi que ceux détaillés dans la section
« Reproduire FLoRa » du README) :

- `flora_mode=True` configure automatiquement les seuils de détection, le
  modèle radio `omnet_full` et les temporisations historiques FLoRa.【F:loraflexsim/launcher/simulator.py†L354-L457】
- `environment="flora"` (ou `"flora_hata"`, `"flora_oulu"`) et
  `flora_loss_model` sélectionnent les presets longue portée repris du projet
  d’origine pour les canaux supplémentaires gérés par `MultiChannel`.【F:loraflexsim/launcher/channel.py†L68-L114】
- Les démos longue portée (`run.py --long-range-demo flora_hata`, etc.)
  exploitent ces presets et sont vérifiées par des tests dédiés (>5 km).【F:tests/test_long_range_presets.py†L1-L55】

Ces paramètres peuvent être combinés avec les scénarios fournis plus haut pour
recréer les figures du dépôt FLoRa ou exécuter la matrice de validation.

## Exemple complet

```bash
python scripts/run_mobility_multichannel.py --nodes 200 --interval 1 --replicates 5
python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv
python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv --allowed 50,1 200,3
python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv --scenarios n50_c1_static n50_c1_mobile
```

Le premier script génère `results/mobility_multichannel.csv`, puis le second
crée des graphiques dans le dossier `figures`.

## Activer EXPLoRa-AT

Pour répartir équitablement le temps d'antenne entre les spreading factors,
le simulateur peut être configuré avec l'algorithme EXPLoRa‑AT :

```python
from loraflexsim.launcher import Simulator, explora_at

sim = Simulator(nodes=100, packets=50)
explora_at.apply(sim)  # active l'ADR EXPLoRa‑AT
sim.run()
```

Après quelques uplinks, le serveur trie les nœuds par RSSI et leur assigne
un SF de sorte que `N_sf * Channel.airtime(sf)` soit identique pour tous les
groupes. La puissance d'émission est ensuite augmentée ou réduite par pas
de 3 dB pour maintenir une marge SNR positive.

## Interprétation des métriques

- **PDR** (« Packet Delivery Ratio ») : pourcentage de paquets reçus. 100 %
  est le plafond théorique ; une valeur basse indique une forte perte.
- **Énergie par nœud** : joules consommés par un nœud. Le plafond est la
  capacité de la batterie ; une valeur élevée signifie une durée de vie plus
  courte.
- **Délai moyen** : temps moyen entre l'émission et la réception. Des délais
  trop importants peuvent violer les exigences applicatives.
- **Taux de collision** : pourcentage de transmissions perdues pour collision
  (plafond à 100 %). Un taux élevé signale une saturation du canal.

## Reproduction des figures

Cette section récapitule les utilitaires nécessaires pour générer les figures
fournies et les tests de mobilité associés.

### Prérequis

- Python ≥ 3.10
- Bibliothèques : `pandas` et `matplotlib`

### Scripts automatisés

#### `generate_all_figures.py`
- **Arguments** : `--config`, `--nodes`, `--packets`, `--seed`, `--area-size`.
- **Fichiers produits** : agrégats CSV dans `results/` et figures dans
  `figures/` (PDR, délai, énergie, collisions et suivi batterie).
- **Prérequis** : accès aux scripts décrits ci‑dessous.

### Exécution manuelle

1. `run_mobility_multichannel.py` : génère
   `results/mobility_multichannel.csv`.
2. `plot_mobility_multichannel.py <csv> [--allowed N,C ...]` : crée des PNG
   (PDR, collisions, délai moyen, énergie par nœud).
3. `run_mobility_latency_energy.py` : produit
   `results/mobility_latency_energy.csv`.
4. `plot_mobility_latency_energy.py <csv>` : génère des SVG (PDR, délai,
   énergie, collisions).
5. `run_battery_tracking.py` : enregistre `results/battery_tracking.csv`.
6. `plot_battery_tracking.py` : sauvegarde `figures/battery_tracking.png`.
7. `plot_node_positions.py` : exporte la carte des nœuds
    (`figures/node_positions.png`).

### Abréviations des figures

Les légendes des graphiques utilisent les abréviations suivantes :

- `N` : nombre de nœuds.
- `C` : nombre de canaux.
- `speed` : vitesse des nœuds en m/s.

### Génération des tests de mobilité

Les tests créent les CSV utilisés par les scripts de tracé :

```bash
pytest tests/test_mobility_latency.py
pytest tests/test_mobility_energy_per_packet.py
```

Les fichiers `results/mobility_latency.csv` et `results/mobility_energy.csv`
peuvent ensuite être tracés via les utilitaires `plot_*` correspondants.

