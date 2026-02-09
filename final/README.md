# Dossier `final/`

Ce dossier regroupe un flux de travail **reproductible** pour générer des scénarios, stocker les CSV et centraliser les figures produites par LoRaFlexSim.

## Pré requis

- **Python 3.10+** (recommandé : environnement virtuel).
- **Dépendances LoRaFlexSim** installées depuis la racine du dépôt :

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Windows 11 (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

> Si PowerShell bloque l’activation, utilisez : `powershell -ExecutionPolicy Bypass -File .\.venv\Scripts\Activate.ps1`.

## Exécuter une simulation en CLI

Les commandes ci‑dessous écrivent les CSV dans `final/data/`.

### Linux / macOS

```bash
python -m loraflexsim.run --nodes 30 --gateways 1 --mode random --interval 10 --steps 100 --output final/data/simulation.csv
```

### Windows 11 (PowerShell)

```powershell
python -m loraflexsim.run --nodes 30 --gateways 1 --mode random --interval 10 --steps 100 --output final/data/simulation.csv
```

## Tracer une figure à partir des CSV

L’exemple suivant lit un ou plusieurs CSV et génère une figure de PDR moyenne dans `final/figures/`.

### Linux / macOS

```bash
python examples/analyse_resultats.py final/data/simulation.csv --output-dir final/figures --basename pdr_by_nodes
```

### Windows 11 (PowerShell)

```powershell
python examples/analyse_resultats.py final/data/simulation.csv --output-dir final/figures --basename pdr_by_nodes
```

## Format des CSV générés

Les fichiers produits par `--output` contiennent l’en‑tête suivant :

```
nodes,gateways,channels,mode,interval,steps,delivered,collisions,PDR(%),energy,avg_delay,throughput_bps
```

**Emplacement des sorties**

- CSV de simulation : `final/data/`
- Figures : `final/figures/`
- Scénarios personnalisés (fichiers d’entrée, INI, etc.) : `final/scenarios/`
- Graphiques complémentaires (plots intermédiaires) : `final/plots/`

## Ajuster les paramètres clés

- **Période d’émission** : ajustez `--interval` (en secondes). Exemple : `--interval 60`.
- **Rayon / taille de zone** : pour des scénarios plus larges, privilégiez les presets longue portée (`--long-range-demo`) ou l’auto‑calibrage (`--long-range-auto <surface_km2> [distance_km]`). Pour un contrôle fin de la zone (mètres), créez un script Python qui instancie `Simulator(area_size=...)` et placez‑le dans `final/scenarios/`.
- **Taille de paquet** : la CLI `loraflexsim.run` utilise la valeur par défaut, mais vous pouvez la surcharger en Python via `Simulator(payload_size_bytes=...)` (script à déposer dans `final/scenarios/`).

### Exemple de script minimal (à placer dans `final/scenarios/`)

```python
from loraflexsim.launcher.simulator import Simulator

sim = Simulator(
    nodes=30,
    gateways=1,
    area_size=2000.0,          # zone carrée de 2 km
    payload_size_bytes=40,     # taille de paquet
    interval=60.0,             # période d’émission
    steps=600,
)
metrics = sim.run()
print(metrics)
```

Vous pouvez ensuite rediriger les métriques vers un CSV en vous inspirant de `loraflexsim.run` ou en réutilisant les utilitaires existants dans `scripts/`.
