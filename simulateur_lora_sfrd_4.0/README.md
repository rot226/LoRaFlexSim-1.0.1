# Simulateur de réseau LoRa 4.0

Ce dépôt contient un simulateur LoRa léger entièrement écrit en Python. Le code
source principal se trouve dans le dossier `VERSION_4`. Il intègre une couche
physique simplifiée inspirée d'OMNeT++ pour modéliser le bruit thermique ainsi
que les dérives de fréquence et d'horloge, tout en restant indépendant de la
pile complète OMNeT++.

## Fonctionnalités clés

- Application du duty cycle pour reproduire les contraintes LoRa réelles
- Mobilité optionnelle des nœuds : trajectoires de Bézier, RandomWaypoint avec
  carte de terrain, suivi de traces GPS ou navigation évitant les obstacles
- Prise en charge multi‑canal avec répartition configurable
- Modèles de propagation avancés (pertes, bruit, fading, interférences)
- Gestion des gains d'antenne et des pertes de câbles dans le bilan de liaison
- Modèles de path loss COST231 ou Okumura‑Hata et calculs 3D
- Bandes d'interférence sélectives et bruit dépendant de la météo
- Effet capture et durée minimale de chevauchement pour ignorer certains conflits
- Modules LoRaWAN complets : ADR, classes B et C, plans de canaux régionaux
- Modèle de batterie inspiré de FLoRa pour suivre l'énergie restante

## Démarrage rapide

```bash
# Installation
cd VERSION_4
python3 -m venv env
source env/bin/activate    # Sous Windows : env\Scripts\activate
pip install -r requirements.txt

# Lancement du tableau de bord
panel serve launcher/dashboard.py --show
# Depuis la racine du dépôt
panel serve VERSION_4/launcher/dashboard.py --show
```

L'interface web propose un champ **Graine** afin de reproduire exactement le
placement des nœuds entre deux exécutions. Activez **Positions manuelles** pour
saisir vous‑même les coordonnées des nœuds ou passerelles sous la forme
`node,id=3,x=120,y=40` ou `gw,id=1,x=10,y=80`. Le champ **CSV FLoRa** permet de
charger un export de référence et d'afficher la comparaison en direct.

### Lancer une simulation

```bash
python run.py --nodes 20 --steps 100
```

Ajoutez `--seed <n>` pour conserver le même placement à chaque lancement et
`--runs <n>` pour répéter la simulation plusieurs fois et moyenner les
métriques. Le simulateur peut aussi être exécuté depuis la racine :

```bash
python VERSION_4/run.py --nodes 20 --steps 100
```

Toutes les options sont détaillées dans `VERSION_4/README.md`.

## Utilisation avancée

```bash
# Simulation multi‑canaux avec mobilité
python VERSION_4/run.py --nodes 50 --gateways 2 --channels 3 \
  --mobility --steps 500 --output advanced.csv

# Démonstration LoRaWAN avec downlinks
python VERSION_4/run.py --lorawan-demo --steps 100 --output lorawan.csv
```

### Exemples de classes B et C

```python
from launcher import Simulator

# Nœuds en classe B avec slots périodiques
sim_b = Simulator(num_nodes=10, node_class="B", beacon_interval=128,
                  ping_slot_interval=1.0)
sim_b.run(1000)

# Nœuds en classe C à écoute quasi continue
sim_c = Simulator(num_nodes=5, node_class="C", class_c_rx_interval=0.5)
sim_c.run(500)
```

### Mobilité réaliste

```python
from launcher import Simulator

sim = Simulator(num_nodes=20, num_gateways=3, area_size=2000.0, mobility=True,
                mobility_speed=(1.0, 5.0))
sim.run(1000)
```

### Exemple de scénario FLoRa

```bash
python examples/run_flora_example.py --runs 5 --seed 123
```

Le script affiche le taux de livraison et l'histogramme des SF pour chaque run.
Le paramètre `flora_mode=True` applique automatiquement le seuil officiel de
-110 dBm et une fenêtre d'interférence de 5 s. L'option `--flora-csv <fichier>`
permet de comparer directement avec un export FLoRa. Les utilitaires du dossier
`tools` facilitent la conversion entre CSV et fichiers `.sca`/`.vec` et offrent
désormais `convert_flora_scenario.py` pour transformer un scénario FLoRa `.ini`
en JSON (et inversement).

Un fichier `examples/flora_full.ini` reproduit les positions de référence.
Chargez‑le via `Simulator(config_file="examples/flora_full.ini")`. D'autres
exports (par exemple `flora_full.csv` ou `flora_collisions.csv`) sont disponibles
pour vérification.

Ci‑dessous un extrait de fichier INI utilisable avec le mode FLoRa :

```ini
[General]
network = flora.simulations.LoRaNetworkTest
**.maxTransmissionDuration = 4s
**.energyDetection = -110dBm
cmdenv-output-file = cmd_env_log.txt
**.vector-recording = true

rng-class = "cMersenneTwister"
**.loRaGW[*].numUdpApps = 1
**.loRaGW[*].packetForwarder.localPort = 2000
**.loRaGW[*].packetForwarder.destPort = 1000
**.loRaGW[*].packetForwarder.destAddresses = "networkServer"
**.loRaGW[*].packetForwarder.indexNumber = 1

**.networkServer.numApps = 1
**.networkServer.**.evaluateADRinServer = true
**.networkServer.app[0].typename = "NetworkServerApp"
**.networkServer.app[0].destAddresses = "loRaGW[0]"
**.networkServer.app[0].destPort = 2000
**.networkServer.app[0].localPort = 1000
**.networkServer.app[0].adrMethod = ${"avg"}

**.numberOfPacketsToSend = 80
sim-time-limit = 1d
repeat = 5
**.timeToFirstPacket = exponential(100s)
**.timeToNextPacket = exponential(100s)
**.alohaChannelModel = true
**.mobility = false

**.loRaNodes[*].**.evaluateADRinNode = true
**.loRaNodes[*].**initialLoRaBW = 125 kHz
**.loRaNodes[*].**initialLoRaCR = 4
**.loRaNodes[*].**initialLoRaSF = 12

output-scalar-file = ../results/novo5-80-gw1-s${runnumber}.ini.sca
```

Tout fichier INI doit au minimum définir les sections `[gateways]` et `[nodes]`
avec les coordonnées de chaque entité. Un fichier JSON contenant les listes
`gateways` et `nodes` est également accepté pour décrire des scénarios plus
complexes.

Utilisation d'un environnement de propagation prédéfini :

```python
from VERSION_4.launcher.channel import Channel
suburban = Channel(environment="suburban")
```

Analyse d'un CSV obtenu après simulation :

```bash
python examples/analyse_resultats.py advanced.csv
```

Si plusieurs runs sont regroupés dans un même fichier via le tableau de bord ou
`python VERSION_4/run.py --runs <n> --output results.csv`, utilisez
`analyse_runs.py` pour calculer la moyenne des métriques :

```bash
python examples/analyse_runs.py results.csv
```

## Nettoyage et validation des résultats

```bash
python VERSION_4/launcher/clean_results.py results.csv
pytest -q
```

Les tests vérifient notamment le calcul du RSSI, de l'airtime et la gestion des
collisions.

### Comparaison avec FLoRa

L'outil `VERSION_4/launcher/compare_flora.py` lit les exports du simulateur
FLoRa (fichiers `.sca` ou CSV convertis) et extrait plusieurs métriques : PDR,
histogramme de SF, énergie consommée, débit et collisions. Les scripts
`compare_flora_report.py` et `compare_flora_multi.py` génèrent des rapports pour
visualiser les différences entre plusieurs références et un résultat Python.

La commande suivante calibre automatiquement le canal pour se rapprocher au
mieux d'un export FLoRa :

```bash
python tools/calibrate_flora.py examples/flora_full.csv
```

## Versioning

La version courante se trouve dans `pyproject.toml`. Consultez `CHANGELOG.md`
pour le détail des évolutions. Ce projet est distribué sous licence
[MIT](LICENSE).

## Limites actuelles

- La couche physique reste simplifiée et n'imite pas toutes les imperfections du
  matériel réel.
- La mobilité s'appuie par défaut sur des trajectoires de Bézier ; un modèle
  RandomWaypoint peut utiliser des cartes de terrain pour éviter les obstacles.
- La sécurité LoRaWAN repose désormais sur un chiffrement AES-128 complet avec calcul du MIC. Un serveur de jointure gère l'intégralité de la procédure OTAA.

Les contributions sont bienvenues pour améliorer ces points ou ajouter de
nouvelles fonctionnalités.
