# LoRaFlexSim 1.0.1 : simulateur réseau LoRa (Python 3.10+)

Bienvenue ! **LoRaFlexSim** est un **simulateur complet de réseau LoRa**, inspiré du fonctionnement de FLoRa sous OMNeT++, codé entièrement en Python.
Pour un aperçu des différences avec FLoRa, consultez docs/lorawan_features.md.
Les principales équations sont décrites dans docs/equations_flora.md.
Une synthèse des protocoles ADR est disponible dans docs/adr_protocols.md.

Le protocole `ADR_ML` fournit une stratégie ADR basée sur le machine learning.

Par défaut, le module `Channel` charge la table de bruit de FLoRa en analysant
`flora-master/src/LoRaPhy/LoRaAnalogModel.cc` si le fichier est présent. Cette
table est injectée dans la fonction `_flora_noise_dBm` pour les calculs de
sensibilité. Un chemin personnalisé peut être fourni via `flora_noise_path`.
## 🛠️ Installation

1. **Clonez ou téléchargez** le projet.
2. **Créez un environnement virtuel et installez le projet :**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # PowerShell : .\.venv\Scripts\Activate.ps1
   pip install -e .
   ```

   Sous Windows 11, l'activation via PowerShell requiert parfois l'option
   `-ExecutionPolicy Bypass` (`powershell -ExecutionPolicy Bypass -File ...`).
   Les dépendances Python principales (NumPy, Pandas, SciPy, Matplotlib, Plotly,
   Panel, FastAPI, Uvicorn) sont installées automatiquement par `pip install -e
   .` ; aucune étape supplémentaire n'est nécessaire pour les scripts de
   génération de figures.

   > **Remarque :** les tests automatisés utilisent un stub minimal
   > ``numpy_stub`` situé dans ``tests/stubs`` uniquement pour les tests. Pour
   > exécuter les scripts ou les exemples de LoRaFlexSim, assurez‑vous que la
   > véritable bibliothèque NumPy est installée dans votre environnement.

## ✅ Vérification avant mise à jour

Avant de mettre à jour votre branche ou de soumettre une contribution,
exécutez la commande suivante depuis la racine du dépôt :

```bash
make validate
```

Cette cible `make` enchaîne les suites de tests par domaine (`-k channel`,
`-k class_bc`, etc.) puis lance `scripts/run_validation.py` afin de garantir
qu'aucune régression n'a été introduite. Sur Windows, exécutez cette commande
depuis un terminal disposant de `make` (Git Bash, WSL ou équivalent).

## 🚀 Commandes de lancement recommandées

### Tableau de bord Panel

```bash
panel serve loraflexsim/launcher/dashboard.py --show
```

- Activez le bouton **Mode FLoRa complet** pour verrouiller `flora_mode`,
  `flora_timing`, le modèle physique `omnet_full` et la matrice de capture
  historique.【F:loraflexsim/launcher/dashboard.py†L194-L916】
- Le champ **Graine** garantit un placement et une séquence pseudo-aléatoire
  reproductibles ; **Nombre de runs** permet d'enchaîner automatiquement les
  simulations en incrémentant la graine.
- L'option **Positions manuelles** accepte des entrées `node,id=3,x=120,y=40`
  ou `gw,id=1,x=10,y=80` pour rejouer les coordonnées extraites des INI FLoRa.

### Ligne de commande (`run.py`)

```bash
python -m loraflexsim.run --nodes 30 --gateways 1 --mode random --interval 10 --steps 100 --output resultats.csv
python -m loraflexsim.run --nodes 5 --mode periodic --interval 10 --runs 3 --seed 42
python -m loraflexsim.run --long-range-demo flora_hata --seed 3 --output long_range.csv
```

- Utilisez `--phy-model flora` pour activer la chaîne radio calibrée FLoRa dans
  la CLI et alignez vos paramètres sur les presets fournis par `--long-range-demo`.
- L'équivalent Python consiste à instancier directement `Simulator(...,
  flora_mode=True)` ; le script `examples/run_flora_example.py` fournit un
  scénario prêt à l'emploi basé sur `n100-gw1.ini` et renvoie les métriques
  lorsque `scripts/run_validation.py` est exécuté.【F:examples/run_flora_example.py†L1-L47】【F:scripts/run_validation.py†L1-L220】
- Ajoutez `--seed` pour répéter exactement un run et `--runs <n>` pour calculer
  automatiquement la moyenne des métriques.【F:loraflexsim/run.py†L352-L735】

### Matrice Step 1 sous Windows

Depuis la racine du projet, vous pouvez générer la matrice Step 1 utilisée dans
`scripts/run_step1_matrix.py`. Sous Windows 11, créez/activez votre
environnement virtuel puis lancez :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_step1_matrix_windows.ps1
```

Le script `scripts/run_step1_matrix_windows.ps1` active le venv (par défaut
`.\env`) puis exécute la commande équivalente à :

```powershell
python scripts/run_step1_matrix.py --algos adr apra mixra_h mixra_opt --with-snir true false --seeds 1 2 3 --nodes 1000 5000 --packet-intervals 300 600
```

Les CSV générés sont rangés par état SNIR dans
`results/step1/<snir_state>/seed_<seed>/` avec un suffixe explicite
`_snir-on` ou `_snir-off` sur chaque fichier.【F:scripts/run_step1_matrix.py†L16-L92】

Après l'exécution de la matrice, agrégerez les CSV présents dans
`results/step1/**/*.csv` afin d'obtenir les statistiques nécessaires aux CDF
et aux courbes d'erreur :

```bash
python scripts/aggregate_step1_results.py --strict-snir-detection
```

Le script détecte automatiquement l'état SNIR et génère
`results/step1/summary.csv` (moyennes/écarts-types) ainsi que
`results/step1/raw_index.csv` (index des runs bruts). L'option
`--strict-snir-detection` est recommandée pour refuser les CSV dont l'état
SNIR n'est pas explicite et aligner le pipeline sur la détection stricte
utilisée pour les figures.【F:scripts/aggregate_step1_results.py†L21-L126】

Vous pouvez ensuite tracer les figures Step 1 en comparant directement SNIR
on/off (rouge : SNIR activé, bleu : SNIR désactivé) et en utilisant les
agrégations précédentes :

```bash
python scripts/plot_step1_results.py --compare-snir --use-summary
```

L'option `--compare-snir` est désormais activée par défaut (ajoutez
`--no-compare-snir` pour la désactiver) afin que les figures combinées soient
toujours générées, notamment lorsque vous utilisez `--use-summary`. Les figures
combinées `*_snir_compare_*.png` sont déposées dans `figures/step1/`, et les
éventuelles CDF/figures étendues sont rangées dans `figures/step1/extended/`.
Les **figures officielles** doivent être générées via :

```bash
python scripts/plot_step1_results.py --official --use-summary --plot-cdf
```

Dans ce mode, toutes les figures sont déposées exclusivement dans
`figures/step1/extended/`, qui constitue désormais la **source officielle**.
Les couleurs par défaut restent **rouge** pour SNIR activé et **bleu** pour
SNIR désactivé (`#d62728` et `#1f77b4`), telles que définies dans
`scripts/plot_step1_results.py`.【F:scripts/plot_step1_results.py†L16-L24】【F:scripts/plot_step1_results.py†L450-L520】

### Récapitulatif complet (commande unique + agrégation + tracé)

Si vous souhaitez enchaîner l'exécution de la matrice, son agrégation puis le
tracé comparatif, le workflow complet se résume à :

```bash
python scripts/run_step1_matrix.py --algos adr apra mixra_h mixra_opt --with-snir true false --seeds 1 2 3 --nodes 1000 5000 --packet-intervals 300 600
python scripts/aggregate_step1_results.py --strict-snir-detection
python scripts/plot_step1_results.py --compare-snir --use-summary
```

- Les CSV intermédiaires sont écrits dans `results/step1/<snir_state>/seed_*`.
- Les figures combinées `*_snir_compare_*.png` sont générées dans
  `figures/step1/` (les variantes étendues dans `figures/step1/extended/`).
- Les codes couleur restent **rouge** pour SNIR activé et **bleu** pour SNIR
  désactivé (`#d62728` et `#1f77b4`).

Pour vérifier rapidement que les courbes combinées SNIR on/off restent
fonctionnelles, lancez le test léger suivant :

```bash
python scripts/validate_snir_plots.py
```

Le script exécute deux simulations courtes (SNIR activé/désactivé), agrège
les CSV puis s'assure qu'au moins une figure `*_snir_compare_*.png` est
générée dans `figures/step1/`.

### Banc QoS par clusters

Le scénario de validation QoS multi-algorithmes est automatisé via les scripts
`scripts/run_qos_cluster_bench.py` (simulations) et
`scripts/run_qos_cluster_pipeline.py` (simulations + figures). Consultez
[`docs/qos_cluster_validation_pipeline.md`](docs/qos_cluster_validation_pipeline.md)
pour les commandes complètes, les préréglages (rapide, intermédiaire, complet)
et la checklist de vérification.【F:scripts/run_qos_cluster_bench.py†L1-L119】【F:scripts/run_qos_cluster_pipeline.py†L1-L153】【F:docs/qos_cluster_validation_pipeline.md†L1-L142】

Pour générer manuellement les figures QoS à partir d'un dossier de résultats,
utilisez la CLI dédiée :

```bash
python -m qos_cli.lfs_plots --in results/qos --config qos_cli/config.yaml --out figures/qos
```

Les figures générées incluent désormais des barres d'erreur sur le PDR/DER,
un découpage succès/collisions/sans-signal, ainsi que des vues SNIR
(histogramme+CDF, PDR par bin de SNIR) et une dispersion collision vs charge,
avec une légende globale harmonisée par méthode.

> `qos_cli/lfs_plots.py` peut également être exécuté directement (`python qos_cli/lfs_plots.py`),
> mais l'exécution en tant que module est recommandée pour garantir des imports cohérents.

### API FastAPI + WebSocket

```bash
uvicorn loraflexsim.launcher.web_api:app --reload
```

- `POST /simulations/start` accepte des paramètres `Simulator`, par exemple :
  ```bash
  curl -X POST http://localhost:8000/simulations/start \
    -H "Content-Type: application/json" \
    -d '{"command": "start_sim", "params": {"nodes": 50, "gateways": 1, "flora_mode": true, "steps": 3600}}'
  ```
- `GET /simulations/status` retourne l'état (`idle|running|stopped`) et les
  métriques cumulées ; le WebSocket `/ws` diffuse les métriques en temps réel
  pour alimenter des tableaux de bord personnalisés.【F:loraflexsim/launcher/web_api.py†L23-L84】

## Reproduire FLoRa

Pour aligner strictement LoRaFlexSim sur les scénarios FLoRa, assurez-vous que
les paramètres suivants sont appliqués lors de la création du `Simulator` ou du
`Channel` :

> 📘 Consultez également [`docs/reproduction_flora.md`](docs/reproduction_flora.md)
> pour une checklist détaillée des paramètres et du mode compatibilité.

| Phénomène | Paramètres/équations clés | Modules concernés | Détails complémentaires |
| --- | --- | --- | --- |
| **Pertes radio** | Courbes log-normales (`flora_loss_model`), profils `environment="flora"` et variantes Hata/Oulu | [`loraflexsim/launcher/channel.py`](loraflexsim/launcher/channel.py) | Synthèse des équations de pertes, variables `PL(d)` et constantes dans [`docs/equations_flora.md`](docs/equations_flora.md#pertes-radio) |
| **Bruit & sensibilité** | Seuils `detection_threshold_dBm`, `energy_detection_dBm`, bruit thermique de référence | [`loraflexsim/launcher/channel.py`](loraflexsim/launcher/channel.py), [`loraflexsim/launcher/gateway.py`](loraflexsim/launcher/gateway.py) | Dérivation des équations de bruit et tables de sensibilité dans [`docs/equations_flora.md`](docs/equations_flora.md#bruit-et-sensibilite) |
| **Capture & collisions** | Fenêtre de capture (6 symboles), matrice `FLORA_NON_ORTH_DELTA`, logique `capture_mode="aloha"` | [`loraflexsim/launcher/channel.py`](loraflexsim/launcher/channel.py), [`loraflexsim/launcher/gateway.py`](loraflexsim/launcher/gateway.py) | Modélisation C++ et équations de capture détaillées dans [`docs/equations_flora.md`](docs/equations_flora.md#capture-et-collisions) |
| **Probabilité d'erreur (PER)** | Modèles logistique/Croce, paramètres `flora_per_model`, intégration PHY | [`loraflexsim/launcher/channel.py`](loraflexsim/launcher/channel.py), [`loraflexsim/launcher/server.py`](loraflexsim/launcher/server.py) | Formules PER et coefficients normalisés décrits dans [`docs/equations_flora.md`](docs/equations_flora.md#probabilite-derreur) |

- `flora_mode=True` — active automatiquement les courbes logistiques de FLoRa,
  impose le modèle physique `omnet_full`, applique le seuil de détection
  historique et réutilise les presets de propagation « flora » sur l'ensemble
  des canaux gérés par `MultiChannel`.【F:loraflexsim/launcher/simulator.py†L354-L457】
- `use_flora_curves=True` — charge les équations de perte et de PER de FLoRa.
  Ce paramètre est forcé par `flora_mode`, mais peut être activé manuellement
  lorsque seul le canal doit reproduire les courbes historiques.【F:loraflexsim/launcher/simulator.py†L354-L384】
- `detection_threshold_dBm=-110` et `energy_detection_dBm=-90` — valeurs par
  défaut des scénarios FLoRa ; elles sont propagées à tous les canaux et
  passerelles lorsque `flora_mode` est actif, avec un fallback de `-110` dBm si
  une combinaison SF/BW n'est pas définie par la table d'origine.【F:loraflexsim/launcher/simulator.py†L354-L470】【F:loraflexsim/launcher/channel.py†L68-L204】
- **Presets de propagation** — utilisez `environment="flora"`, `"flora_hata"`
  ou `"flora_oulu"` pour sélectionner respectivement la perte log-normale, la
  variante Hata-Okumura ou le modèle Oulu reproduits depuis FLoRa. Ces presets
  partagent les constantes de référence et peuvent être combinés avec
  `flora_loss_model` pour calquer exactement la variante choisie.【F:loraflexsim/launcher/channel.py†L68-L80】
- **Collisions inter-SF et capture** — dès qu'un scénario active `flora_mode`,
  un `phy_model` commençant par `"flora"` ou `use_flora_curves`, LoRaFlexSim
  force `orthogonal_sf=False`, charge `FLORA_NON_ORTH_DELTA` et verrouille la
  fenêtre de capture à 6 symboles comme dans le `LoRaReceiver` C++. Aucun
  paramètre supplémentaire n'est nécessaire pour retrouver les collisions
  inter-SF et l'effet capture historiques.【F:loraflexsim/launcher/simulator.py†L392-L470】【F:loraflexsim/launcher/channel.py†L454-L520】

Les paramètres récents suivent désormais fidèlement la séparation FLoRa entre
**sensibilité** (`detection_threshold_dBm`) et **détection d'énergie**
(`energy_detection_dBm`), ce qui permet aux passerelles d'ignorer les signaux
faibles avant même de tester la sensibilité, tout en conservant le seuil de
sensibilité original à −110 dBm.【F:loraflexsim/launcher/channel.py†L330-L347】【F:loraflexsim/launcher/gateway.py†L162-L238】
Le **mode capture « pure ALOHA »** est disponible via `capture_mode="aloha"`
ou automatiquement dès que `validation_mode="flora"` est activé, garantissant
que toute collision simultanée annule les paquets concernés comme dans les
traces historiques.【F:loraflexsim/launcher/simulator.py†L411-L415】【F:loraflexsim/launcher/gateway.py†L219-L238】 Enfin, la
sélection du **modèle de PER** passe par `flora_per_model` (logistique, Croce ou
désactivé) et reflète les équations du module `FloraPHY` pour contrôler la
perte aléatoire introduite dans les scénarios de compatibilité.【F:loraflexsim/launcher/channel.py†L273-L278】【F:loraflexsim/launcher/flora_phy.py†L149-L161】

Pour étendre fidèlement les scénarios FLoRa au-delà de 10 km, utilisez les
presets longue portée fournis par `run.py`. Par exemple :

```bash
python run.py --long-range-demo very_long_range --seed 3  # couverture 15 km
python run.py --long-range-demo flora_hata --seed 1        # reproduction terrain FLoRa (10-12 km)
```

Ces commandes activent automatiquement la matrice inter-SF historique, la
fenêtre de capture FLoRa et les réglages d'émission adaptés au preset choisi.

## Classes B & C

La pile LoRaWAN embarquée dans LoRaFlexSim reproduit les mécanismes clés des
classes B et C afin de valider les scénarios à fenêtres descendantes.

### Classe B : beacons et ping slots

- Le simulateur planifie un beacon toutes les `128 s` et diffuse des événements
  `PING_SLOT` pour chaque nœud de classe B selon la périodicité signalée par la
  commande `PingSlotInfoReq`. Les pertes de beacon et la dérive d’horloge sont
  modélisées via `beacon_loss_prob` et `beacon_drift` sur chaque nœud.【F:loraflexsim/launcher/simulator.py†L432-L470】【F:loraflexsim/launcher/simulator.py†L1416-L1488】【F:loraflexsim/launcher/node.py†L65-L217】
- `DownlinkScheduler.schedule_class_b` utilise `next_ping_slot_time` pour
  réserver le créneau immédiatement disponible en tenant compte de la durée de
  la trame et de l’occupation de la passerelle, garantissant une file d’attente
  réaliste des downlinks.【F:loraflexsim/launcher/downlink_scheduler.py†L1-L83】【F:loraflexsim/launcher/lorawan.py†L835-L889】
- Les commandes MAC `PingSlotChannelReq`, `PingSlotInfoReq`, `BeaconFreqReq`
  et `BeaconTimingReq` mettent à jour la configuration locale (fréquence,
  périodicité, canal) de chaque nœud pour rester conforme aux échanges
  LoRaWAN.【F:loraflexsim/launcher/node.py†L820-L910】

### Classe C : écoute continue

- `schedule_class_c` planifie les downlinks dès qu’un nœud de classe C a un
  message en attente, tout en respectant l’occupation RF de la passerelle et le
  temps de transmission calculé par le canal.【F:loraflexsim/launcher/downlink_scheduler.py†L60-L83】
- Le simulateur quantifie l’écoute quasi continue grâce au paramètre
  `class_c_rx_interval`, qui reprogramme automatiquement une fenêtre de
  réception tant que la simulation est active, tout en comptabilisant l’énergie
  consommée en mode RX.【F:loraflexsim/launcher/simulator.py†L234-L470】【F:loraflexsim/launcher/simulator.py†L1416-L1478】

### Limites connues

- Les commandes modifiant la fréquence de beacon ou des ping slots sont stockées
  sur les nœuds mais ne reconfigurent pas encore la fréquence des transmissions
  descendantes, ce qui impose l’utilisation d’un canal unique pour les essais de
  classe B.【F:loraflexsim/launcher/node.py†L201-L910】【F:loraflexsim/launcher/server.py†L200-L335】
- La classe C repose sur un polling toutes les `class_c_rx_interval` secondes
  plutôt que sur une écoute strictement continue, ce qui peut sous-estimer le
  temps passé en réception pour des applications ultra-denses.【F:loraflexsim/launcher/simulator.py†L234-L470】【F:loraflexsim/launcher/simulator.py†L1416-L1478】

### Exemple complet (run.py)

Une fois le projet installé (`pip install -e .`), le script console `run.py`
expose le module `loraflexsim.run`. Depuis la racine du dépôt, vous pouvez
exécuter la commande suivante pour obtenir une simulation alignée FLoRa :

```bash
python -m loraflexsim.run --nodes 5 --gateways 1 --channels 1 \
  --mode random --interval 120 --steps 3600 --seed 42 --runs 1
```

Ce qui produit la sortie attendue ci-dessous (identique via `python run.py` une
fois le script installé) :

```
Simulation d'un réseau LoRa : 5 nœuds, 1 gateways, 1 canaux, mode=random, intervalle=120.0, steps=3600, first_interval=10.0
Run 1/1 : PDR=100.00% , Paquets livrés=165, Collisions=0, Énergie consommée=2.258 J, Délai moyen=0.00 unités de temps, Débit moyen=7.33 bps
Moyenne : PDR=100.00% , Paquets livrés=165.00, Collisions=0.00, Énergie consommée=2.258 J, Délai moyen=0.00 unités de temps, Débit moyen=7.33 bps
```
【F:loraflexsim/run.py†L406-L503】【7e31c7†L1-L4】

## Grandes distances

Le module `loraflexsim.scenarios.long_range` fournit un scénario « grandes distances »
reproductible basé sur les recommandations `LongRangeParameters`. Chaque preset
choisi via `--long-range-demo` configure automatiquement la puissance d'émission et
les gains d'antenne du couple passerelle/nœud pour garantir des liaisons de 10 à
12 km avec des SF9–12.【F:loraflexsim/scenarios/long_range.py†L9-L74】 Les profils
disponibles sont résumés ci-dessous :

| Preset CLI (`--long-range-demo <preset>`) | P<sub>TX</sub> (dBm) | Gains TX/RX (dBi) | Perte câble (dB) | Effet observé sur la PDR SF12 |
|-------------------------------------------|---------------------:|------------------:|-----------------:|------------------------------|
| `flora` / `flora_hata`                    |                 23.0 |             16/16 |              0.5 | ≈ 75 % : PDR limitée par la marge de sensibilité pour reproduire les mesures historiques.【F:docs/long_range.md†L15-L25】|
| `rural_long_range`                        |                 16.0 |              6/6  |              0.5 | ≈ 96 % : combinaison adaptée aux déploiements ruraux avec une marge RSSI/SNR confortable.【F:docs/long_range.md†L15-L29】|
| `very_long_range`                         |                 27.0 |             19/19 |              0.5 | 100 % : ajoute deux nœuds à 13,5–15 km tout en conservant la marge SF12 au-dessus des sensibilités `FLORA_SENSITIVITY`.【F:docs/long_range.md†L15-L38】|

L'utilitaire `suggest_parameters(area_km2, max_distance_km)` interpole ces presets pour
fournir automatiquement un budget de liaison adapté à une surface cible (km²) et une
distance maximale (km). La CLI expose cette fonctionnalité via `--long-range-auto` :

| Surface cible | Distance max | P<sub>TX</sub> (dBm) | Gains TX/RX (dBi) | Références ancrées | Commande |
|---------------|--------------|---------------------:|------------------:|-------------------|----------|
| 10 km²        | auto (1,58 km) | 16.0 | 6 / 6  | `rural_long_range → rural_long_range` | `python -m loraflexsim.run --long-range-auto 10` |
| 576 km²       | 13 km         | 24.3 | 17 / 17 | `flora → very_long_range` | `python -m loraflexsim.run --long-range-auto 576 13` |
| 1 024 km²     | auto (16 km)  | 27.0 | 19 / 19 | `very_long_range → very_long_range` | `python -m loraflexsim.run --long-range-auto 1024` |

Exécutez `python -m loraflexsim.run --long-range-demo` pour lancer le preset
par défaut `flora_hata` (identique à `python run.py --long-range-demo`). Ajoutez
`rural_long_range` pour les essais de très grande portée avec un PDR SF12 plus
élévé ou `very_long_range` pour valider des liens jusqu'à 15 km. Les suggestions
automatiques (`--long-range-auto <surface> [distance]`) permettent d'explorer des
budgets intermédiaires. Les métriques détaillées, un tableau des marges RSSI pour
10/12/15 km et un utilitaire CLI (`scripts/long_range_margin.py`) sont présentés
dans `docs/long_range.md`.

Le comportement attendu est verrouillé par le test d'intégration
`pytest tests/integration/test_long_range_large_area.py`, qui s'assure que chaque
preset conserve une PDR SF12 ≥ 70 % tout en vérifiant les marges de
sensibilité.【F:tests/integration/test_long_range_large_area.py†L1-L63】 Le test
`test_auto_suggestion_preserves_sf12_reliability` valide en outre que
`suggest_parameters` maintient un PDR SF12 ≥ 70 % pour une surface de 10 km² tout
en restant dans l'enveloppe des presets existants.【F:tests/integration/test_long_range_large_area.py†L65-L88】

## Plan de vérification

Deux commandes permettent de rejouer la matrice de validation et de suivre les
écarts par rapport aux références FLoRa décrites dans `VALIDATION.md` :

1. `pytest tests/integration/test_validation_matrix.py` exécute chaque scénario
   et vérifie que les écarts de PDR, de collisions et de SNR restent dans les
   tolérances définies par scénario.【F:tests/integration/test_validation_matrix.py†L1-L78】
2. `python scripts/run_validation.py --output results/validation_matrix.csv`
   (script disponible via [scripts/run_validation.py](scripts/run_validation.py))
   génère un rapport synthétique et retourne un code de sortie non nul si une
   dérive dépasse une tolérance.【F:scripts/run_validation.py†L1-L112】

Le fichier `results/validation_matrix.csv` regroupe pour chaque scénario les
valeurs simulées (`*_sim`), les références FLoRa (`*_ref`), les écarts (`*_delta`)
et un statut `ok/fail`. Les colonnes `tolerance_*` rappellent les seuils
utilisés par les tests ; le fichier `VALIDATION.md` décrit le contexte de chaque
scénario et la façon d’interpréter les écarts en cas d’échec.【F:results/validation_matrix.csv†L1-L11】【F:VALIDATION.md†L1-L67】

## Exemples d'utilisation avancés

Quelques commandes pour tester des scénarios plus complexes :

```bash
# Simulation multi-canaux avec mobilité
python run.py --nodes 50 --gateways 2 --channels 3 \
  --mobility --steps 500 --output advanced.csv

# Démonstration LoRaWAN avec downlinks
python run.py --lorawan-demo --steps 100 --output lorawan.csv
```

### Exemples classes B et C

Utilisez l'API Python pour tester les modes B et C :

```python
from loraflexsim.launcher import Simulator

# Nœuds en classe B avec slots réguliers
sim_b = Simulator(num_nodes=10, node_class="B", beacon_interval=128,
                  ping_slot_interval=1.0)
sim_b.run(1000)

# Nœuds en classe C à écoute quasi continue
sim_c = Simulator(num_nodes=5, node_class="C", class_c_rx_interval=0.5)
sim_c.run(500)

```

### Scénario de mobilité réaliste

Les déplacements peuvent être rendus plus doux en ajustant la plage de vitesses :

```python
from loraflexsim.launcher import Simulator

sim = Simulator(num_nodes=20, num_gateways=3, area_size=2000.0, mobility=True,
                mobility_speed=(1.0, 5.0))
sim.run(1000)
```

## Duty cycle

LoRaFlexSim applique par défaut un duty cycle de 1 % pour se rapprocher des
contraintes LoRa réelles. Le gestionnaire de duty cycle situé dans
`duty_cycle.py` peut être configuré en passant un autre paramètre `duty_cycle`
à `Simulator` (par exemple `0.02` pour 2 %). Transmettre `None` désactive ce
mécanisme. Les transmissions sont automatiquement retardées pour respecter ce
pourcentage.

## Mobilité optionnelle

La mobilité des nœuds peut désormais être activée ou désactivée lors de la
création du `Simulator` grâce au paramètre `mobility` (booléen). Dans le
`dashboard`, cette option correspond à la case « Activer la mobilité des
nœuds ». Si elle est décochée, les positions des nœuds restent fixes pendant
la simulation.
Lorsque la mobilité est active, les déplacements sont progressifs et suivent
des trajectoires lissées par interpolation de Bézier. La vitesse des nœuds est
tirée aléatoirement dans la plage spécifiée (par défaut 2 à 10 m/s) et peut être
modifiée via le paramètre `mobility_speed` du `Simulator`. Les mouvements sont
donc continus et sans téléportation.
Un modèle `PathMobility` permet également de suivre des chemins définis sur une
grille en évitant les obstacles et peut prendre en compte un relief ainsi que
des hauteurs de bâtiments. L'altitude du nœud est alors mise à jour à chaque
déplacement pour un calcul radio plus réaliste. Ce modèle peut désormais lire
une **carte d'obstacles dynamiques** (fichier JSON) listant les positions,
rayons et vitesses des objets à éviter.
Deux champs « Vitesse min » et « Vitesse max » sont disponibles dans le
`dashboard` pour définir cette plage avant de lancer la simulation.
Plusieurs schémas supplémentaires peuvent être utilisés :
- `RandomWaypoint` gère les déplacements aléatoires en s'appuyant sur une carte
  de terrain et sur des obstacles dynamiques optionnels.
- `PlannedRandomWaypoint` applique la même logique mais choisit un point d'arrivée
  aléatoire puis planifie un chemin en A* pour contourner un relief 3D ou des
  obstacles fixes. Une option `slope_limit` permet d'éviter les pentes trop fortes.
- `TerrainMapMobility` permet désormais de suivre une carte rasterisée en
  pondérant la vitesse par cellule et en tenant compte d'obstacles 3D.
- `PathMobility` et le planificateur A* acceptent également un `slope_limit`
  pour ignorer les transitions dépassant une inclinaison donnée.
- `GaussMarkov` et les traces GPS restent disponibles pour modéliser des
  mouvements plus spécifiques.
- `Trace3DMobility` lit une trace temporelle et suit le relief 3D en bloquant
  les passages au-dessus d'une hauteur maximale.

Le script `scripts/run_mobility_models.py` peut comparer ces modèles. Pour
utiliser `PathMobility`, fournissez la carte des chemins (fichier JSON ou CSV)
avec `--path-map` :

```bash
python scripts/run_mobility_models.py --model path --path-map carte.json
```

Le paramètre `--replicates` répète chaque scénario (valeur par défaut : 5).
Nous recommandons d'exécuter au moins 5 répétitions pour obtenir des
statistiques fiables.

## Multi-canaux

LoRaFlexSim permet d'utiliser plusieurs canaux radio. Passez une instance
`MultiChannel` ou une liste de fréquences à `Simulator` via les paramètres
`channels` et `channel_distribution`. Dans le `dashboard`, réglez **Nb
sous-canaux** et **Répartition canaux** pour tester un partage Round‑robin ou
aléatoire des fréquences entre les nœuds.

## Durée et accélération de la simulation

Le tableau de bord permet maintenant de fixer une **durée réelle maximale** en secondes. Par défaut cette limite vaut `86400` s (24 h). Lorsque cette limite est atteinte, la simulation s'arrête automatiquement. Un bouton « Accélérer jusqu'à la fin » lance l'exécution rapide pour obtenir aussitôt les métriques finales.
**Attention :** cette accélération ne fonctionne que si un nombre fini de paquets est défini. Si le champ *Nombre de paquets* vaut 0 (infini), la simulation ne se termine jamais et l'export reste impossible.
Une fois toutes les transmissions envoyées, l'accélération désactive la mobilité des nœuds restants afin d'éviter un blocage de LoRaFlexSim.

## Suivi de batterie

Chaque nœud peut être doté d'une capacité d'énergie (en joules) grâce au paramètre `battery_capacity_j` du `Simulator`. La consommation est calculée selon le profil d'énergie FLoRa (courants typiques en veille, réception, etc.) puis retranchée de cette réserve. Le champ `battery_remaining_j` indique l'autonomie restante.
Un champ **Capacité batterie (J)** est disponible dans le tableau de bord pour
saisir facilement cette valeur. Indiquez `0` pour une capacité illimitée : ce
nombre est automatiquement converti en `None`, valeur attendue par LoRaFlexSim.

Pour documenter ces tests hors du tableau de bord, le duo `scripts/run_battery_tracking.py` / `scripts/plot_battery_tracking.py` fournit un flux CLI complet : définissez une capacité (`--battery-capacity-j` ou `--node-capacity`), activez l'arrêt automatique avec `--watch-nodes` et `--stop-on-depletion`, puis générez la figure de suivi avec l'option `--annotate-depletion`. Les exécutions successives alimentent `results/battery_tracking.csv`, enrichi pour faciliter l'analyse reproductible.

## Paramètres de LoRaFlexSim

Le constructeur `Simulator` accepte de nombreux arguments afin de reproduire les
scénarios FLoRa. Voici la liste complète des options :

- `num_nodes` : nombre de nœuds à créer lorsque aucun fichier INI n'est fourni.
- `num_gateways` : nombre de passerelles générées automatiquement.
- `area_size` : longueur du côté (m) du carré dans lequel sont placés nœuds et
  passerelles. Pour estimer rapidement la surface couverte : `area_size = 1000`
  représente `1 km²`, `area_size = 2000` correspond à `4 km²` et `area_size =
  5000` couvre `25 km²`.
- `transmission_mode` : `Random` (émissions Poisson) ou `Periodic`.
- `packet_interval` : moyenne ou période fixe entre transmissions (s).
  La valeur par défaut est `100` s.
- `first_packet_interval` : moyenne exponentielle appliquée uniquement au
  premier envoi (`None` pour reprendre `packet_interval`). Par défaut `100` s.
- `interval_variation`: coefficient de jitter appliqué multiplicativement
  à l'intervalle exponentiel (0 par défaut pour coller au comportement FLoRa). L'intervalle est multiplié par `1 ± U` avec `U` échantillonné dans `[-interval_variation, interval_variation]`.
 - Les instants de transmission suivent strictement une loi exponentielle de
   moyenne `packet_interval` lorsque le mode `Random` est sélectionné.
- Tous les échantillons sont conservés ; si une transmission est encore en
  cours, la date tirée est simplement repoussée après son terme. Cette logique
  est implémentée par `ensure_poisson_arrivals` et `schedule_event` à partir de
  la valeur extraite via `parse_flora_interval`.
- Les intervalles restent indépendants des collisions et du duty cycle : le
  prochain tirage Poisson est basé sur le début réel de la dernière émission
  (`last_tx_time`).
- `packets_to_send` : nombre de paquets émis **par nœud** avant arrêt (0 = infini).
- `lock_step_poisson` : pré-génère une séquence Poisson réutilisée entre exécutions (nécessite `packets_to_send`).
- `adr_node` / `adr_server` : active l'ADR côté nœud ou serveur.
- `duty_cycle` : quota d'émission appliqué à chaque nœud (`None` pour désactiver).
- `mobility` : active la mobilité aléatoire selon `mobility_speed`.
- `channels` : instance de `MultiChannel` ou liste de fréquences/`Channel`.
- `channel_distribution` : méthode d'affectation des canaux (`round-robin` ou
  `random`).
- `mobility_speed` : couple *(min, max)* définissant la vitesse des nœuds
  mobiles (m/s).
- `fixed_sf` / `fixed_tx_power` : valeurs initiales communes de SF et puissance.
- `battery_capacity_j` : énergie disponible par nœud (`None` pour illimité ;
  la valeur `0` saisie dans le tableau de bord est convertie en `None`).
- `payload_size_bytes` : taille du payload utilisée pour calculer l'airtime.
- `node_class` : classe LoRaWAN de tous les nœuds (`A`, `B` ou `C`).
- `detection_threshold_dBm` : RSSI minimal pour qu'une réception soit valide.
- `min_interference_time` : durée de chevauchement minimale pour déclarer une
  collision (s). Deux transmissions sont en collision lorsqu'elles partagent la
  même fréquence et le même Spreading Factor tout en se superposant plus
  longtemps que cette valeur.
- `config_file` : chemin d'un fichier INI ou JSON décrivant
  positions, SF et puissance.
- `seed` : graine aléatoire utilisée pour reproduire le placement des nœuds et le même ordre statistique des intervalles.
- Chaque composant radio (canal, passerelle, suivi des interférences) reçoit
  désormais son propre générateur pseudo-aléatoire dérivé de `seed` pour
  éviter les corrélations entre fading, placement et génération des collisions.
- `class_c_rx_interval` : période de vérification des downlinks en classe C.
- `beacon_interval` : durée séparant deux beacons pour la classe B (s).
- `ping_slot_interval` : intervalle de base (classe B). L'espacement réel entre
  deux ping slots consécutifs est `ping_slot_interval * 2^(7 - periodicity)` en
  appliquant la périodicité LoRaWAN bornée à `[0, 7]`.
- `ping_slot_offset` : délai après le beacon avant le premier ping slot (s).
- `dump_intervals` : conserve l'historique des dates Poisson et effectives.
  La méthode `dump_interval_logs()` écrit un fichier Parquet par nœud pour
  analyser la planification finale et vérifier empiriquement la loi exponentielle.
- `phase_noise_std_dB` : bruit de phase appliqué au SNR.
- `clock_jitter_std_s` : gigue d'horloge ajoutée à chaque calcul.
- `tx_start_delay_s` / `rx_start_delay_s` : délai d'activation de l'émetteur ou du récepteur.
- `pa_ramp_up_s` / `pa_ramp_down_s` : temps de montée et de descente du PA.

## Paramètres radio avancés

Le constructeur `Channel` accepte plusieurs options pour modéliser plus finement la
réception :

- `cable_loss` : pertes fixes (dB) entre le transceiver et l'antenne.
- `tx_antenna_gain_dB` : gain d'antenne de l'émetteur (dB).
- `rx_antenna_gain_dB` : gain d'antenne du récepteur (dB).
- `receiver_noise_floor` : bruit thermique de référence en dBm/Hz (par défaut
  `-174`). Cette valeur est utilisée directement par le modèle OMNeT++ pour le
  calcul du bruit de fond.
- `noise_figure` : facteur de bruit du récepteur en dB.
- `noise_floor_std` : écart-type de la variation aléatoire du bruit (dB).
- `fast_fading_std` : amplitude du fading multipath en dB.
- `multipath_taps` : nombre de trajets multipath simulés pour un
  fading plus réaliste.
- `fine_fading_std` : écart-type du fading fin corrélé.
- `variable_noise_std` : bruit thermique lentement variable (dB).
- `freq_drift_std_hz` et `clock_drift_std_s` : dérives de fréquence et
  d'horloge corrélées utilisées pour le calcul du SNR.
- `clock_jitter_std_s` : gigue d'horloge ajoutée à chaque calcul.
- `temperature_std_K` : variation de température pour le calcul du bruit.
- `humidity_percent` et `humidity_noise_coeff_dB` : ajoutent un bruit
  supplémentaire proportionnel à l'humidité relative. La variation temporelle
  peut être définie via `humidity_std_percent`.
- `pa_non_linearity_dB` / `pa_non_linearity_std_dB` : modélisent la
  non‑linéarité de l'amplificateur de puissance.
- `pa_non_linearity_curve` : triplet de coefficients polynomiaux pour
  définir une non‑linéarité personnalisée.
- `pa_distortion_std_dB` : variation aléatoire due aux imperfections du PA.
- `pa_ramp_up_s` / `pa_ramp_down_s` : temps de montée et de descente du PA
  influençant la puissance effective.
  Ces paramètres interagissent désormais avec le calcul OMNeT++ pour
  reproduire fidèlement la distorsion du signal.
- `impulsive_noise_prob` / `impulsive_noise_dB` : ajout de bruit impulsif selon
  une probabilité donnée.
- Ces phénomènes sont désormais pris en compte par le modèle OMNeT++ complet
  afin d'obtenir un PER très proche des simulations FLoRa.
- `adjacent_interference_dB` : pénalité appliquée aux brouilleurs situés sur un
  canal adjacent.
- `phase_noise_std_dB` : bruit de phase ajouté au SNR.
- `oscillator_leakage_dB` / `oscillator_leakage_std_dB` : fuite
  d'oscillateur ajoutée au bruit.
- `rx_fault_std_dB` : défauts de réception aléatoires pénalisant le SNR.
- `capture_threshold_dB` : différence de puissance requise pour que le paquet
  le plus fort soit décodé malgré les interférences (≥ 6 dB par défaut).
- `orthogonal_sf` : lorsqu'il vaut `False`, les transmissions de SF différents
  peuvent entrer en collision comme celles du même SF.
- `freq_offset_std_hz` et `sync_offset_std_s` : variations du décalage
  fréquentiel et temporel prises en compte par le modèle OMNeT++.
- Ces offsets corrélés sont désormais appliqués à chaque transmission,
  affinant la synchronisation et rapprochant le PER du comportement FLoRa.
- `dev_frequency_offset_hz` / `dev_freq_offset_std_hz` : dérive propre à
  chaque émetteur.
- `band_interference` : liste de brouilleurs sélectifs sous la forme
 `(freq, bw, dB)` appliqués au calcul du bruit. Chaque entrée définit
 un niveau de bruit spécifique pour la bande concernée.
- `environment` : preset rapide pour le modèle de propagation
  (`urban`, `urban_dense`, `suburban`, `rural`, `rural_long_range`, `indoor`,
  `flora`, `flora_hata` ou `flora_oulu`).
- `phy_model` : "omnet", `"omnet_full"`, "flora", "flora_full" ou `"flora_cpp"` pour utiliser un modèle physique avancé reprenant les formules de FLoRa. Le mode `omnet_full` applique directement les équations du `LoRaAnalogModel` d'OMNeT++ avec bruit variable, sélectivité de canal et une gestion précise des collisions partielles. Le mode `flora_cpp` charge la bibliothèque C++ compilée depuis FLoRa pour une précision accrue.
- `use_flora_curves` : applique directement les équations FLoRa pour la
  puissance reçue et le taux d'erreur.

```python
from loraflexsim.launcher.channel import Channel
canal = Channel(environment="urban")
```

Ces valeurs influencent le calcul du RSSI et du SNR retournés par
`Channel.compute_rssi`.

Le preset `rural_long_range` a été ajouté pour couvrir les scénarios LoRaWAN où
les capteurs se situent à 10–15 km des passerelles. Il abaisse l'exposant de
perte à `γ = 1.7` et allonge la distance de référence à 100 m afin de maintenir
le RSSI proche des seuils FLoRa (−130…−120 dBm) tout en conservant une marge de
sécurité pour le SF12.【F:loraflexsim/launcher/channel.py†L69-L78】  Le tableau
ci-dessous indique les RSSI attendus pour une puissance d'émission de 14 dBm,
calculés sans shadowing pour illustrer la tendance de ce preset.

| Distance (km) | RSSI `rural_long_range` (dBm) |
|---------------|-------------------------------|
| 1             | −108.0                        |
| 5             | −119.9                        |
| 10            | −125.0                        |
| 12            | −126.4                        |
| 15            | −128.0                        |

Un module **`propagation_models.py`** regroupe désormais plusieurs modèles :
`LogDistanceShadowing` pour la perte de parcours classique, `multipath_fading_db`
pour générer un fading Rayleigh, et la nouvelle classe `CompletePropagation`
qui combine ces effets avec un bruit thermique calibré.
Il reprend les paramètres des fichiers INI de FLoRa, par exemple `sigma=3.57` pour le preset *flora*.

```python
from loraflexsim.launcher.propagation_models import CompletePropagation

model = CompletePropagation(environment="flora", multipath_taps=3, fast_fading_std=1.0)
loss = model.path_loss(1000)
fad = model.rssi(14, 1000)  # RSSI avec fading multipath
sense = model.sensitivity_table(125e3)
```


Depuis cette mise à jour, la largeur de bande (`bandwidth`) et le codage
(`coding_rate`) sont également configurables lors de la création d'un
`Channel`. On peut modéliser des interférences externes via `interference_dB`
et simuler un environnement multipath avec `fast_fading_std` et
`multipath_taps`. Des variations
aléatoires de puissance sont possibles grâce à `tx_power_std`. Un seuil de
détection peut être fixé via `detection_threshold_dBm` (par
exemple `-110` dBm comme dans FLoRa) pour ignorer les signaux trop faibles.
Le paramètre `min_interference_time` de `Simulator` permet de définir une durée
de chevauchement sous laquelle deux paquets ne sont pas considérés comme en
collision.

### Modélisation physique détaillée

 Un module optionnel `advanced_channel.py` introduit des modèles de
 propagation supplémentaires inspirés de la couche physique OMNeT++. Le
 mode `cost231` applique la formule Hata COST‑231 avec les hauteurs de
 stations paramétrables et un coefficient d'ajustement via
 `cost231_correction_dB`. Un mode `cost231_3d` tient compte de la distance
 3D réelle et des hauteurs renseignées dans `tx_pos`/`rx_pos`. Un mode
 `okumura_hata` reprend la variante d'origine (urbain, suburbain ou zone
 ouverte) avec un terme correctif `okumura_hata_correction_dB`. Un mode
 `itu_indoor` permet de simuler des environnements intérieurs. Le mode
 `3d` calcule simplement la distance réelle en 3D et les autres modèles
 peuvent également prendre en compte un dénivelé si `tx_pos` et `rx_pos`
 comportent une altitude. Il est également possible de simuler un fading
 `rayleigh`, `rician` ou désormais `nakagami` pour représenter des
multi-trajets plus réalistes. Des gains d'antenne et pertes de câble
peuvent être précisés, ainsi qu'une variation temporelle du bruit grâce
à `noise_floor_std`. Des pertes liées aux conditions météo peuvent être
ajoutées via `weather_loss_dB_per_km`. Cette perte peut varier au cours
du temps en utilisant `weather_loss_std_dB_per_km` et
`weather_correlation`. Un bruit supplémentaire dépendant
de l'humidité peut également être activé grâce aux paramètres
`humidity_percent` et `humidity_noise_coeff_dB`.

```python
from loraflexsim.launcher.advanced_channel import AdvancedChannel
ch = AdvancedChannel(
    propagation_model="cost231_3d",
    terrain="suburban",
    okumura_hata_correction_dB=2.0,
    weather_loss_dB_per_km=1.0,
    weather_loss_std_dB_per_km=0.5,
    fading="nakagami",  # modèle corrélé dans le temps
    obstacle_losses={"wall": 5.0, "building": 20.0},
    modem_snr_offsets={"lora": 0.0},
)
```

L'objet `AdvancedChannel` peut également introduire des offsets de
fréquence et de synchronisation variables au cours du temps pour se
rapprocher du comportement observé avec OMNeT++. Les paramètres
`freq_offset_std_hz` et `sync_offset_std_s` contrôlent l'amplitude de ces
variations corrélées et améliorent la précision du taux d'erreur.
Une non‑linéarité d'amplificateur peut être
spécifiée grâce aux paramètres `pa_non_linearity_dB`,
`pa_non_linearity_std_dB` et `pa_non_linearity_curve`. Le SNR peut en
outre être corrigé par modem à l'aide de `modem_snr_offsets`.

Les autres paramètres (fréquence, bruit, etc.) sont transmis au
constructeur de `Channel` classique et restent compatibles avec le
tableau de bord. Les modèles ``rayleigh`` et ``rician`` utilisent
désormais une corrélation temporelle pour reproduire le comportement de
FLoRa et un bruit variable peut être ajouté via ``variable_noise_std``.
Un paramètre ``clock_jitter_std_s`` modélise la gigue d'horloge sur le
temps de réception. Les dérives ``freq_drift_std_hz`` et ``clock_drift_std_s``
sont gérées en continu, et le démarrage/arrêt du PA peut être simulé via
``tx_start_delay_s``/``rx_start_delay_s`` et ``pa_ramp_*``. Les équations
d'atténuation et de PER de FLoRa peuvent être activées via ``use_flora_curves``
pour un rendu encore plus fidèle. Le capture effect reprend désormais la
logique exacte de la version C++ lorsque ``phy_model`` vaut ``flora``.
Une carte ``obstacle_height_map`` peut bloquer complètement un lien en
fonction de l'altitude parcourue et les différences de hauteur sont
prises en compte dans tous les modèles lorsque ``tx_pos`` et ``rx_pos``
indiquent une altitude.
Une ``obstacle_map`` peut désormais contenir des identifiants (par
exemple ``wall`` ou ``building``) associés à des pertes définies via le
paramètre ``obstacle_losses`` pour modéliser précisément les obstacles
traversés.
Un paramètre ``obstacle_variability_std_dB`` ajoute une variation
temporelle corrélée de cette absorption pour simuler un canal évolutif.
Il est désormais possible de modéliser la sélectivité du filtre RF grâce aux
paramètres ``frontend_filter_order`` et ``frontend_filter_bw``. Une valeur non
nulle applique une atténuation dépendante du décalage fréquentiel via un filtre
Butterworth de même ordre que celui employé dans la pile FLoRa d'OMNeT++.
La sensibilité calculée utilise désormais la largeur de bande du filtre,
si bien qu'un filtre plus étroit réduit le bruit thermique et améliore
automatiquement la portée.

Le tableau de bord propose désormais un bouton **Mode FLoRa complet**. Quand il
est activé, `detection_threshold_dBm` est automatiquement fixé à `-110` dBm et
`min_interference_time` à `5` s, valeurs tirées du fichier INI de FLoRa. Un
profil radio ``flora`` est aussi sélectionné pour appliquer l'exposant et la
variance de shadowing correspondants. Les champs restent modifiables si ce mode
est désactivé. Pour reproduire fidèlement les scénarios FLoRa d'origine, pensez
également à renseigner les positions des nœuds telles qu'indiquées dans l'INI.
L'équivalent en script consiste à passer `flora_mode=True` au constructeur `Simulator`.
Lorsque `phy_model="omnet_full"` est utilisé (par exemple en mode FLoRa), le preset
`environment="flora"` est désormais appliqué automatiquement afin de conserver
un exposant de 2,7 et un shadowing de 3,57 dB identiques au modèle d'origine.
Le capture effect complet du code C++ est alors activé tandis que le PA démarre
et s'arrête selon `tx_start_delay_s`/`rx_start_delay_s` et `pa_ramp_*`. Les
dérives de fréquence ainsi que la gigue d'horloge sont incluses par défaut.

### Aligner le modèle de propagation

Pour n'utiliser que le modèle de propagation de FLoRa, créez le `Simulator`
avec l'option `flora_mode=True`. Ce mode applique automatiquement :

- un exposant de perte de parcours fixé à `2.7` ;
- un shadowing de `σ = 3.57` dB ;
- un seuil de détection d'environ `-110` dBm.
- l'utilisation automatique du modèle `omnet_full`.
- un intervalle moyen de `100` s appliqué si aucun intervalle n'est spécifié.

`Simulator` interprète `packet_interval` et `first_packet_interval` comme les
moyennes d'intervalles exponentiels lorsque le mode **Aléatoire** est actif.
Si ces deux paramètres restent à leurs valeurs par défaut en mode FLoRa, ils
sont automatiquement ramenés à `100` s afin de reproduire le comportement des
scénarios d'origine. Vous pouvez saisir d'autres valeurs dans le tableau de bord
pour personnaliser la fréquence d'émission.

### Équations FLoRa de perte de parcours et de PER

Le module `flora_phy.py` implémente la même perte de parcours que dans FLoRa :

```
loss = PATH_LOSS_D0 + 10 * n * log10(distance / REFERENCE_DISTANCE)
```

avec `PATH_LOSS_D0 = 127.41` dB et `REFERENCE_DISTANCE = 40` m. L'exposant
`n` vaut `2.7` lorsque le profil `flora` est sélectionné. Le taux d'erreur
(PER) est approché par une courbe logistique :

```
PER = 1 / (1 + exp(2 * (snr - (th + 2))))
```

où `th` est le seuil SNR par Spreading Factor ({7: -7.5, 8: -10, 9: -12.5,
10: -15, 11: -17.5, 12: -20} dB). Ces équations sont activées en passant
`phy_model="omnet_full"` ou `use_flora_curves=True` au constructeur du `Channel`.
Lorsque `flora_mode=True`, qu'un modèle physique `phy_model` commençant par
`"flora"` est sélectionné ou que `use_flora_curves=True` est activé, LoRaFlexSim
applique automatiquement cette approximation logistique via `FloraPHY`.
Pour revenir explicitement au modèle analytique de Croce et al. (2018), créez
le canal avec `phy_model="omnet_full"` ou `"omnet"` et laissez
`use_flora_curves=False`.
Pour le mode OMNeT++, le taux d'erreur binaire est déterminé grâce à la
fonction `calculateBER` de `LoRaModulation` transposée telle quelle en
Python afin de reproduire fidèlement les performances de décodage.

### Débogage du bruit/SNR

Les modèles `flora_full` et `flora_cpp` s'appuient désormais sur la table de
bruit issue de `LoRaAnalogModel.cc`. Cela garantit que le SNR retourné par
`Channel.compute_rssi` reste identique entre l'implémentation Python et la
bibliothèque native. Pour inspecter une divergence de SNR :

- vérifiez la valeur de `channel.last_noise_dBm`, mise à jour à chaque appel
  à `compute_rssi` ;
- forcez `processing_gain=True` si vous souhaitez retrouver le calcul
  historique `rssi - bruit + 10·log10(2**sf)` ;
- assurez-vous que le preset CLI sélectionné (ex. `--long-range-demo`) active
  bien les courbes FLoRa (`use_flora_curves=True`) lorsque vous comparez les
  résultats aux traces d'OMNeT++.

Les tests `tests/test_flora_cpp.py` et
`tests/test_flora_equivalence.py` peuvent être exécutés isolément afin de
vérifier la cohérence entre les deux implémentations.

Le paramètre ``flora_loss_model`` permet de choisir parmi plusieurs modèles
d'atténuation : ``"lognorm"`` (par défaut), ``"oulu"`` correspondant à
``LoRaPathLossOulu`` (B = 128.95 dB, n = 2.32, d0 = 1000 m) ou ``"hata"`` pour
``LoRaHataOkumura`` (K1 = 127.5, K2 = 35.2).

Les deux derniers modèles utilisent les expressions suivantes :

```python
# Hata-Okumura
loss = K1 + K2 * log10(distance_km)

# Oulu
loss = B + 10 * n * log10(distance / d0) - antenna_gain
```

Exemple pour une distance de `2` km avec les paramètres par défaut et sans gain
d'antenne :

```text
Hata-Okumura : 127.5 + 35.2 * log10(2) ≈ 138.1 dB
Oulu : 128.95 + 23.2 * log10(2) ≈ 135.9 dB
```


## SF et puissance initiaux

Deux nouvelles cases à cocher du tableau de bord permettent de fixer le
Spreading Factor et/ou la puissance d'émission de tous les nœuds avant le
lancement de la simulation. Une fois la case cochée, sélectionnez la valeur
souhaitée via le curseur associé (SF 7‑12 et puissance 2‑20 dBm). Si la case est
décochée, chaque nœud conserve des valeurs aléatoires par défaut.

## Fonctionnalités LoRaWAN

Une couche LoRaWAN simplifiée est maintenant disponible. Le module
`lorawan.py` définit la structure `LoRaWANFrame` ainsi que les fenêtres
`RX1` et `RX2`. Les nœuds possèdent des compteurs de trames et les passerelles
peuvent mettre en file d'attente des downlinks via `NetworkServer.send_downlink`.

Depuis cette version, la gestion ADR suit la spécification LoRaWAN : en plus des
commandes `LinkADRReq`/`LinkADRAns`, les bits `ADRACKReq` et `ADR` sont pris en
charge, le `ChMask` et le `NbTrans` influencent réellement les transmissions,
le compteur `adr_ack_cnt` respecte le délai `ADR_ACK_DELAY`, est remis à zéro
à chaque downlink et le serveur répond automatiquement lorsqu'un équipement
sollicite `ADRACKReq`. Cette
implémentation est complète et directement inspirée du modèle FLoRa,
adaptée ici sous une forme plus légère sans OMNeT++.

La décision d'ajuster le débit repose sur la marge SNR calculée côté
serveur :

```text
SNRmargin = SNRm - requiredSNR - adrDeviceMargin
Nstep = round(SNRmargin / 3)
```

Avec `SNRm = 5` dB, `requiredSNR = -12.5` dB (SF9) et `adrDeviceMargin = 10` dB,
on obtient `SNRmargin = 7.5` dB et `Nstep = 3`【F:flora-master/src/LoRa/NetworkServerApp.cc†L361-L372】.

Lancer l'exemple minimal :

```bash
python run.py --lorawan-demo
```

L'option `--long-range-demo` prépare quant à elle une topologie de grande aire
avec les gains d'antennes recommandés pour les presets `flora`, `flora_hata` ou
`rural_long_range`. Les métriques produites (PDR par SF, RSSI/SNR SF12) sont
documentées dans [`docs/long_range.md`](docs/long_range.md).

Le tableau de bord inclut désormais un sélecteur **Classe LoRaWAN** permettant de choisir entre les modes A, B ou C pour l'ensemble des nœuds, ainsi qu'un champ **Taille payload (o)** afin de définir la longueur utilisée pour calculer l'airtime. Ces réglages facilitent la reproduction fidèle des scénarios FLoRa.

## Différences par rapport à FLoRa

Cette réécriture en Python reprend la majorité des concepts du modèle OMNeT++
mais simplifie volontairement certains aspects.

**Fonctionnalités entièrement prises en charge**
- respect du duty cycle, effet capture et interférence cumulative
- transmissions multi-canaux et distribution configurable
- mobilité des nœuds avec trajectoires lissées
- consommation d'énergie basée sur le profil FLoRa
- plans de fréquences régionaux prédéfinis (EU868, US915, AU915, AS923, IN865, KR920)
- profils d'énergie personnalisables
- commandes ADR (`LinkADRReq/Ans`, `ADRACKReq`, masque de canaux, `NbTrans`)
- procédure OTAA et file de downlinks programmés
- chiffrement AES-128 avec MIC pour tous les messages
- gestion complète des classes LoRaWAN B et C avec perte de beacon et dérive d'horloge optionnelles

**Fonctionnalités absentes**
- interface graphique OMNeT++ et couche physique détaillée

### Écarts connus avec FLoRa
- le canal radio est désormais plus complet (multipath, interférences
  cumulées et sensibilité par SF calculée automatiquement) mais certains
  paramètres restent approximés
- les calculs détaillés de puissance reçue avec antennes directionnelles et
  l'influence des états TX/RX/IDLE de la radio ne sont pas encore modélisés
- les temporisations et la file d'événements sont maintenant alignées sur
  FLoRa pour un PDR et des délais comparables à ±1 %
- la sensibilité et le bruit thermiques sont maintenant calculés à partir du
  bruit de fond théorique et du facteur de bruit, ce qui se rapproche des
  valeurs des modems Semtech

LoRaFlexSim gère désormais l'ensemble des commandes MAC de LoRaWAN : réglage
des paramètres ADR, réinitialisation de clés, rejoins et changement de classe.

Pour des résultats plus proches du terrain, activez `fast_fading_std` et
`multipath_taps` pour simuler un canal multipath. Utilisez également
`interference_dB` pour introduire un bruit extérieur constant ou variable.

### Effet de capture

Le canal `Channel` applique par défaut un seuil de capture de **6 dB** : un
signal plus fort peut être décodé en présence d'interférences s'il dépasse le
plus faible d'au moins 6 dB et si ce signal domine pendant **cinq symboles de
preambule** au minimum. Dès que vous activez le mode FLoRa (`flora_mode=True`),
choisissez un modèle physique FLoRa (`phy_model` commençant par `"flora"`) ou
demandez les courbes FLoRa (`use_flora_curves=True`), LoRaFlexSim bascule
automatiquement en capture non orthogonale : le simulateur force
`orthogonal_sf=False` et charge la matrice `nonOrthDelta` issue de FLoRa pour
tous les canaux et nœuds, sans recourir à un script ADR externe.【F:loraflexsim/launcher/simulator.py†L392-L470】【F:loraflexsim/launcher/multichannel.py†L8-L51】
La différence de puissance exigée dépend alors des Spreading Factors en
présence.

| SF\Interf. | 7  | 8   | 9   | 10  | 11  | 12  |
|------------|----|-----|-----|-----|-----|-----|
| **7**      | 1  | -8  | -9  | -9  | -9  | -9  |
| **8**      | -11| 1   | -11 | -12 | -13 | -13 |
| **9**      | -15| -13 | 1   | -13 | -14 | -15 |
| **10**     | -19| -18 | -17 | 1   | -17 | -18 |
| **11**     | -22| -22 | -21 | -20 | 1   | -20 |
| **12**     | -25| -25 | -25 | -24 | -23 | 1   |

Un paquet est conservé si `signalRSSI - interferenceRSSI` est supérieur ou égal
à la valeur correspondante. Ainsi, un message SF7 à `-97` dBm face à une
interférence SF9 à `-90` dBm reste décodable car `-97 - (-90) = -7` dB ≥ `-9` dB
【F:flora-master/src/LoRaPhy/LoRaReceiver.h†L60-L67】.

Lorsque vous devez reproduire les scénarios « pure ALOHA » historiques de
FLoRa, forcez la désactivation de tout effet capture avec
`Simulator(capture_mode="aloha")` (ou
`Gateway.start_reception(..., capture_mode="aloha")`). Les scénarios de
validation FLoRa l'activent automatiquement via
`Simulator(validation_mode="flora")`, ce qui court-circuite immédiatement toute
réception dès qu'un chevauchement est détecté, quel que soit l'écart de
puissance.【F:loraflexsim/launcher/simulator.py†L232-L308】【F:loraflexsim/launcher/gateway.py†L197-L236】【F:loraflexsim/validation/__init__.py†L38-L54】

Pour reproduire un scénario FLoRa :
1. Passez `flora_mode=True` et `flora_timing=True` lors de la création du
   `Simulator` (ou activez **Mode FLoRa complet**). Le canal radio utilise alors
   le modèle log-normal de FLoRa avec un fading Rayleigh léger
   (`multipath_taps=3`), un seuil de détection fixé à `-110 dBm` et une fenêtre
   d'interférence minimale de `5 s`. Le délai réseau est également de 10 ms avec
   un traitement serveur de 1,2 s comme dans OMNeT++.
2. Appliquez l'algorithme ADR1 via `from loraflexsim.launcher.adr_standard_1 import apply as adr1` puis `adr1(sim, degrade_channel=True, profile="flora")`.
   Cette fonction reprend la logique du serveur FLoRa original tout en
   remplaçant les canaux idéaux par des `AdvancedChannel` plus réalistes.
3. Spécifiez `adr_method="avg"` lors de la création du `Simulator` (ou sur
   `sim.network_server`) pour utiliser la moyenne des 20 derniers SNR.
4. Fournissez le chemin du fichier INI à `Simulator(config_file=...)` ou
   saisissez les coordonnées manuellement via **Positions manuelles**.
5. Renseignez **Graine** pour conserver exactement le même placement et la même
   séquence d'intervalles d'une exécution à l'autre.
6. Ou lancez `python examples/run_flora_example.py` qui combine ces réglages.

### Compilation de FLoRa (OMNeT++)

Le dossier `flora-master` contient la version originale du simulateur FLoRa.
Après avoir installé OMNeT++ et cloné le framework INET 4.4 à la racine du
projet :

```bash
git clone https://github.com/inet-framework/inet.git -b v4.4 inet4.4
cd inet4.4 && make makefiles && make -j$(nproc)
```

Compilez ensuite FLoRa :

```bash
cd ../flora-master
make makefiles
make -j$(nproc)
```

Pour interfacer LoRaFlexSim avec la couche physique C++, construisez
la bibliothèque partagée `libflora_phy.so` :

```bash
cd ../flora-master
make libflora_phy.so
```

Vous pouvez également exécuter directement `./scripts/build_flora_cpp.sh` depuis
la racine du dépôt pour automatiser cette compilation.

Placez ce fichier à la racine du projet ou dans `flora-master` puis lancez
LoRaFlexSim avec `phy_model="flora_cpp"` pour utiliser ces routines natives.

Exécutez enfin le scénario d'exemple pour générer un fichier `.sca` dans
`flora-master/results` :

```bash
./src/run_flora -u Cmdenv simulations/examples/n100-gw1.ini
```

## Format du fichier CSV

L'option `--output` de `run.py` permet d'enregistrer les métriques de la
simulation dans un fichier CSV. Ce dernier contient l'en‑tête suivant :

```
nodes,gateways,channels,mode,interval,steps,delivered,collisions,PDR(%),energy,avg_delay,throughput_bps
```

* **nodes** : nombre de nœuds simulés.
* **gateways** : nombre de passerelles.
* **channels** : nombre de canaux radio simulés.
* **mode** : `Random` ou `Periodic`.
* **interval** : intervalle moyen/fixe entre deux transmissions.
* **steps** : nombre de pas de temps simulés.
* **delivered** : paquets reçus par au moins une passerelle.
* **collisions** : paquets perdus par collision.
* **PDR(%)** : taux de livraison en pourcentage.
* **energy** : énergie totale consommée (unités arbitraires).
* **avg_delay** : délai moyen des paquets livrés.
* **throughput_bps** : débit binaire moyen des paquets délivrés.

## Analysis example

An example Python script named `analyse_resultats.py` is available in the
`examples` folder. It aggregates several CSV files and plots PDR against the
number of nodes:

```bash
python examples/analyse_resultats.py results1.csv results2.csv
```

The script prints the average PDR and saves a `pdr_by_nodes` figure
as PNG, JPG and EPS.

If the same CSV file contains multiple runs produced with the dashboard or
`run.py --runs`, the `analyse_runs.py` script computes the mean per run:

```bash
python examples/analyse_runs.py results.csv
```

Two other utilities work with the `metrics_*.csv` files exported by the
dashboard:

```bash
python examples/plot_sf_distribution.py metrics1.csv metrics2.csv
python examples/plot_energy.py metrics.csv            # total energy
python examples/plot_energy.py --per-node metrics.csv # per node
python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv
python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv --allowed 50,1 200,3
python scripts/plot_mobility_multichannel.py results/mobility_multichannel.csv --scenarios n50_c1_static n50_c1_mobile
python scripts/plot_mobility_latency_energy.py results/mobility_latency_energy.csv
python scripts/benchmark_energy_classes.py --nodes 20 --packets 5 --output results/energy_classes.csv
python -m scripts.mne3sd.article_a.plots.plot_energy_duty_cycle --results results/mne3sd/article_a
```

Pour exécuter l'ensemble des scénarios et graphiques associés aux articles MNE3SD
en tirant parti du parallélisme des scénarios, utilisez par exemple :

```bash
py -m scripts.mne3sd.run_all_article_outputs --scenario-workers 8
```

`plot_sf_distribution.py` generates `sf_distribution` in PNG, JPG and EPS,
while `plot_energy.py` creates `energy_total` or `energy_per_node` in the
same formats.
`plot_mobility_multichannel.py` saves `pdr_vs_scenario.png`,
`collision_rate_vs_scenario.png` and `avg_energy_per_node_vs_scenario.png`
in the `figures/` folder. Use `--allowed N,C ...` to limit the plot to
specific node/channel pairs and `--scenarios NAME ...` to select particular
scenarios.
`plot_mobility_latency_energy.py` creates `pdr_vs_scenario.svg`,
`avg_delay_vs_scenario.svg` and `avg_energy_per_node_vs_scenario.svg` as
vector graphics.

`benchmark_energy_classes.py` exécute trois simulations dédiées (classes A,
B et C) et exporte un fichier CSV contenant la consommation totale et la
décomposition TX/RX/veille, ce qui permet de comparer rapidement les profils
énergétiques.

La commande `plot_energy_duty_cycle` est détaillée dans la documentation
[« Profils énergétiques »](docs/energy_profiles.md#visualisation-du-cycle-dactivité-énergétique)
et produit automatiquement un résumé (`results/.../energy_consumption_summary.csv`)
et les figures associées (`figures/.../*.png`, `figures/.../*.eps`).
Lorsque les écarts d'énergie moyenne entre classes deviennent très grands,
le script bascule désormais automatiquement l'axe Y du graphique
`energy_per_node_vs_duty_cycle` en échelle logarithmique pour maintenir la
lisibilité.  L'option `--y-scale {auto,linear,log}` permet de forcer un mode
spécifique si besoin.

## Calcul de l'airtime

La durée d'un paquet LoRa est obtenue à partir de la formule théorique :

```
T_sym = 2**SF / BW
T_preamble = (preamble_symbols + 4.25) * T_sym
N_payload = 8 + max(ceil((8*payload_size - 4*SF + 28 + 16) / (4*(SF - 2*DE))), 0)
           * (coding_rate + 4)
T_payload = N_payload * T_sym
airtime = T_preamble + T_payload
```

Chaque entrée de `events_log` comporte `start_time` et `end_time` ; leur
différence représente l'airtime réel du paquet.

```python
from loraflexsim.launcher.channel import Channel
ch = Channel()
temps = ch.airtime(sf=7, payload_size=20)
```


## Nettoyage des résultats

Le script `launcher/clean_results.py` supprime les doublons et les valeurs
manquantes d'un fichier CSV, puis sauvegarde `<fichier>_clean.csv` :

```bash
python launcher/clean_results.py résultats.csv
```

## Validation des résultats

L'exécution de `pytest` permet de vérifier la cohérence des calculs de RSSI et le traitement des collisions :

```bash
pytest -q
```

Un test dédié compare également les résultats de LoRaFlexSim avec ceux du
FLoRa original lorsqu'un fichier `.sca` est disponible :

```bash
pytest tests/test_flora_sca.py -q
```

Vous pouvez aussi comparer les métriques générées avec les formules théoriques détaillées dans `tests/test_simulator.py`.

### Distribution des intervalles

`timeToFirstPacket` et les inter-arrivals suivent la loi `Exp(1/µ_LoRaFlexSim)`. Les tests `tests/test_interval_distribution.py` vérifient que la moyenne reste dans une tolérance de ±2 %, que le coefficient de variation est proche de 1 et que la p‑value du test de Kolmogorov–Smirnov dépasse 0,05. Le duty cycle et la gestion des collisions ne modifient pas cette distribution : seules les transmissions effectives sont retardées, comme le montrent `tests/test_poisson_independence.py`.

Pour suivre les évolutions du projet, consultez le fichier `CHANGELOG.md`.

Ce projet est distribué sous licence [MIT](LICENSE).

## Exemples complets

Plusieurs scripts sont fournis dans le dossier `examples` pour illustrer
l'utilisation de LoRaFlexSim :

```bash
python examples/run_basic.py          # simulation rapide avec 20 nœuds
python examples/run_basic.py --dump-intervals  # exporte les intervalles
python examples/run_flora_example.py  # reproduction d'un scénario FLoRa
python scripts/run_mobility_multichannel.py --nodes 50 --packets 100 --seed 1
python scripts/run_mobility_latency_energy.py --nodes 50 --packets 100 --seed 1
```

L'option `--dump-intervals` active `dump_interval_logs` : un fichier Parquet est
généré pour chaque nœud avec la date Poisson attendue et l'instant réel de
transmission. Ces traces permettent de vérifier empiriquement la distribution
des arrivées.

Le script `run_mobility_multichannel.py` exécute huit scénarios prédéfinis
combinant nombre de nœuds (`N`), canaux (`C`) et mobilité :

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

`run_mobility_latency_energy.py` produit `results/mobility_latency_energy.csv`
que `plot_mobility_latency_energy.py` peut visualiser.

Les utilitaires `analyse_resultats.py` et `analyse_runs.py` aident à traiter les
fichiers CSV produits par `run.py` ou par le tableau de bord.

Le script `scripts/run_qos_comparison.py` orchestre deux simulations (QoS activé
vs ADR désactivé) en s'appuyant sur `loraflexsim.scenarios.run_qos_vs_adr`. Il
produit automatiquement les métriques et journaux correspondants dans
`results/qos_comparison/` afin de faciliter l'analyse des gains liés au QoS.

## Guide d'extension du dashboard

Le fichier [docs/extension_guide.md](docs/extension_guide.md) détaille comment
ajouter des options au tableau de bord et intégrer vos propres modules. Ce guide
vise à faciliter les contributions extérieures.

## Accélérations pour les runs exploratoires

- `python -m loraflexsim.run --fast` réduit automatiquement la durée simulée et
  le nombre de nœuds (avec un minimum de 600 s) afin de valider rapidement une
  configuration. L'option `--sample-size` accepte en complément une fraction de
  durée pour tronquer explicitement un run.
- Le script `scripts/run_all_fast.sh` enchaîne plusieurs scénarios
  représentatifs en mode rapide et constitue une vérification de fumée avant
  des campagnes lourdes.
- `scripts/profile_simulation.py` encapsule `cProfile` :

  ```bash
  python scripts/profile_simulation.py --output stats.prof -- --nodes 200 --steps 86400 --fast
  ```

  Un résumé cumulé est affiché en console et le fichier `stats.prof` peut être
  exploré avec `snakeviz` ou `pstats`.
- Le canal LoRa possède désormais un cache optionnel (`Channel.enable_propagation_cache`)
  pour réutiliser les pertes de propagation entre paires immobiles.
- `scripts/simulation_analysis_utils.py` propose `export_lightweight_trace` et
  `cache_metrics_ready` afin de produire des CSV/Parquet allégés ainsi que des
  métriques agrégées prêtes pour les scripts de traçage.

## Améliorations possibles

Les points suivants ont été intégrés à LoRaFlexSim :

- **PDR par nœud et par type de trafic.** Chaque nœud maintient l'historique de ses vingt dernières transmissions afin de calculer un taux de livraison global et récent. Ces valeurs sont visibles dans le tableau de bord et exportées dans un fichier `metrics_*.csv`.
- **Historique glissant et indicateurs QoS.** LoRaFlexSim calcule désormais le délai moyen de livraison ainsi que le nombre de retransmissions sur la période récente.
- **Indicateurs supplémentaires.** La méthode `get_metrics()` retourne le PDR par SF, passerelle, classe et nœud. Le tableau de bord affiche un récapitulatif et l'export produit deux fichiers CSV : un pour les événements détaillés et un pour les métriques agrégées.
  Les décompositions d'énergie exposent également une clé `"ramp"` dédiée aux phases de montée/descente du PA, exportée dans les CSV (`energy_ramp_J_node`) et visible dans le tableau de bord.
 - **Moteur d'événements précis.** La file de priorité gère désormais un délai réseau de 10 ms et un traitement serveur de 1,2 s, reproduisant ainsi fidèlement l'ordonnancement d'OMNeT++.
- **Suivi détaillé des ACK.** Chaque nœud mémorise les confirmations reçues pour appliquer fidèlement la logique ADR de FLoRa.
- **Scheduler de downlinks prioritaire.** Le module `downlink_scheduler.py` organise les transmissions B/C en donnant la priorité aux commandes et accusés de réception.

## Reproduction des figures

Pour générer toutes les figures fournies avec le projet, utilisez:

```bash
python scripts/generate_all_figures.py --nodes 50 --packets 100 --seed 1
```

Les paramètres peuvent aussi être définis dans un fichier INI:

```bash
python scripts/generate_all_figures.py --config mon_fichier.ini
```

Contenu minimal de `mon_fichier.ini` :

```ini
[simulation]
nodes = 50
packets = 100
seed = 1
```

### Abréviations des figures

Les légendes des graphiques utilisent les abréviations suivantes :

- `N` : nombre de nœuds.
- `C` : nombre de canaux.
- `speed` : vitesse des nœuds en m/s.

## Limites actuelles

LoRaFlexSim reste volontairement léger et certaines fonctionnalités manquent
encore de maturité :

- La couche physique est simplifiée et n'imite pas parfaitement les comportements
  réels des modems LoRa.
- La mobilité par défaut s'appuie sur des trajets de Bézier. Un modèle RandomWaypoint et son planificateur A* permettent d'éviter relief et obstacles 3D.
- La sécurité LoRaWAN s'appuie désormais sur un chiffrement AES-128 complet et la validation du MIC. Le serveur de jointure gère l'ensemble de la procédure OTAA.

Les contributions sont les bienvenues pour lever ces limitations ou proposer de
nouvelles idées.
