# Scénarios Article A (A1–A10)

Ce guide décrit les dix scénarios de référence utilisés pour l'article A de la campagne MNE3SD. Chaque entrée précise les paramètres principaux, la commande de simulation, la commande de tracé associée, le chemin de stockage attendu pour le CSV (après renommage) et l'interprétation qualitative attendue.

## Tableau récapitulatif

| Scénario | Script de simulation | CSV renommé | Script de tracé recommandé |
|----------|---------------------|-------------|-----------------------------|
| A1 | `run_class_density_sweep.py` | `results/mne3sd/article_a/A1_class_density_metrics.csv` | `plot_class_density_metrics.py` |
| A2 | `run_class_load_sweep.py` | `results/mne3sd/article_a/A2_class_load_metrics.csv` | `plot_class_load_results.py` |
| A3 | `simulate_pdr_load.py` (ADR, trafic aléatoire) | `results/mne3sd/article_a/A3_pdr_load_adr_random.csv` | `plot_pdr_load_metrics.py` |
| A4 | `simulate_pdr_load.py` (ADR, trafic périodique) | `results/mne3sd/article_a/A4_pdr_load_adr_periodic.csv` | `plot_pdr_load_metrics.py` |
| A5 | `simulate_pdr_load.py` (SF7 fixe) | `results/mne3sd/article_a/A5_pdr_load_sf7.csv` | `plot_pdr_load_metrics.py` |
| A6 | `simulate_pdr_density.py` (ADR) | `results/mne3sd/article_a/A6_pdr_density_adr.csv` | `plot_pdr_density_metrics.py` |
| A7 | `simulate_pdr_density.py` (SF9 fixe) | `results/mne3sd/article_a/A7_pdr_density_sf9.csv` | `plot_pdr_density_metrics.py` |
| A8 | `simulate_pdr_density.py` (SF12 fixe) | `results/mne3sd/article_a/A8_pdr_density_sf12.csv` | `plot_pdr_density_metrics.py` |
| A9 | `simulate_energy_classes.py` | `results/mne3sd/article_a/A9_energy_consumption.csv` (et `_summary.csv`) | `plot_energy_duty_cycle.py` |
| A10 | `run_class_downlink_energy_profile.py` | `results/mne3sd/article_a/A10_class_downlink_energy.csv` | `plot_class_downlink_energy.py` |

> **Astuce :** Après chaque simulation, renommez ou copiez le CSV généré vers le chemin listé ci-dessus afin de préserver un historique par scénario tout en conservant les noms attendus par les scripts de tracé.

---

### Scénario A1 – Densité de nœuds par classe (référence)
- **Paramètres :**
  - Nœuds : 50, 100, 250, 500
  - Paquets par nœud : 40
  - Intervalle d'émission : 300 s
  - Durée effective : jusqu'à ≈ 12 000 s (40 paquets × 300 s)
  - Classes : A, B, C (classe C avec `--class-c-rx-interval` par défaut = 1 s si profil `fast`)
  - Duty-cycle : non contraint (utilisation par défaut du simulateur)
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.run_class_density_sweep \
      --interval 300 --packets 40 --replicates 5 --seed 1 --profile full
  mv results/mne3sd/article_a/class_density_metrics.csv \
      results/mne3sd/article_a/A1_class_density_metrics.csv
  ```
- **Tracé :**
  ```bash
  python -m scripts.mne3sd.article_a.plots.plot_class_density_metrics \
      --input results/mne3sd/article_a/A1_class_density_metrics.csv \
      --figures-dir figures/mne3sd/article_a/class_density --format pdf
  ```
- **Interprétation attendue :** PDR stable (>0,9) pour 50–100 nœuds puis dégradation progressive, particulièrement pour la classe C en raison des fenêtres RX plus fréquentes qui augmentent les collisions et la consommation énergétique par nœud.

### Scénario A2 – Charge temporelle par classe
- **Paramètres :**
  - Nœuds : 50 (par défaut)
  - Paquets par nœud : 40
  - Intervalles explorés : 60, 300, 900 s
  - Durée effective : 2 400 s (intervalle 60) à 36 000 s (intervalle 900)
  - Classes : A, B, C
  - Duty-cycle : non contraint
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.run_class_load_sweep \
      --nodes 50 --packets 40 --interval-list 60 --interval-list 300 --interval-list 900 \
      --replicates 5 --seed 1 --profile full
  mv results/mne3sd/article_a/class_load_metrics.csv \
      results/mne3sd/article_a/A2_class_load_metrics.csv
  ```
- **Tracé :**
  ```bash
  python -m scripts.mne3sd.article_a.plots.plot_class_load_results \
      --input results/mne3sd/article_a/A2_class_load_metrics.csv \
      --figures-dir figures/mne3sd/article_a/class_load --format pdf
  ```
- **Interprétation attendue :** confirmer la chute du PDR à mesure que l'intervalle diminue (charge plus élevée) et observer l'augmentation de l'énergie moyenne par nœud, en particulier pour les classes B/C qui maintiennent davantage d'écoute.

### Scénario A3 – Charge aléatoire avec ADR actif
- **Paramètres :**
  - Nœuds : 100
  - Paquets par nœud : 20
  - Intervalles : 900, 600, 300, 120, 60 s (ordre décroissant par défaut)
  - Durée effective : jusqu'à 18 000 s
  - Classe : mixte (ADR autorisant l'ajustement SF par nœud)
  - Duty-cycle : non contraint
  - Mode de trafic : Poisson (`random`)
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.simulate_pdr_load \
      --nodes 100 --packets 20 --mode random --replicates 5 --seed 3 \
      --adr-node --adr-server --profile full
  mv results/mne3sd/article_a/pdr_load.csv \
      results/mne3sd/article_a/A3_pdr_load_adr_random.csv
  mv results/mne3sd/article_a/pdr_load_summary.csv \
      results/mne3sd/article_a/A3_pdr_load_adr_random_summary.csv
  ```
- **Tracé :**
  ```bash
  python -m scripts.mne3sd.article_a.plots.plot_pdr_load_metrics \
      --input results/mne3sd/article_a/A3_pdr_load_adr_random.csv \
      --figures-dir figures/mne3sd/article_a/pdr_load --format pdf
  ```
- **Interprétation attendue :** l'ADR doit maintenir un PDR supérieur à 0,9 pour des intervalles ≥ 300 s et limiter la collision même sous trafic aléatoire, avec une consommation énergétique modérée.

### Scénario A4 – Charge périodique avec ADR
- **Paramètres :** identiques à A3 sauf pour le mode : `periodic`.
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.simulate_pdr_load \
      --nodes 100 --packets 20 --mode periodic --replicates 5 --seed 3 \
      --adr-node --adr-server --profile full
  mv results/mne3sd/article_a/pdr_load.csv \
      results/mne3sd/article_a/A4_pdr_load_adr_periodic.csv
  mv results/mne3sd/article_a/pdr_load_summary.csv \
      results/mne3sd/article_a/A4_pdr_load_adr_periodic_summary.csv
  ```
- **Tracé :**
  ```bash
  python -m scripts.mne3sd.article_a.plots.plot_pdr_load_metrics \
      --input results/mne3sd/article_a/A4_pdr_load_adr_periodic.csv \
      --figures-dir figures/mne3sd/article_a/pdr_load --format pdf
  ```
- **Interprétation attendue :** le trafic périodique expose davantage de collisions synchrones ; la comparaison avec A3 doit mettre en évidence un léger recul du PDR pour les intervalles les plus courts et une hausse du temps moyen d'attente.

### Scénario A5 – Charge aléatoire en SF7 fixe
- **Paramètres :**
  - Nœuds : 100
  - Paquets : 20
  - Intervalles : 900 à 60 s
  - Durée effective : ≤ 18 000 s
  - Classe : A (équivalent, SF7 imposé)
  - Duty-cycle : non contraint
  - Mode : `random`
  - SF : `--fixed-sf 7` (désactive ADR)
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.simulate_pdr_load \
      --nodes 100 --packets 20 --mode random --fixed-sf 7 --replicates 5 --seed 3 \
      --profile full
  mv results/mne3sd/article_a/pdr_load.csv \
      results/mne3sd/article_a/A5_pdr_load_sf7.csv
  mv results/mne3sd/article_a/pdr_load_summary.csv \
      results/mne3sd/article_a/A5_pdr_load_sf7_summary.csv
  ```
- **Tracé :** identique à A3/A4 en remplaçant le chemin d'entrée.
- **Interprétation attendue :** observer les limites d'un SF unique : PDR élevé aux longs intervalles mais chute notable sous 300 s, avec un temps d'acheminement réduit mais plus de collisions à fort trafic.

### Scénario A6 – Densité spatiale avec ADR
- **Paramètres :**
  - Aire : 4 km² (carré ≈ 2 000 m de côté)
  - Densités de passerelles : 0,25, 0,5, 1,0 gw/km²
  - Nœuds : 50, 100, 200
  - Paquets : 20
  - Intervalle : 300 s
  - Durée effective : ≈ 6 000 s
  - SF : adaptatif (ADR activé)
  - Duty-cycle : non contraint
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.simulate_pdr_density \
      --density 0.25 --density 0.5 --density 1.0 \
      --nodes 50 --nodes 100 --nodes 200 \
      --sf-mode adaptive --packets 20 --interval 300 --replicates 5 --seed 5 \
      --profile full --adr-node --adr-server
  mv results/mne3sd/article_a/pdr_density.csv \
      results/mne3sd/article_a/A6_pdr_density_adr.csv
  mv results/mne3sd/article_a/pdr_density_summary.csv \
      results/mne3sd/article_a/A6_pdr_density_adr_summary.csv
  ```
- **Tracé :**
  ```bash
  python -m scripts.mne3sd.article_a.plots.plot_pdr_density_metrics \
      --input results/mne3sd/article_a/A6_pdr_density_adr.csv \
      --figures-dir figures/mne3sd/article_a/pdr_density --format pdf
  ```
- **Interprétation attendue :** l'ADR doit permettre un PDR > 0,9 dès 0,5 gw/km² pour ≤ 100 nœuds. Les densités plus faibles ou ≥ 200 nœuds doivent illustrer la nécessité d'une couverture renforcée.

### Scénario A7 – Densité spatiale en SF9 fixe
- **Paramètres :** identiques à A6 mais `--sf-mode fixed9`.
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.simulate_pdr_density \
      --density 0.25 --density 0.5 --density 1.0 \
      --nodes 50 --nodes 100 --nodes 200 \
      --sf-mode fixed9 --packets 20 --interval 300 --replicates 5 --seed 5 \
      --profile full
  mv results/mne3sd/article_a/pdr_density.csv \
      results/mne3sd/article_a/A7_pdr_density_sf9.csv
  mv results/mne3sd/article_a/pdr_density_summary.csv \
      results/mne3sd/article_a/A7_pdr_density_sf9_summary.csv
  ```
- **Tracé :** identique à A6.
- **Interprétation attendue :** illustrer le compromis portée/débit : SF9 améliore le PDR par rapport à SF7 en faible densité mais reste en deçà du mode adaptatif pour les scénarios les plus chargés.

### Scénario A8 – Densité spatiale en SF12 fixe
- **Paramètres :** identiques à A6 mais `--sf-mode fixed12`.
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.simulate_pdr_density \
      --density 0.25 --density 0.5 --density 1.0 \
      --nodes 50 --nodes 100 --nodes 200 \
      --sf-mode fixed12 --packets 20 --interval 300 --replicates 5 --seed 5 \
      --profile full
  mv results/mne3sd/article_a/pdr_density.csv \
      results/mne3sd/article_a/A8_pdr_density_sf12.csv
  mv results/mne3sd/article_a/pdr_density_summary.csv \
      results/mne3sd/article_a/A8_pdr_density_sf12_summary.csv
  ```
- **Tracé :** identique à A6.
- **Interprétation attendue :** SF12 maximise la portée mais accroît le temps à l'antenne ; attendez-vous à un PDR robuste même à 0,25 gw/km², au prix d'une consommation énergétique en hausse et d'un délai moyen supérieur.

### Scénario A9 – Consommation énergétique par classe et duty-cycle
- **Paramètres :**
  - Nœuds : 40
  - Paquets : 40
  - Intervalle : 300 s
  - Durée effective : ≈ 12 000 s
  - Classes : A, B, C
  - Duty-cycles évalués : 1 % (0,01) et 0,1 % (0,001) – ajuster avec `--duty-cycle`
  - Mode : `random`
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.simulate_energy_classes \
      --nodes 40 --packets 40 --interval 300 \
      --classes A --classes B --classes C \
      --duty-cycle 0.01 --duty-cycle 0.001 \
      --replicates 5 --seed 7 --mode random --profile full
  mv results/mne3sd/article_a/energy_consumption.csv \
      results/mne3sd/article_a/A9_energy_consumption.csv
  mv results/mne3sd/article_a/energy_consumption_summary.csv \
      results/mne3sd/article_a/A9_energy_consumption_summary.csv
  ```
- **Tracé :**
  ```bash
  python -m scripts.mne3sd.article_a.plots.plot_energy_duty_cycle \
      --results results/mne3sd/article_a \
      --input results/mne3sd/article_a/A9_energy_consumption_summary.csv \
      --figures-dir figures/mne3sd/article_a/energy_duty_cycle --format pdf
  ```
- **Interprétation attendue :** comparer l'énergie moyenne par message entre classes. La classe C devrait montrer la plus forte sensibilité au duty-cycle, tandis que la classe A reste la plus économe.

### Scénario A10 – Énergie downlink par classe
- **Paramètres :**
  - Nœuds : 100
  - Gateways : 1
  - Durée : 3 600 s (définie par `--duration`)
  - Intervalle uplink : 300 s
  - Période downlink : 600 s (`--downlink-period`)
  - Taille payload uplink/downlink : 12 / 6 octets (valeurs par défaut)
  - Classe C : fenêtres RX toutes les 1 s (`--class-c-rx-interval 1.0`)
  - Duty-cycle : non contraint
- **Simulation :**
  ```bash
  python -m scripts.mne3sd.article_a.scenarios.run_class_downlink_energy_profile \
      --runs 5 --duration 3600 --nodes 100 --packet-interval 300 \
      --downlink-period 600 --beacon-interval 128 --class-c-rx-interval 1.0 \
      --seed 11 --profile full
  mv results/mne3sd/article_a/class_downlink_energy.csv \
      results/mne3sd/article_a/A10_class_downlink_energy.csv
  ```
- **Tracé :**
  ```bash
  python -m scripts.mne3sd.article_a.plots.plot_class_downlink_energy \
      --input results/mne3sd/article_a/A10_class_downlink_energy.csv \
      --figures-dir figures/mne3sd/article_a/class_downlink_energy --format pdf
  ```
- **Interprétation attendue :** vérifier que la classe C absorbe le coût énergétique du downlink (écoute continue), que la classe B amortit les beacons et que la classe A reste quasi inchangée côté RX. Les PDR uplink/downlink doivent rester > 0,9 avec ces paramètres.

---

## Rappel des scripts de visualisation `plot_*`

- `plot_class_density_metrics.py` : compare le PDR et l'énergie par classe en fonction du nombre de nœuds.
- `plot_class_load_results.py` : analyse l'impact de l'intervalle entre messages sur le PDR, la latence et l'énergie.
- `plot_pdr_load_metrics.py` : confronte les scénarios de charge (modes random/périodique, ADR ou SF fixe) en termes de PDR, collisions et énergie.
- `plot_pdr_density_metrics.py` : cartographie les performances en fonction de la densité de passerelles et des stratégies SF.
- `plot_energy_duty_cycle.py` : illustre l'évolution de l'énergie consommée par classe selon le duty-cycle.
- `plot_class_downlink_energy.py` : détaille la consommation TX/RX/veille et les taux de réussite uplink/downlink par classe.

Ces scripts acceptent l'option `--figures-dir` pour isoler les figures (par exemple sous `figures/mne3sd/article_a/`) et `--format` pour choisir l'extension (PDF recommandé pour la publication).
