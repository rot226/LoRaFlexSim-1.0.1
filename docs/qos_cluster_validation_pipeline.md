# Banc de validation QoS par clusters

Ce guide décrit les étapes pour vérifier l'implémentation « QoS par clusters »
de LoRaFlexSim. Les scripts fournis orchestrent la génération des scénarios,
le calcul des métriques et la production des graphiques associés. Toutes les
commandes ci-dessous doivent être exécutées depuis la racine du dépôt, dans un
terminal disposant de Python 3.10+ (Windows 11 : PowerShell, Git Bash ou WSL).

## 1. Pré-requis

1. Créer et activer un environnement virtuel :
   ```bash
   python -m venv env
   # Windows (PowerShell) :
   .\\env\\Scripts\\Activate.ps1
   # Linux/macOS :
   source env/bin/activate
   ```
2. Installer LoRaFlexSim en mode édition ainsi que les dépendances graphiques :
   ```bash
   pip install -e .
   ```

## 2. Préréglages de scénarios

Les scripts s'appuient sur trois préréglages. Chaque préréglage couvre tous les
algorithmes (ADR, APRA-like, Aimi-like, MixRA-H, MixRA-Opt) avec les mêmes
paramètres radio (aire circulaire 2,5 km, 1 GW, 8 canaux 125 kHz, SF7–SF12,
capture 1 dB, Rayleigh, payload 20 B, duty-cycle 1 %).

| Nom       | Label                  | Couples (N×T) | Runs (algos×couples) | Durée simulée (par run) | Runtime estimé* |
|-----------|-----------------------|---------------|----------------------|-------------------------|-----------------|
| `quick`   | Scénario rapide        | 2×1           | 10                   | 4 h                     | ~0,5 h          |
| `baseline`| Campagne intermédiaire | 3×2           | 30                   | 12 h                    | ~2,5 h          |
| `full`    | Campagne complète      | 5×3           | 75                   | 24 h                    | ~8 h            |

*Estimations réalisées sur une machine 8 cœurs. La durée réelle dépendra de la
configuration matérielle et du solveur MixRA-Opt.

Pour afficher ce tableau en ligne de commande :
```bash
python -m scripts.run_qos_cluster_bench --list-presets
```

## 3. Pipeline de validation

La commande suivante exécute un scénario complet (simulation + figures) en une
seule étape. Par défaut, le préréglage `quick` est utilisé.

```bash
python -m scripts.run_qos_cluster_pipeline --preset quick --seed 1
```

Options importantes :
- `--preset {quick,baseline,full}` : niveau de couverture.
- `--mixra-solver {auto,greedy}` : sélectionne SciPy (si disponible) ou le
  proxy glouton pour MixRA-Opt.
- `--duration <secondes>` : force la durée maximale simulée (sinon valeur du
  préréglage).
- `--results-dir <dossier>` : emplacement personnalisé des CSV.
- `--figures-dir <dossier>` : emplacement personnalisé des figures.
- `--skip-plots` : n'exécute que les simulations.
- `--quiet` : désactive les impressions intermédiaires.

Les résultats et figures sont rangés par défaut dans `results/qos_clusters/<preset>`
et `figures/qos_clusters/<preset>`.

## 4. Exécution progressive

Pour observer rapidement l'évolution des métriques, exécuter les préréglages
suivants dans l'ordre :

1. **Rapide (~30 min)**
   ```bash
   python -m scripts.run_qos_cluster_pipeline --preset quick --seed 1
   ```
2. **Intermédiaire (~2h30)**
   ```bash
   python -m scripts.run_qos_cluster_pipeline --preset baseline --seed 1
   ```
3. **Complète (~8 h)**
   ```bash
   python -m scripts.run_qos_cluster_pipeline --preset full --seed 1
   ```

Chaque exécution est indépendante ; vous pouvez interrompre la série à tout
moment. Les fichiers existants sont remplacés.

## 5. Génération séparée

Pour lancer uniquement les simulations (sans figures) :
```bash
python -m scripts.run_qos_cluster_bench --preset baseline --seed 1 --quiet
```

Pour générer les figures à partir de résultats existants :
```bash
python -m scripts.qos_cluster_plots --results-dir results/qos_clusters/baseline \\
    --figures-dir figures/qos_clusters/baseline
```

## 6. Vérifications à effectuer

Après chaque campagne :

1. Ouvrir `docs/qos_cluster_bench_report.md` (généré automatiquement) et
   vérifier la checklist PASS/FAIL.
2. Contrôler `results/qos_clusters/<preset>/summary.json` pour confirmer les
   PDR cibles par cluster et identifier le point de rupture.
3. Examiner les figures produites :
   - PDR par cluster (`pdr_clusters_tx_*.png`)
   - DER, débit et énergie (`der_*.png`, `throughput_*.png`, `energy_*.png`)
   - Histogrammes SNIR et SF (`snr_cdf_max_load.png`, `sf_histogram_max_load.png`)
   - Heatmap SF×canal (`heatmap_sf_channel_*.png`)
4. Comparer MixRA-H et MixRA-Opt ; consigner toute violation de PDR dans le
   rapport Markdown.

En cas d'écart, relancer le scénario incriminé en réduisant le preset ou en
utilisant `--node-counts`/`--tx-periods` pour isoler la combinaison fautive.

---

Ce document peut être complété avec des observations et captures des runs
réalisés pour constituer un dossier de validation.
