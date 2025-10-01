# Profils énergétiques et conservation E = V·I·t

LoRaFlexSim s'appuie sur des `EnergyProfile` pour calculer l'énergie
consommée par chaque nœud.  Chaque profil définit une tension d'alimentation,
des courants caractéristiques (`sleep`, `rx`, `listen`, `processing`) ainsi
qu'une table reliant la puissance d'émission au courant TX.  Les phases
transitoires propres à la radio (montée/descente du PA, démarrage, préambule)
peuvent également être décrites via les attributs `ramp_*`, `startup_*` et
`preamble_*`.

Depuis la version courante, toutes les transitions vers les états radio
principaux (TX, RX/écoute, veille) sont recalculées via la méthode
`EnergyProfile.enforce_energy`.  Lorsqu'une durée `t` est fournie à
`Node.add_energy(..., duration_s=t)`, le moteur compare l'énergie transmise
avec la formule physique `E = V · I · t` et remplace automatiquement la valeur
si nécessaire.  Les tests `tests/test_energy_conservation.py` vérifient ce
comportement pour chaque état.

Pour rappel :

- `EnergyProfile.current_for(state)` retourne le courant associé à un état.
- `EnergyProfile.energy_for(state, duration_s, power_dBm)` calcule l'énergie
  attendue pour une durée donnée.
- `EnergyProfile.enforce_energy(...)` garantit la conservation d'énergie en
  recadrant l'accumulation.

Les compteurs exportés dans les CSV et le tableau de bord distinguent les
composantes `tx`, `rx/listen`, `sleep`, `processing`, `startup`, `preamble` et
`ramp`, ce qui facilite l'analyse fine des profils.

## Benchmarks par classe LoRaWAN

Le script [`scripts/benchmark_energy_classes.py`](../scripts/benchmark_energy_classes.py)
permet de comparer rapidement les classes A, B et C.  Il lance trois
simulations indépendantes avec les paramètres fournis et exporte un CSV
contenant :

- le PDR agrégé,
- l'énergie totale consommée (`energy_nodes_J`) et moyenne par nœud,
- les composantes TX/RX/veille agrégées.

Exemple d'utilisation :

```bash
python scripts/benchmark_energy_classes.py --nodes 20 --packets 5 --output results/energy_classes.csv
```

Le fichier généré peut ensuite être tracé avec un tableur ou un outil dédié
afin de documenter et comparer les profils de consommation.

## Visualisation du cycle d'activité énergétique

Pour analyser la consommation issue des scénarios de l'article A, le module
[`scripts.mne3sd.article_a.plots.plot_energy_duty_cycle`](../scripts/mne3sd/article_a/plots/plot_energy_duty_cycle.py)
propose une commande prête à l'emploi :

```bash
python -m scripts.mne3sd.article_a.plots.plot_energy_duty_cycle --results results/mne3sd/article_a
```

Cette commande charge le résumé énergétique généré par les simulations
(`results/.../energy_consumption_summary.csv`) et produit plusieurs figures
comparant le cycle d'activité des nœuds.  Les graphiques exportés se trouvent
dans `figures/.../` au format PNG et EPS (par exemple
`figures/.../energy_duty_cycle.png` et `figures/.../energy_duty_cycle.eps`).

Les options principales permettent de :

- préciser la source des données via `--results` ;
- activer la génération d'un rendu LaTeX (`--latex`) ;
- afficher les figures à l'écran (`--show`) ;
- contrôler l'échelle de l'axe Y avec `--y-scale {auto,linear,log}`.  En mode
  `auto` (valeur par défaut), le script détecte les écarts importants entre
  classes (rapport max/min ≥ 20) et applique automatiquement une échelle
  logarithmique afin de mettre en évidence les courbes de faible consommation.

La commande fonctionne également sous Windows 11 (fenêtre **cmd**) en adaptant
les chemins (`python -m ...`).  Elle facilite ainsi la traçabilité des
consommations mesurées dans l'étude.
