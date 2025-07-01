# Simulateur Réseau LoRa (Python 3.10+)

Bienvenue ! Ce projet est un **simulateur complet de réseau LoRa**, inspiré du fonctionnement de FLoRa sous OMNeT++, codé entièrement en Python.

## 🛠️ Installation

1. **Clonez ou téléchargez** le projet.
2. **Créez un environnement virtuel et installez les dépendances :**
   ```bash
   python3 -m venv env
   source env/bin/activate  # Sous Windows : env\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Lancez le tableau de bord :**
   ```bash
   panel serve launcher/dashboard.py --show
   ```
4. **Exécutez des simulations en ligne de commande :**
   ```bash
   python run.py --nodes 30 --gateways 1 --area 1000 --mode Random --interval 10 --steps 100 --output resultats.csv
   python run.py --nodes 20 --mode Random --interval 15
   python run.py --nodes 5 --mode Periodic --interval 10
   ```

## Exemples d'utilisation avancés

Quelques commandes pour tester des scénarios plus complexes :

```bash
# Simulation multi-canaux avec mobilité
python run.py --nodes 50 --gateways 2 --area 2000 --channels 3 \
  --mobility --steps 500 --output advanced.csv

# Démonstration LoRaWAN avec downlinks
python run.py --lorawan-demo --steps 100 --output lorawan.csv
```

## Duty cycle

Le simulateur applique par défaut un duty cycle de 1 % pour se rapprocher des
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
Deux champs « Vitesse min » et « Vitesse max » sont disponibles dans le
`dashboard` pour définir cette plage avant de lancer la simulation.

## Multi-canaux

Le simulateur permet d'utiliser plusieurs canaux radio. Passez une instance
`MultiChannel` ou une liste de fréquences à `Simulator` via les paramètres
`channels` et `channel_distribution`. Dans le `dashboard`, réglez **Nb
sous-canaux** et **Répartition canaux** pour tester un partage Round‑robin ou
aléatoire des fréquences entre les nœuds.

## Durée et accélération de la simulation

Le tableau de bord permet maintenant de fixer une **durée réelle maximale** en secondes. Lorsque cette limite est atteinte, la simulation s'arrête automatiquement. Un bouton « Accélérer jusqu'à la fin » lance l'exécution rapide pour obtenir aussitôt les métriques finales.
**Attention :** cette accélération ne fonctionne que si un nombre fini de paquets est défini. Si le champ *Nombre de paquets* vaut 0 (infini), la simulation ne se termine jamais et l'export reste impossible.

## Suivi de batterie

Chaque nœud peut être doté d'une capacité d'énergie (en joules) grâce au paramètre `battery_capacity_j` du `Simulator`. La consommation est calculée selon le profil d'énergie FLoRa (courants typiques en veille, réception, etc.) puis retranchée de cette réserve. Le champ `battery_remaining_j` indique l'autonomie restante.

## Paramètres radio avancés

Le constructeur `Channel` accepte plusieurs options pour modéliser plus finement la
réception :

- `cable_loss` : pertes fixes (dB) entre le transceiver et l'antenne.
- `receiver_noise_floor` : bruit thermique de référence en dBm/Hz (par défaut
  `-174`).
- `noise_figure` : facteur de bruit du récepteur en dB.
- `noise_floor_std` : écart-type de la variation aléatoire du bruit (dB).
- `fast_fading_std` : amplitude du fading multipath en dB.

Ces valeurs influencent le calcul du RSSI et du SNR retournés par
`Channel.compute_rssi`.

Depuis cette mise à jour, la largeur de bande (`bandwidth`) et le codage
(`coding_rate`) sont également configurables lors de la création d'un
`Channel`. On peut modéliser des interférences externes via `interference_dB`
et introduire des variations aléatoires de puissance avec `tx_power_std`.

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
le compteur `adr_ack_cnt` respecte le délai `ADR_ACK_DELAY` et le serveur
répond automatiquement lorsqu'un équipement sollicite `ADRACKReq`. Cette
implémentation est complète et directement inspirée du modèle FLoRa,
adaptée ici sous une forme plus légère sans OMNeT++.

Lancer l'exemple minimal :

```bash
python run.py --lorawan-demo
```

## Format du fichier CSV

L'option `--output` de `run.py` permet d'enregistrer les métriques de la
simulation dans un fichier CSV. Ce dernier contient l'en‑tête suivant :

```
nodes,gateways,area,channels,mode,interval,steps,delivered,collisions,PDR(%),energy,avg_delay,throughput_bps
```

* **nodes** : nombre de nœuds simulés.
* **gateways** : nombre de passerelles.
* **area** : côté du carré de simulation en mètres.
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

## Exemple d'analyse

Un script Python d'exemple nommé `analyse_resultats.py` est disponible dans le
dossier `examples`. Il agrège plusieurs fichiers CSV et trace le PDR en fonction
du nombre de nœuds :

```bash
python examples/analyse_resultats.py resultats1.csv resultats2.csv
```

Le script affiche le PDR moyen puis sauvegarde un graphique dans
`pdr_par_nodes.png`.

## Validation des résultats

L'exécution de `pytest` permet de vérifier la cohérence des calculs de RSSI et le traitement des collisions :

```bash
pytest -q
```

Vous pouvez aussi comparer les métriques générées avec les formules théoriques détaillées dans `tests/test_simulator.py`.

Pour suivre les évolutions du projet, consultez le fichier `CHANGELOG.md`.

Ce projet est distribué sous licence [MIT](../LICENSE).

## Améliorations possibles

Pour aller plus loin, on pourrait :

- **Calculer des PDR par nœud ou par type de trafic.** Chaque nœud dispose déjà d'un historique de ses transmissions. On peut ainsi déterminer un taux de livraison individuel ou différencié suivant la classe ou le mode d'envoi.
- **Conserver un historique glissant pour afficher un PDR moyen sur les dernières transmissions.** Le simulateur stocke désormais les vingt derniers événements de chaque nœud et calcule un PDR « récent ».
- **Ajouter des indicateurs supplémentaires :** PDR par SF, par passerelle et par nœud sont exposés via la méthode `get_metrics()`.
- **Intégrer des métriques de QoS :** délai moyen et nombre de retransmissions sont suivis pour affiner la vision du réseau.
 
