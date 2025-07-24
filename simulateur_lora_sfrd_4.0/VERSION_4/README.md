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
Définissez la valeur du champ **Graine** pour réutiliser le même placement de
nœuds d'une simulation à l'autre. Le champ **Nombre de runs** permet quant à lui
d'enchaîner automatiquement plusieurs simulations identiques (la graine est
incrémentée à chaque run).
Activez l'option **Positions manuelles** pour saisir les coordonnées exactes de
certains nœuds ou passerelles ; chaque ligne suit par exemple `node,id=3,x=120,y=40`
ou `gw,id=1,x=10,y=80`. Cela permet notamment de reprendre les positions
fournies dans l'INI de FLoRa.
4. **Exécutez des simulations en ligne de commande :**
   ```bash
   python VERSION_4/run.py --nodes 30 --gateways 1 --mode Random --interval 10 --steps 100 --output résultats.csv
   python VERSION_4/run.py --nodes 20 --mode Random --interval 15
   python VERSION_4/run.py --nodes 5 --mode Periodic --interval 10
   ```
    Ajoutez l'option `--seed` pour reproduire exactement le placement des nœuds
    et passerelles.
    Utilisez `--runs <n>` pour exécuter plusieurs simulations d'affilée et
    obtenir une moyenne des métriques.

## Exemples d'utilisation avancés

Quelques commandes pour tester des scénarios plus complexes :

```bash
# Simulation multi-canaux avec mobilité
python VERSION_4/run.py --nodes 50 --gateways 2 --channels 3 \
  --mobility --steps 500 --output advanced.csv

# Démonstration LoRaWAN avec downlinks
python VERSION_4/run.py --lorawan-demo --steps 100 --output lorawan.csv
```

### Exemples classes B et C

Utilisez l'API Python pour tester les modes B et C :

```python
from launcher import Simulator

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
from launcher import Simulator

sim = Simulator(num_nodes=20, num_gateways=3, area_size=2000.0, mobility=True,
                mobility_speed=(1.0, 5.0))
sim.run(1000)
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
Un modèle `PathMobility` permet également de suivre des chemins définis sur une
grille en évitant les obstacles et peut prendre en compte un relief ainsi que
des hauteurs de bâtiments. L'altitude du nœud est alors mise à jour à chaque
déplacement pour un calcul radio plus réaliste.
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
Depuis la version 4.0.1, une fois toutes les transmissions envoyées, l'accélération désactive la mobilité des nœuds restants afin d'éviter un blocage du simulateur.

## Suivi de batterie

Chaque nœud peut être doté d'une capacité d'énergie (en joules) grâce au paramètre `battery_capacity_j` du `Simulator`. La consommation est calculée selon le profil d'énergie FLoRa (courants typiques en veille, réception, etc.) puis retranchée de cette réserve. Le champ `battery_remaining_j` indique l'autonomie restante.
Un champ **Capacité batterie (J)** est disponible dans le tableau de bord pour saisir facilement cette valeur (mettre `0` pour une capacité illimitée).

## Paramètres du simulateur

Le constructeur `Simulator` accepte de nombreux arguments afin de reproduire les
scénarios FLoRa. Voici la liste complète des options :

- `num_nodes` : nombre de nœuds à créer lorsque aucun fichier INI n'est fourni.
- `num_gateways` : nombre de passerelles générées automatiquement.
- `area_size` : dimension (m) du carré dans lequel sont placés nœuds et
  passerelles.
- `transmission_mode` : `Random` (émissions Poisson) ou `Periodic`.
- `packet_interval` : moyenne ou période fixe entre transmissions (s).
- `interval_variation` : coefficient de jitter appliqué à l'intervalle
  exponentiel (0 par défaut pour coller au comportement FLoRa).
- L'intervalle est tronqué à cinq fois `packet_interval` pour éviter des
  écarts trop importants d'une exécution à l'autre.
- `packets_to_send` : nombre de paquets émis **par nœud** avant arrêt (0 = infini).
- `adr_node` / `adr_server` : active l'ADR côté nœud ou serveur.
- `duty_cycle` : quota d'émission appliqué à chaque nœud (`None` pour désactiver).
- `mobility` : active la mobilité aléatoire selon `mobility_speed`.
- `channels` : instance de `MultiChannel` ou liste de fréquences/`Channel`.
- `channel_distribution` : méthode d'affectation des canaux (`round-robin` ou
  `random`).
- `mobility_speed` : couple *(min, max)* définissant la vitesse des nœuds
  mobiles (m/s).
- `fixed_sf` / `fixed_tx_power` : valeurs initiales communes de SF et puissance.
- `battery_capacity_j` : énergie disponible par nœud (`None` pour illimité).
- `payload_size_bytes` : taille du payload utilisée pour calculer l'airtime.
- `node_class` : classe LoRaWAN de tous les nœuds (`A`, `B` ou `C`).
- `detection_threshold_dBm` : RSSI minimal pour qu'une réception soit valide.
- `min_interference_time` : durée de chevauchement minimale pour déclarer une
  collision (s).
- `config_file` : chemin d'un fichier INI ou JSON décrivant
  positions, SF et puissance.
- `seed` : graine aléatoire utilisée uniquement pour reproduire le placement des nœuds et passerelles.
- `class_c_rx_interval` : période de vérification des downlinks en classe C.
- `beacon_interval` : durée séparant deux beacons pour la classe B (s).
- `ping_slot_interval` : intervalle de base entre ping slots successifs (s).
- `ping_slot_offset` : délai après le beacon avant le premier ping slot (s).

## Paramètres radio avancés

Le constructeur `Channel` accepte plusieurs options pour modéliser plus finement la
réception :

- `cable_loss` : pertes fixes (dB) entre le transceiver et l'antenne.
- `receiver_noise_floor` : bruit thermique de référence en dBm/Hz (par défaut
  `-174`).
- `noise_figure` : facteur de bruit du récepteur en dB.
- `noise_floor_std` : écart-type de la variation aléatoire du bruit (dB).
- `fast_fading_std` : amplitude du fading multipath en dB.
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
- `phase_noise_std_dB` : bruit de phase ajouté au SNR.
- `oscillator_leakage_dB` / `oscillator_leakage_std_dB` : fuite
  d'oscillateur ajoutée au bruit.
- `rx_fault_std_dB` : défauts de réception aléatoires pénalisant le SNR.
- `band_interference` : liste de brouilleurs sélectifs sous la forme
  `(freq, bw, dB)` appliqués au calcul du bruit.
- `environment` : preset rapide pour le modèle de propagation
  (`urban`, `urban_dense`, `suburban`, `rural`, `indoor` ou `flora`).

```python
from launcher.channel import Channel
canal = Channel(environment="urban")
```

Ces valeurs influencent le calcul du RSSI et du SNR retournés par
`Channel.compute_rssi`.

Depuis cette mise à jour, la largeur de bande (`bandwidth`) et le codage
(`coding_rate`) sont également configurables lors de la création d'un
`Channel`. On peut modéliser des interférences externes via `interference_dB`
et simuler un environnement multipath avec `fast_fading_std`. Des variations
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
stations paramétrables. Un mode `okumura_hata` reprend la variante
d'origine (urbain, suburbain ou zone ouverte). Un mode `itu_indoor` permet
de simuler des environnements intérieurs. Le mode `3d` calcule la
distance réelle en 3D entre l'émetteur et le récepteur. Il est également
possible de simuler un fading `rayleigh` ou `rician` pour représenter des
multi-trajets plus réalistes. Des gains d'antenne et pertes de câble
peuvent être précisés, ainsi qu'une variation temporelle du bruit grâce
à `noise_floor_std`. Des pertes liées aux conditions météo peuvent être
ajoutées via `weather_loss_dB_per_km`. Cette perte peut varier au cours
du temps en utilisant `weather_loss_std_dB_per_km` et
`weather_correlation`. Un bruit supplémentaire dépendant
de l'humidité peut également être activé grâce aux paramètres
`humidity_percent` et `humidity_noise_coeff_dB`.

```python
from launcher.advanced_channel import AdvancedChannel
ch = AdvancedChannel(
    propagation_model="okumura_hata",
    terrain="suburban",
    weather_loss_dB_per_km=1.0,
    weather_loss_std_dB_per_km=0.5,
    fading="rayleigh",  # modèle corrélé dans le temps
)
```

L'objet `AdvancedChannel` peut également introduire des offsets de
fréquence et de synchronisation variables au cours du temps pour se
rapprocher du comportement observé avec OMNeT++. Les paramètres
`freq_offset_std_hz` et `sync_offset_std_s` contrôlent l'amplitude de ces
variations corrélées.

Les autres paramètres (fréquence, bruit, etc.) sont transmis au
constructeur de `Channel` classique et restent compatibles avec le
tableau de bord. Les modèles ``rayleigh`` et ``rician`` utilisent
désormais une corrélation temporelle pour reproduire le comportement de
FLoRa et un bruit variable peut être ajouté via ``variable_noise_std``.
Une carte ``obstacle_height_map`` peut bloquer complètement un lien en
fonction de l'altitude parcourue.
Il est désormais possible de modéliser la sélectivité du filtre RF grâce aux
paramètres ``frontend_filter_order`` et ``frontend_filter_bw``. Une valeur non
nulle applique une atténuation dépendante du décalage fréquentiel, permettant de
reproduire les effets observés dans OMNeT++.

Le tableau de bord propose désormais un bouton **Mode FLoRa complet**. Quand il
est activé, `detection_threshold_dBm` est automatiquement fixé à `-110` dBm et
`min_interference_time` à `5` s, valeurs tirées du fichier INI de FLoRa. Un
profil radio ``flora`` est aussi sélectionné pour appliquer l'exposant et la
variance de shadowing correspondants. Les champs restent modifiables si ce mode
est désactivé. Pour reproduire fidèlement les scénarios FLoRa d'origine, pensez
également à renseigner les positions des nœuds telles qu'indiquées dans l'INI.
L'équivalent en script consiste à passer `flora_mode=True` au constructeur `Simulator`.

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
python VERSION_4/run.py --lorawan-demo
```

Le tableau de bord inclut désormais un sélecteur **Classe LoRaWAN** permettant de choisir entre les modes A, B ou C pour l'ensemble des nœuds, ainsi qu'un champ **Taille payload (o)** afin de définir la longueur utilisée pour calculer l'airtime. Ces réglages facilitent la reproduction fidèle des scénarios FLoRa.

## Différences par rapport à FLoRa

Cette réécriture en Python reprend la majorité des concepts du modèle OMNeT++
mais simplifie volontairement certains aspects.

**Fonctionnalités entièrement prises en charge**
- respect du duty cycle et effet capture
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
- le canal radio reste moins détaillé (propagation et capture simplifiées)
- absence de modélisation fine du moteur d'événements OMNeT++
- certaines temporisations ou files d'attente du serveur diffèrent
- la sensibilité et le bruit thermiques sont approchés de manière empirique

Le simulateur gère désormais l'ensemble des commandes MAC de LoRaWAN : réglage
des paramètres ADR, réinitialisation de clés, rejoins et changement de classe.

Pour des résultats plus proches du terrain, activez `fast_fading_std` pour
simuler un canal multipath et utilisez `interference_dB` pour introduire un
bruit extérieur constant ou variable.

Pour reproduire un scénario FLoRa :
1. Passez `flora_mode=True` lors de la création du `Simulator` (ou activez
   **Mode FLoRa complet**). Cela applique un seuil de détection à -110 dBm et une
   fenêtre d'interférence de 5 s.
2. Appliquez l'algorithme ADR1 via `from VERSION_4.launcher.adr_standard_1 import apply as adr1` puis `adr1(sim)`.
   Cette fonction reprend la logique du serveur FLoRa original.
3. Fournissez le chemin du fichier INI à `Simulator(config_file=...)` ou
   saisissez les coordonnées manuellement via **Positions manuelles**.
4. Renseignez **Graine** pour conserver exactement le même placement d'une
   exécution à l'autre.
5. Ou lancez `python examples/run_flora_example.py` qui combine ces réglages.
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

## Exemple d'analyse

Un script Python d'exemple nommé `analyse_resultats.py` est disponible dans le
dossier `examples`. Il agrège plusieurs fichiers CSV et trace le PDR en fonction
du nombre de nœuds :

```bash
python examples/analyse_resultats.py resultats1.csv resultats2.csv
```

Le script affiche le PDR moyen puis sauvegarde un graphique dans
`pdr_par_nodes.png`.

Si le même fichier CSV contient plusieurs runs produits avec le dashboard ou
`run.py --runs`, le script `analyse_runs.py` permet d'obtenir les moyennes par
run :

```bash
python examples/analyse_runs.py résultats.csv
```

## Nettoyage des résultats

Le script `launcher/clean_results.py` supprime les doublons et les valeurs
manquantes d'un fichier CSV, puis sauvegarde `<fichier>_clean.csv` :

```bash
python VERSION_4/launcher/clean_results.py résultats.csv
```

## Validation des résultats

L'exécution de `pytest` permet de vérifier la cohérence des calculs de RSSI et le traitement des collisions :

```bash
pytest -q
```

Vous pouvez aussi comparer les métriques générées avec les formules théoriques détaillées dans `tests/test_simulator.py`.

Pour suivre les évolutions du projet, consultez le fichier `CHANGELOG.md`.

Ce projet est distribué sous licence [MIT](../LICENSE).

## Améliorations possibles

Les points suivants ont été intégrés au simulateur :

- **PDR par nœud et par type de trafic.** Chaque nœud maintient l'historique de ses vingt dernières transmissions afin de calculer un taux de livraison global et récent. Ces valeurs sont visibles dans le tableau de bord et exportées dans un fichier `metrics_*.csv`.
- **Historique glissant et indicateurs QoS.** Le simulateur calcule désormais le délai moyen de livraison ainsi que le nombre de retransmissions sur la période récente.
- **Indicateurs supplémentaires.** La méthode `get_metrics()` retourne le PDR par SF, passerelle, classe et nœud. Le tableau de bord affiche un récapitulatif et l'export produit deux fichiers CSV : un pour les événements détaillés et un pour les métriques agrégées.

## Limites actuelles

Le simulateur reste volontairement léger et certaines fonctionnalités manquent
encore de maturité :

- La couche physique est simplifiée et n'imite pas parfaitement les comportements
  réels des modems LoRa.
- La mobilité par défaut s'appuie sur des trajets de Bézier. Un modèle RandomWaypoint peut exploiter une carte de terrain pour éviter les obstacles. Un module de navigation peut désormais planifier des chemins à partir d'une carte d'obstacles.
- La sécurité LoRaWAN s'appuie désormais sur un chiffrement AES-128 complet et la validation du MIC. Le serveur de jointure gère l'ensemble de la procédure OTAA.

Les contributions sont les bienvenues pour lever ces limitations ou proposer de
nouvelles idées.

