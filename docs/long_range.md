# Scénario longue portée

Le module `loraflexsim.scenarios.long_range` fournit un scénario reproductible visant à
valider la faisabilité de liaisons LoRa supérieures à 10 km. Une passerelle unique est
placée au centre d'une aire carrée de 24 km de côté (576 km²) afin de positionner trois
nœuds SF12 à 10–11 km et six nœuds supplémentaires entre 4 et 9 km. Le preset
`very_long_range` étend l'aire à 32 km (1 024 km²) et ajoute deux nœuds SF12 à 13,5 et
15 km pour valider la robustesse des liaisons très longues. Les nœuds sont répartis sur
trois largeurs de bande (125/250/500 kHz) et utilisent les SF 9 à 12 pour représenter un
réseau hétérogène.

## Hypothèses radio et recommandations

Le scénario désactive le shadowing (`σ = 0 dB`) pour fournir une référence stable et se
calibre selon le preset radio choisi :

| Preset              | P<sub>TX</sub> (dBm) | Gains TX/RX (dBi) | Perte câble (dB) | PDR SF12* | RSSI max SF12 | SNR max SF12 |
|---------------------|---------------------:|------------------:|-----------------:|----------:|---------------|--------------|
| `flora`             |                 23.0 |             16/16 |             0.5  |     75 %  | −116 dBm      | 0.8 dB       |
| `flora_hata`        |                 23.0 |             16/16 |             0.5  |     75 %  | −116 dBm      | 0.7 dB       |
| `rural_long_range`  |                 16.0 |              6/6  |             0.5  |     96 %  | −105 dBm      | 12.1 dB      |
| `very_long_range`   |                 27.0 |             19/19 |             0.5  |    100 %  | −106 dBm      | 10.8 dB      |

*PDR mesuré avec `packets_per_node=8` et `seed=3`.

Ces réglages correspondent à l'utilisation d'antennes directionnelles (≈16 dBi) pour les
profils `flora`/`flora_hata`, et d'antennes colinéaires (≈6 dBi) pour `rural_long_range`.
Les niveaux de RSSI observés restent largement au‑dessus des sensibilités définies par
`Channel.FLORA_SENSITIVITY` pour chaque combinaison SF/BW, garantissant un PDR SF12 ≥ 70 %
jusqu'à 15 km selon le preset sélectionné.

## Choisir puissance et gains selon la distance

Les presets fournissent des combinaisons de puissance/gain directement exploitables.
La table ci-dessous résume les marges SF12 obtenues avec la largeur de bande 125 kHz :

| Distance cible | Preset | P<sub>TX</sub> (dBm) | Gains TX/RX (dBi) | Marge RSSI estimée* |
|----------------|--------|---------------------:|------------------:|--------------------:|
| 10 km          | `rural_long_range` | 16.0 | 6 / 6  | +28.7 dB |
| 12 km          | `flora_hata`       | 23.0 | 16 / 16 | +10.4 dB |
| 15 km          | `very_long_range`  | 27.0 | 19 / 19 | +21.3 dB |

*Marge calculée via `scripts/long_range_margin.py --preset <preset> --distances 10 12 15`.

Pour explorer des combinaisons intermédiaires, la fonction Python
`suggest_parameters(area_km2, max_distance_km)` interpole ces budgets de liaison à
partir d'une surface cible (km²) et d'une distance maximale (km). L'option
CLI `--long-range-auto` expose directement ce calcul et lance la simulation
associée.

| Surface cible | Distance max | P<sub>TX</sub> (dBm) | Gains TX/RX (dBi) | Références ancrées | Commande CLI |
|---------------|--------------|---------------------:|------------------:|-------------------|--------------|
| 10 km²        | auto (1,58 km) | 16.0 | 6 / 6  | `rural_long_range → rural_long_range` | `python -m loraflexsim.run --long-range-auto 10` |
| 576 km²       | 13 km         | 24.3 | 17 / 17 | `flora → very_long_range` | `python -m loraflexsim.run --long-range-auto 576 13` |
| 1 024 km²     | auto (16 km)  | 27.0 | 19 / 19 | `very_long_range → very_long_range` | `python -m loraflexsim.run --long-range-auto 1024` |

## Exécution et vérification

L'intégration `tests/integration/test_long_range_large_area.py` vérifie que le PDR SF12
reste supérieur à 70 % pour les trois presets tout en contrôlant les marges de RSSI/SNR.
Le test `test_auto_suggestion_preserves_sf12_reliability` complète ce dispositif en
validant que `suggest_parameters` maintient un PDR SF12 ≥ 70 % pour une surface cible
de 10 km² tout en restant aligné sur les presets historiques.

### Parité FLoRa 12 km

Une configuration FLoRa équivalente est fournie dans `flora-master/simulations/examples/long_range_flora.ini`
afin de conserver les distances, SF et puissances attendues par le preset `flora`.【F:flora-master/simulations/examples/long_range_flora.ini†L1-L18】
Le test d'intégration `tests/integration/test_long_range_flora_parity.py` charge ce
fichier, construit le simulateur via `build_long_range_simulator("flora")`, puis compare
les métriques obtenues (PDR, collisions, SNR moyen) à la trace FLoRa
`tests/integration/data/long_range_flora.sca` avec des tolérances resserrées de
±0,01 sur la PDR, 0 collision et 0,2 dB sur le SNR.【F:tests/integration/test_long_range_flora_parity.py†L1-L58】【F:tests/integration/data/long_range_flora.sca†L1-L5】
La simulation LoRaFlexSim reste alignée sur la référence (PDR = 0,903, aucune collision,
SNR moyen −1,94 dB), sans écart à documenter au-delà du bruit de quantification de la
trace FLoRa.【F:tests/integration/data/long_range_flora.sca†L1-L5】
Le scénario peut également être lancé depuis la CLI :

```bash
python -m loraflexsim.run --long-range-demo        # preset par défaut : flora_hata
python -m loraflexsim.run --long-range-demo flora  # forcé sur le preset log-normal
python -m loraflexsim.run --long-range-demo very_long_range --seed 3
python -m loraflexsim.run --long-range-auto 576 13  # interpolation auto pour 13 km sur 24x24 km
```

### Exemple de configuration CLI

```bash
python -m loraflexsim.run \
  --long-range-demo rural_long_range \
  --seed 3 \
  --runs 2 \
  --output results/long_range_rural.csv
```

Cette configuration reproduit les hypothèses `LongRangeParameters` du preset
`rural_long_range`, exécute deux runs consécutifs avec la graine utilisée par les tests
d'intégration et enregistre un récapitulatif des PDR et marges RSSI/SNR dans `results/`.

Le script affiche la PDR agrégée, les métriques par SF et la marge RSSI/SNR maximale
mesurée sur les paquets SF12. Pour explorer d'autres combinaisons puissance/gain, utilisez
`python scripts/long_range_margin.py --preset very_long_range --distances 10 12 15 --csv results/very_long_range_margins.csv`.
