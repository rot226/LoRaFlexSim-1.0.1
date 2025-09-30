# Modèle d'atténuation par obstacles

Le module `obstacle_loss` de LoRaFlexSim permet d'ajouter des pertes supplémentaires en fonction
des bâtiments ou obstacles présents entre deux nœuds. Une carte d'obstacles peut
être fournie sous forme de fichier GeoJSON ou de raster (matrice de hauteurs).

## Chargement depuis un GeoJSON

```python
from loraflexsim.launcher import Channel, ObstacleLoss

loss = ObstacleLoss.from_geojson("examples/urban_buildings.geojson")
ch = Channel(obstacle_loss=loss)

tx = (0.0, 0.0)
rx = (150.0, 0.0)
dist = 150.0
rssi, snr = ch.compute_rssi(14.0, dist, tx_pos=tx, rx_pos=rx)
```

Chaque entité GeoJSON peut spécifier les propriétés `height` (en mètres) et
`material` (`concrete`, `glass`, `wood`, `brick`, `steel`, `vegetation`). La
perte appliquée est la somme d'une constante selon le matériau et d'un terme
proportionnel à la hauteur.

## Exemple rapide avec une carte raster

```python
raster = [
    [0, 20, 0],
    [0, 30, 0],
    [0, 20, 0],
]
loss = ObstacleLoss.from_raster(raster, cell_size=50, material="concrete")
ch = Channel(obstacle_loss=loss)
```

## Environnement urbain

Dans un scénario urbain, un GeoJSON contenant les bâtiments permet de
représenter les pertes liées aux immeubles. Une carte simple pourrait
ressembler à :

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"height": 25, "material": "concrete"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[0,0],[50,0],[50,40],[0,40],[0,0]]]
      }
    }
  ]
}
```

En chargeant cette carte via `ObstacleLoss`, toute liaison traversant le
bâtiment subira une atténuation additionnelle dépendant de sa hauteur et de
son matériau.
