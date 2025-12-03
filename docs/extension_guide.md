# Guide d'extension du tableau de bord de LoRaFlexSim

Ce document explique comment personnaliser `launcher/dashboard.py` du tableau de
bord de LoRaFlexSim et ajouter de nouvelles fonctionnalités au simulateur.

## Principe général

Le tableau de bord repose sur [Panel](https://panel.holoviz.org/) pour
l'affichage et sur `plotly` pour les graphiques. Les options utilisateur sont
définies dans la classe `SimulatorUI`.

## Ajouter un paramètre au tableau de bord

1. Déclarez le nouveau champ dans `SimulatorUI.__init__`.
2. Passez la valeur au constructeur de `Simulator` dans `launch_sim`.
3. Mettez à jour `update_metrics` pour afficher la métrique associée.

Les paramètres radio QoS (activation SNIR, couplage inter-SF, seuils de capture)
suivent déjà cette logique : ils sont instanciés dans `dashboard.py` mais
restent masqués et ignorés tant que le toggle `qos_toggle` est désactivé pour
préserver le modèle ADR historique. Une fois le QoS activé, ces valeurs sont
transmises à `qos_manager.apply`. 【F:loraflexsim/launcher/dashboard.py†L229-L764】【F:loraflexsim/launcher/dashboard.py†L1467-L1537】

Les fonctions existantes illustrent chaque étape en détail.

## Intégrer un module personnalisé

Vous pouvez remplacer les classes du simulateur pour tester d'autres
comportements :

```python
from loraflexsim.launcher import Simulator, PathMobility

class MyMobility(PathMobility):
    def step(self, node, dt):
        # Implémentation spécifique
        super().step(node, dt)

sim = Simulator(mobility_model=MyMobility(...))
```

Les fichiers `gateway.py`, `node.py` et `server.py` peuvent être hérités pour
ajouter de nouvelles logiques.

## Conseils aux contributeurs

Avant de proposer une pull request, vérifiez que `pytest` s'exécute sans échec et ajoutez des tests lorsque c'est pertinent.
