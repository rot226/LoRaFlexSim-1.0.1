# Validation Traffic Exponentiel

Ce document rassemble les valeurs obtenues en testant la fonction
`traffic.exponential.sample_interval` du simulateur LoRaFlexSim.

Paramètres utilisés :
- `mean_interval` = 10 secondes
- `N` = 10 000 échantillons
- `seed` = 0

Mesures empiriques :
- Moyenne : 10.002 s
- Coefficient de variation : 1.0017
- p-value du test KS : 0.968

Ces résultats proviennent de l’exécution locale des tests `pytest`.
