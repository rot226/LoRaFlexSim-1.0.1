# TICKET-001 – Filtrage fréquentiel du calcul SNIR

> ✅ Fermeture : résolu via [PR #240](https://github.com/rot226/LoRaFlexSim/pull/240)
> (merge commit [`ee7bafb`](https://github.com/rot226/LoRaFlexSim/commit/ee7bafb3f91cca730a1606b382af23ce6ee384c9))
> comprenant la correction principale [`b170412`](https://github.com/rot226/LoRaFlexSim/commit/b1704120a5ed8c978999df9a1ff85151515f35b9).

## Description
Le calcul `OmnetPHY.compute_snrs` additionne l’énergie de toutes les transmissions en collision sans distinguer leur fréquence. Dans un scénario multi-canaux, un paquet reçu sur 868,1 MHz est donc dégradé par des transmissions voisines sur 868,3 MHz alors que FLoRa n’agrège que les signaux partageant exactement la même porteuse et largeur de bande.【F:loraflexsim/launcher/omnet_phy.py†L393-L507】【F:flora-master/src/LoRaPhy/LoRaAnalogModel.cc†L123-L160】

## Impact
Cette approximation provoque des faux positifs de collision/capture et fausse le SNR remonté au serveur, ce qui entraîne des pertes de paquets injustifiées dans les simulations multi-canaux. Les scénarios de validation par rapport au traceur FLoRa divergent dès que plusieurs canaux sont actifs.

## Pistes de résolution
- Ajouter `freq_list` comme paramètre obligatoire de `compute_snrs` et ignorer les transmissions dont la fréquence diffère de celle du signal étudié.
- Aligner la logique sur `LoRaAnalogModel::computeNoise` en levant une exception si des bandes se recouvrent partiellement.
- Étendre la couverture de tests pour valider le cas de collisions sur fréquences distinctes (cf. `tests/test_flora_capture.py`).

## Résolution
- `OmnetPHY.compute_snrs` filtre désormais les transmissions dont la porteuse ou la bande ne correspondent pas avant d'intégrer l'énergie de bruit, et aligne son comportement sur `LoRaAnalogModel::computeNoise` en signalant les bandes partiellement superposées.
- Le simulateur propage la fréquence effective de chaque paquet aux calculs de capture OMNeT++/FLoRa afin que les collisions inter-canaux ne dégradent plus le SNR.
- Le test `test_compute_snrs_ignores_other_channels` couvre deux paquets EU868 sur des fréquences distinctes pour prévenir toute régression.

Cette entrée peut être retirée des notes de tickets une fois la correction fusionnée.
