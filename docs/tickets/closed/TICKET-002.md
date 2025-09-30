# TICKET-002 – SNIR passerelle pour l’ADR serveur

> ✅ Fermeture : résolu via [PR #241](https://github.com/rot226/LoRaFlexSim/pull/241)
> (merge commit [`9298575`](https://github.com/rot226/LoRaFlexSim/commit/92985757d6b24685abf50c96c5bfee23750831d7))
> incluant la mise à jour principale [`71598d2`](https://github.com/rot226/LoRaFlexSim/commit/71598d21fe66032fda055dd2f0109f6b545e16fb).

## Description
Lors de la réception d’un uplink, le serveur ADR estime le SNR en soustrayant un bruit moyen global `Channel.noise_floor_dBm()` au RSSI remonté par la passerelle.【F:loraflexsim/launcher/server.py†L524-L604】 Dans FLoRa, l’information SNIR par passerelle est transmise directement au serveur (`NetworkServerApp::evaluateADR`), ce qui reflète les conditions radio locales.【F:flora-master/src/LoRa/NetworkServerApp.cc†L300-L370】

## Impact
Avec plusieurs passerelles ou des environnements hétérogènes (bruits différents), l’ADR LoRaFlexSim peut sélectionner des SF/puissances inadaptés : le serveur considère un canal « bruyant » alors que la passerelle qui a décodé la trame dispose d’un SNIR élevé. Cela entraîne des régressions par rapport aux décisions FLoRa.

## Pistes de résolution
- Propager le SNIR calculé par la passerelle (`Gateway.end_reception`) au lieu de recalculer un SNR global côté serveur.
- Stocker l’écart type du bruit par passerelle pour rapprocher l’algorithme de `NetworkServerApp::evaluateADR`.
- Ajouter un test de non-régression comparant l’ADR LoRaFlexSim et FLoRa sur un scénario multi-passerelle.

## Résolution
- La passerelle transmet désormais le SNIR effectivement calculé lors des collisions et le serveur le programme avec `schedule_receive`.
- `NetworkServer.receive` mémorise le SNIR par passerelle et l’intègre directement dans `node.snr_history`, ce qui aligne les décisions ADR `max`, `avg` et `adr-max` sur celles de FLoRa.
- Le test d’intégration `test_adr_standard_alignment_with_flora_trace` vérifie que le SNIR moyen restitué par chaque passerelle dans le scénario multi-GW correspond aux traces FLoRa.
