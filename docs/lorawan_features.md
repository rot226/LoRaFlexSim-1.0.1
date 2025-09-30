# Comparaison des fonctionnalités LoRaWAN

Ce document résume les différences entre la simulation FLoRa d'origine
(`flora-master/src/LoRa/`) et l'implémentation Python présente dans `launcher/`.

## Fonctionnalités prises en charge par FLoRa

- Gestion basique des trames LoRa avec accusés de réception.
- Évaluation de l'ADR côté serveur via `NetworkServerApp::evaluateADR`.
- Types de messages ``JOIN_REQUEST`` et ``JOIN_REPLY`` définis dans
  `LoRaAppPacket.msg`.
- Aucune prise en charge intégrée des classes B ou C.
- Pas de chiffrement ou d'authentification des messages.

## Fonctionnalités de l'implémentation Python

- Support complet du protocole LoRaWAN : frames ``LoRaWANFrame``,
  chiffrement AES et calcul du MIC.
- Gestion des classes A, B et C dans `server.py` et `node.py` (beacons,
  ping slots, réception continue).
- Implémentation de nombreuses commandes MAC
  (``LinkADRReq``, ``DeviceTimeReq``, ``PingSlotChannelReq``…).
- Procédure d’activation OTAA via un serveur de join (`JoinServer`).
- Historique SNR et ajustement ADR conforme à la spécification.
- Agrégation ADR multi-passerelle : chaque passerelle alimente une fenêtre
  glissante de SNIR (`gateway_snr_history`) puis le serveur calcule les
  marges `avg`/`max` par passerelle avant de fusionner les recommandations,
  reproduisant les décisions FLoRa même en présence de doublons.【F:loraflexsim/launcher/server.py†L117-L192】【F:tests/integration/test_adr_standard_alignment.py†L32-L96】
- Sélection automatique des data rates de downlink (RX2 et ping slots) en
  fonction de la région LoRaWAN configurée, couverte par des tests unitaires
  dédiés.【F:loraflexsim/launcher/lorawan.py†L9-L22】【F:tests/test_downlink_dr_regions.py†L1-L13】

## Fonctionnalités équivalentes

- Les deux versions permettent l’émission de messages et un accusé de
  réception optionnel.
- Les performances énergétiques s’appuient sur le profil FLoRa dans les
  deux cas.
- Un mécanisme d’ADR est disponible de part et d’autre, bien que
  l’algorithme diffère légèrement.
- Les conditions de réception (uplink comme downlink) reproduisent
  désormais exactement celles de FLoRa, sans suppression aléatoire
  supplémentaire.
- Les modèles de canal appliquent les mêmes courbes de perte, rejettent les
  distances non physiques et tiennent compte des obstacles même avec le PHY
  OMNeT++, des comportements vérifiés par les tests de validation
  dédiés【F:loraflexsim/launcher/channel.py†L12-L41】【F:loraflexsim/launcher/channel.py†L579-L616】【F:tests/test_channel_path_loss.py†L1-L31】【F:tests/test_channel_path_loss_validation.py†L1-L15】.

## Fonctionnalités propres à la version Python

- Sécurité LoRaWAN (chiffrement des charges utiles et MIC).
- Gestion explicite des classes B et C avec planification des downlinks.
- Grand nombre de commandes MAC supplémentaires.
- Activation OTAA avec dérivation dynamique des clés.

### Prérequis pour l'activation OTAA

Pour déclencher une procédure OTAA complète avec chiffrement AES-128 et
validation du MIC, les éléments suivants sont requis :

- une **AppKey** de 16 octets (AES-128) partagée entre le nœud et le
  `JoinServer` ;
- des identifiants **JoinEUI** et **DevEUI** uniques (8 octets chacun)
  utilisés lors de l'enregistrement du périphérique ;
- la configuration d'un `JoinServer` avec un `net_id` cohérent avec le
  réseau simulé, puis l'appel à `JoinServer.register(join_eui, dev_eui,
  app_key)` avant l'émission de la requête d'adhésion ;
- un `DevNonce` inédit pour chaque tentative de join, afin d'éviter les
  rejets pour réutilisation.

Une fois ces prérequis remplis, le serveur dérive les clés de session
(`NwkSKey`, `AppSKey`), chiffre le `JoinAccept` et signe la réponse avec le
MIC attendu. La simulation peut alors poursuivre l'échange applicatif en
sécurité.

## Éléments pouvant affecter la comparaison des métriques

- L’ajout du chiffrement et des en-têtes LoRaWAN augmente la taille des
  paquets, ce qui se traduit par un airtime plus long qu’avec FLoRa.
- L’implémentation des classes B/C introduit des fenêtres de réception
  supplémentaires qui n’existent pas dans FLoRa.
- Les algorithmes ADR ne prennent pas en compte exactement les mêmes
  seuils, entraînant des évolutions de SF ou de puissance différentes.
- Le compteur `adr_ack_cnt` est désormais remis à zéro à chaque downlink
  et la montée en SF ou puissance suit la logique LoRaWAN après
  `adr_ack_limit + adr_ack_delay` transmissions sans réponse.

## Limitations connues et options longue portée

- **Portée élevée :** les presets `flora`, `flora_hata` et `flora_oulu` reprennent fidèlement les constantes de perte FLoRa mais restent calibrés pour des distances de l'ordre de 10 km. Pour dépasser cette portée tout en conservant les sensibilités d'origine, sélectionnez `environment="rural_long_range"` ou `flora_loss_model` correspondant, qui abaissent l'exposant de perte et le shadowing selon les valeurs documentées dans `Channel.ENV_PRESETS`.【F:loraflexsim/launcher/channel.py†L68-L80】 Les tests de régression `test_long_range_presets` vérifient que ces presets maintiennent un PDR valide au-delà de 5 km en mode FLoRa.【F:tests/test_long_range_presets.py†L1-L55】
- **Modèle de bruit :** la table `FLORA_SENSITIVITY` ne couvre que les combinaisons SF/BW utilisées par FLoRa et retombe sur `-110` dBm lorsque la paire demandée est absente. Les valeurs de bruit chargées via `parse_flora_noise_table` reproduisent fidèlement `LoRaAnalogModel.cc` mais ne modélisent pas d'évolution spatiale supplémentaire.【F:loraflexsim/launcher/channel.py†L93-L133】
- **Alignement FLoRa :** les scénarios `flora_mode=True` demeurent vérifiés contre les traces `.sca` de référence afin de garantir la parité sur les métriques PDR/SNR lorsque les paramètres ci-dessus sont activés.【F:tests/test_flora_sca.py†L18-L39】
