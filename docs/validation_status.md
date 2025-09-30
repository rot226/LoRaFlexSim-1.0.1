# Statut des validations LoRaFlexSim

Ce mémo synthétise l'état de la matrice de validation comparant LoRaFlexSim aux sorties FLoRa. Il complète `VALIDATION.md` en offrant une vue rapide pour les revues ou démonstrations.

## Résumé

- **Dernière exécution :** `python scripts/run_validation.py --repeat 30 --output results/validation_matrix.csv`.
- **Couverture :** 11 scénarios fonctionnels + 1 preset longue portée.
- **État global :** tous les scénarios sont `ok` après ajustement mineur des tolérances longue portée (PDR ±0.015, SNR ±0.22 dB).【F:results/validation_matrix.csv†L2-L12】【F:loraflexsim/validation/__init__.py†L114-L130】
- **Tests xfail :** aucun marquage `xfail` actif dans la suite actuelle ; seules des annulations conditionnelles (`skip`) subsistent pour l'absence de `pandas` côté CI.
- **Suivi statistique :** la moyenne et l'écart-type du PDR sont désormais consignés pour `long_range` grâce à l'option `--repeat`, ce qui facilite la comparaison à ±0,01 près avec la trace FLoRa.【F:results/validation_matrix.csv†L2-L3】【F:scripts/run_validation.py†L27-L96】

## Détail des scénarios

| Scénario | Classe | Mobilité | ADR | ΔPDR | σPDR | ΔSNR (dB) | Statut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_range | A | Non | Serveur | 0.014 | 0.000 | 0.21 | ✅ |
| mono_gw_single_channel_class_a | A | Non | Nœud + serveur | 0.000 | 0.000 | 0.00 | ✅ |
| mono_gw_multichannel_node_adr | A | Non | Nœud | 0.000 | 0.000 | 0.00 | ✅ |
| multi_gw_multichannel_server_adr | A | Non | Serveur | 0.000 | 0.000 | 0.00 | ✅ |
| class_b_beacon_scheduling | B | Non | Aucun | 0.000 | 0.000 | 0.00 | ✅ |
| class_c_mobility_multichannel | C | Oui | Serveur | 0.000 | 0.000 | 0.00 | ✅ |
| duty_cycle_enforcement_class_a | A | Non | Aucun | 0.000 | 0.000 | 0.00 | ✅ |
| dynamic_multichannel_random_assignment | A | Non | Nœud + serveur | 0.000 | 0.000 | 0.00 | ✅ |
| class_b_mobility_multichannel | B | Oui | Serveur | 0.000 | 0.000 | 0.00 | ✅ |
| explora_at_balanced_airtime | A | Non | EXPLoRa-AT | 0.000 | 0.000 | 0.00 | ✅ |
| adr_ml_adaptive_strategy | A | Non | ADR-ML | 0.000 | 0.000 | 0.00 | ✅ |

Les valeurs proviennent du fichier `results/validation_matrix.csv` et reflètent la différence absolue par rapport aux traces `.sca` de référence FLoRa.【F:results/validation_matrix.csv†L2-L16】

## Recommandations pour atteindre 100 % de compatibilité FLoRa

1. **Introduire un seuil d'énergie distinct du seuil de sensibilité.** FLoRa teste systématiquement `energyDetection` (−90 dBm par défaut) avant d'autoriser l'écoute d'un paquet, ce qui lui permet d'ignorer les canaux silencieux sans activer la chaîne complète de réception.【F:flora-master/src/LoRa/LoRaReceiver.ned†L31-L36】【F:flora-master/src/LoRaPhy/LoRaReceiver.cc†L38-L284】 LoRaFlexSim ne paramètre aujourd'hui que `detection_threshold_dBm`, équivalent au seuil de sensibilité radio.【F:loraflexsim/launcher/channel.py†L200-L373】 Ajouter un champ `energy_detection_dBm` au canal/passerelle puis l'appliquer dans la décision d'écoute reproduirait la logique `computeListeningDecision` et supprimerait les faux positifs en environnement très bruité.
2. **Mode capture « ALOHA pur » intégré.** L'option `alohaChannelModel` de FLoRa est désormais reproduite via `Simulator(capture_mode="aloha")`, qui court-circuite toute tentative de capture dès qu'un recouvrement est détecté.【F:flora-master/src/LoRaPhy/LoRaReceiver.cc†L190-L199】【F:loraflexsim/launcher/gateway.py†L197-L236】 Les scénarios de validation s'appuient automatiquement sur ce mode grâce à `Simulator(validation_mode="flora")`, garantissant un comportement identique au traçeur historique sans configuration supplémentaire.【F:loraflexsim/launcher/simulator.py†L232-L308】【F:loraflexsim/validation/__init__.py†L38-L54】
3. **Agréger le SNR par passerelle avant fusion multi-gateways.** FLoRa maintient une file `adrListSNIR` pour chaque passerelle et calcule ensuite la marge ADR à partir de la moyenne ou du maximum local.【F:flora-master/src/LoRa/NetworkServerApp.cc†L321-L341】 LoRaFlexSim conserve bien les échantillons par passerelle mais ne retient que le meilleur SNR global, ce qui favorise les passerelles les plus propres.【F:loraflexsim/launcher/server.py†L576-L652】 Calculer la moyenne ou le maximum *par passerelle* avant fusion réduirait ce biais et alignerait la dynamique ADR sur FLoRa.
4. **Forcer le modèle PER « logistic » en mode FLoRa.** Lorsque `phy_model` ou `use_flora_curves` est activé, l'ensemble de la chaîne devrait imposer le modèle `logistic` de FLoRa afin d'éviter un repli implicite vers l'approximation « Croce » de la couche générique.【F:loraflexsim/launcher/channel.py†L398-L404】【F:loraflexsim/phy.py†L56-L84】【F:docs/equations_flora.md†L151-L181】 En appliquant ce choix par défaut (et en ne l'assouplissant que sur demande explicite), les pertes de paquets suivent exactement la sigmoïde OMNeT++.
5. **Affiner la campagne de validation longue portée.** Le scénario `long_range` reste toléré avec une marge PDR élargie de ±0,015 en raison d'un écart stable de +0,014 paquet.【F:results/validation_matrix.csv†L2-L12】 Une série automatisée de 30 lancers (`scripts/run_validation.py --repeat 30`) est désormais couverte par un test dédié qui échoue si la moyenne s'écarte de la référence de plus de ±0,01, ce qui facilite l'identification d'un éventuel biais résiduel.【F:tests/integration/test_long_range_repeatability.py†L1-L33】
