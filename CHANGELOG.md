# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

_Aucun changement notable pour le moment._

## [1.0.1] - 2025-09-30
### Added
- Ajout du profil d'exécution « fast » pour les scripts MNE3SD afin de réduire automatiquement la taille des balayages et d'appliquer les préréglages associés (nœuds, réplicas, intervalles RX classe C). 
### Changed
- Les générateurs de trafic (simulateur CLI, nœuds complets et chargeur de configuration) s'appuient désormais sur `traffic.exponential.sample_interval` pour échantillonner strictement une loi exponentielle alignée sur OMNeT++ et ne repousser les transmissions que lorsque l'intervalle reste inférieur à la durée d'émission précédente. 
- Le profil `adr_standard_1` active un canal dégradé plus sévère (bruit, fading et capture avancée) par défaut afin de refléter les validations radio.
- Le gain de traitement LoRa n'est plus ajouté implicitement au calcul du SNR ; il devient un comportement opt-in via le paramètre `processing_gain`.
### Fixed
- Les fenêtres périodiques des nœuds de classe C cessent de sonder lorsqu'aucun downlink n'est en attente, ce qui évite les boucles infinies observées lors des campagnes MNE3SD tout en garantissant la livraison finale.
- Les balayages de densité MNE3SD peuvent forcer l'intervalle de sondage des nœuds de classe C, avec une valeur réduite appliquée automatiquement pour le profil « fast » afin d'accélérer les itérations.

## [Draft]
_Brouillon conservé pour une refonte majeure envisagée mais jamais publiée._

### Added
- Complete rewrite of the LoRa network simulator in Python.
- Command-line interface and interactive dashboard.
- FastAPI REST and WebSocket API.
- Advanced propagation models with fading, mobility and obstacle support.
- LoRaWAN implementation with ADR logic, classes B and C, and AES-128 security.
- CSV export and detailed metrics.
- Unit tests with pytest and analysis scripts.

## [1.0.0] - 2025-08-26
### Added
- Initial public release of LoRaFlexSim, offering a flexible LoRa network simulator.
- Command-line interface with example scenarios.
- Documentation and basic unit tests.
