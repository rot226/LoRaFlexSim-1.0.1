# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed
- Significantly increased channel degradation in `adr_standard_1` for simulator validation.
- Send interval distribution now follows a strict exponential law and timestamps are only postponed when a transmission is still ongoing.
- Removed the implicit LoRa processing gain from SNR calculations; the legacy behaviour is available via `processing_gain=True`.

### Migration
- FLoRa-aligned scenarios now inject the historical inter-SF capture matrix and lock the 6-symbol capture window automatically. Remove any custom `orthogonal_sf`/`non_orth_matrix` overrides and rely on the default behaviour when migrating configurations. The new `--long-range-demo very_long_range` preset replaces manual tuning for 15 km studies.

## [5.0] - 2025-07-24
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
