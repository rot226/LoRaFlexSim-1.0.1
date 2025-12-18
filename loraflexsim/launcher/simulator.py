"""Event-driven simulator for LoRaFlexSim."""

import configparser
import heapq
import logging
import math
import numbers
import random
import numpy as np
from collections import deque

from traffic.exponential import sample_interval
from traffic.rng_manager import RngManager
from traffic.numpy_compat import create_generator
from pathlib import Path
from dataclasses import dataclass


# Earlier versions used integer nanoseconds for event timestamps to mimic
# OMNeT++'s scheduler tick.  By default we rely on continuous double precision
# times expressed in seconds to avoid any quantisation effects that could
# slightly bias statistics.  The optional ``tick_ns`` parameter allows to
# reintroduce integer nanosecond scheduling when required.
from enum import IntEnum

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional
    pd = None

from .node import Node
from .gateway import Gateway, FLORA_NON_ORTH_DELTA
from .channel import Channel
from .multichannel import MultiChannel
from .server import NetworkServer
from .duty_cycle import DutyCycleManager
from .smooth_mobility import SmoothMobility
from .id_provider import next_node_id, next_gateway_id, reset as reset_ids


class EventType(IntEnum):
    """Types d'événements traités par le simulateur."""

    TX_END = 0
    TX_START = 1
    MOBILITY = 2
    RX_WINDOW = 3
    BEACON = 4
    PING_SLOT = 5
    SERVER_RX = 6
    SERVER_PROCESS = 7
    QOS_RECONFIG = 8


@dataclass(order=True, slots=True)
class Event:
    time: int | float
    type: int
    id: int
    node_id: int


logger = logging.getLogger(__name__)
diag_logger = logging.getLogger("diagnostics")


def _load_channel_overrides(path: str | Path | None) -> dict[str, float]:
    """Return channel override values defined in ``path`` if present."""

    overrides: dict[str, float] = {}
    if path is None:
        return overrides
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return overrides

    cp = configparser.ConfigParser()
    cp.read(cfg_path)
    if not cp.has_section("channel"):
        return overrides

    keys = (
        "snir_fading_std",
        "noise_floor_std",
        "interference_dB",
        "sensitivity_margin_dB",
        "capture_threshold_dB",
        "marginal_snir_margin_db",
        "marginal_snir_drop_prob",
    )
    for key in keys:
        try:
            value = cp.getfloat("channel", key)
        except Exception:
            continue
        overrides[key] = value
    return overrides


def _validate_positive_real(name: str, value: object) -> float:
    """Return ``value`` as a positive real number or raise a clear error."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive, finite number")
    return float(value)


class _PowerTimeline:
    """Track start/end times of interfering transmissions in linear power."""

    def __init__(self) -> None:
        self._entries: dict[int, tuple[float, float, float, int | None]] = {}

    def add(
        self,
        event_id: int,
        start: float,
        end: float,
        power_mw: float,
        sf: int | None = None,
    ) -> None:
        if end <= start:
            return
        self._entries[event_id] = (start, end, power_mw, sf)

    def remove(self, event_id: int) -> None:
        self._entries.pop(event_id, None)

    def prune(self, time: float) -> None:
        finished = [eid for eid, (_, end, _, _) in self._entries.items() if end <= time]
        for eid in finished:
            self._entries.pop(eid, None)

    def is_empty(self) -> bool:
        return not self._entries

    def average_power(
        self,
        start: float,
        end: float,
        base_power: float = 0.0,
        *,
        target_sf: int | None = None,
        alpha_isf: float = 0.0,
        exclude_event_id: int | None = None,
    ) -> float:
        if end <= start:
            return base_power
        events: dict[float, float] = {}
        if base_power != 0.0:
            events[start] = events.get(start, 0.0) + base_power
            events[end] = events.get(end, 0.0) - base_power
        for eid, (s, e, p, sf) in self._entries.items():
            if exclude_event_id is not None and eid == exclude_event_id:
                continue
            overlap_start = max(start, s)
            overlap_end = min(end, e)
            if overlap_end <= overlap_start:
                continue
            factor = 1.0 if target_sf is None or sf == target_sf else alpha_isf
            if factor == 0.0:
                continue
            weighted = p * factor
            events[overlap_start] = events.get(overlap_start, 0.0) + weighted
            events[overlap_end] = events.get(overlap_end, 0.0) - weighted
        if not events:
            return base_power
        energy = 0.0
        level = 0.0
        last_time: float | None = None
        for t in sorted(events):
            if last_time is not None and t > last_time:
                energy += level * (t - last_time)
            level += events[t]
            last_time = t
        duration = end - start
        if duration <= 0.0:
            return base_power
        return energy / duration

    def total_power(
        self,
        start: float,
        end: float,
        base_power: float = 0.0,
        *,
        target_sf: int | None = None,
        alpha_isf: float = 0.0,
        exclude_event_id: int | None = None,
    ) -> float:
        """Retourne la puissance moyenne totale sur l'intervalle demandé."""

        return self.average_power(
            start,
            end,
            base_power,
            target_sf=target_sf,
            alpha_isf=alpha_isf,
            exclude_event_id=exclude_event_id,
        )

    def power_changes(
        self,
        start: float,
        end: float,
        base_power: float = 0.0,
        *,
        target_sf: int | None = None,
        alpha_isf: float = 0.0,
    ) -> dict[float, float]:
        events: dict[float, float] = {}
        if base_power != 0.0:
            events[start] = events.get(start, 0.0) + base_power
            events[end] = events.get(end, 0.0) - base_power
        for s, e, p, sf in self._entries.values():
            overlap_start = max(start, s)
            overlap_end = min(end, e)
            if overlap_end <= overlap_start:
                continue
            factor = 1.0 if target_sf is None or sf == target_sf else alpha_isf
            if factor == 0.0:
                continue
            weighted = p * factor
            events[overlap_start] = events.get(overlap_start, 0.0) + weighted
            events[overlap_end] = events.get(overlap_end, 0.0) - weighted
        return dict(sorted(events.items()))


class InterferenceTracker:
    """Suivi des transmissions actives par passerelle, fréquence et SF."""

    def __init__(self, rng=None) -> None:
        self.active: dict[tuple[int, float], _PowerTimeline] = {}
        self.rng = rng or create_generator()

    def add(
        self,
        gateway_id: int,
        frequency: float,
        sf: int | None,
        rssi_dBm: float,
        end_time: float,
        event_id: int,
        *,
        start_time: float,
    ) -> None:
        key = (gateway_id, frequency)
        timeline = self.active.setdefault(key, _PowerTimeline())
        timeline.add(event_id, start_time, end_time, 10 ** (rssi_dBm / 10.0), sf)

    def average_power(
        self,
        gateway_id: int,
        frequency: float,
        sf: int | None,
        current_time: float,
        end_time: float,
        *,
        base_noise_mW: float = 0.0,
        alpha_isf: float = 0.0,
    ) -> float:
        key = (gateway_id, frequency)
        timeline = self.active.get(key)
        if timeline is None:
            return base_noise_mW
        timeline.prune(current_time)
        if timeline.is_empty():
            self.active.pop(key, None)
            return base_noise_mW
        return timeline.average_power(
            current_time,
            end_time,
            base_noise_mW,
            target_sf=sf,
            alpha_isf=alpha_isf,
        )

    def total_interference(
        self,
        gateway_id: int,
        frequency: float,
        sf: int | None,
        start_time: float,
        end_time: float,
        *,
        base_noise_mW: float = 0.0,
        alpha_isf: float = 0.0,
        fading_std: float = 0.0,
        exclude_event_id: int | None = None,
    ) -> float:
        """Calcule l'interférence moyenne (hors bruit) sur l'intervalle donné."""

        key = (gateway_id, frequency)
        timeline = self.active.get(key)
        if timeline is None:
            return 0.0

        timeline.prune(start_time)
        if timeline.is_empty():
            self.active.pop(key, None)
            return 0.0

        total_power = timeline.total_power(
            start_time,
            end_time,
            base_power=base_noise_mW,
            target_sf=sf,
            alpha_isf=alpha_isf,
            exclude_event_id=exclude_event_id,
        )
        interference = max(total_power - base_noise_mW, 0.0)
        if fading_std > 0.0:
            fading_db = float(self.rng.normal(0.0, fading_std))
            interference *= 10 ** (fading_db / 10.0)
        return max(interference, 0.0)

    def power_changes(
        self,
        gateway_id: int,
        frequency: float,
        sf: int | None,
        start: float,
        end: float,
        *,
        base_noise_mW: float = 0.0,
        alpha_isf: float = 0.0,
    ) -> dict[float, float]:
        key = (gateway_id, frequency)
        timeline = self.active.get(key)
        if timeline is None:
            if base_noise_mW == 0.0:
                return {}
            return {start: base_noise_mW, end: -base_noise_mW}
        return timeline.power_changes(
            start,
            end,
            base_noise_mW,
            target_sf=sf,
            alpha_isf=alpha_isf,
        )

    def remove(self, event_id: int) -> None:
        for key in list(self.active.keys()):
            timeline = self.active[key]
            timeline.remove(event_id)
            if timeline.is_empty():
                self.active.pop(key, None)


class Simulator:
    """Gère la simulation du réseau LoRa (nœuds, passerelles, événements)."""

    # Constantes ADR LoRaWAN standard
    REQUIRED_SNR = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
    MARGIN_DB = 15.0  # marge d'installation en dB (typiquement 15 dB)

    def channel_index(self, channel) -> int:
        """Return the simulator-wide index associated with ``channel``."""

        if channel is None:
            return self.control_channel_index
        if channel is getattr(self, "control_channel", None):
            return self.control_channel_index
        reverse = getattr(self, "_channel_reverse", None)
        lookup = getattr(self, "_channel_lookup", None)
        if reverse is None or lookup is None:
            if lookup is None:
                lookup = {}
                self._channel_lookup = lookup
            if reverse is None:
                reverse = {}
                self._channel_reverse = reverse
        idx = reverse.get(id(channel))
        if idx is not None:
            return idx
        idx = max(lookup.keys(), default=-1) + 1
        lookup[idx] = channel
        reverse[id(channel)] = idx
        return idx

    def get_channel_by_index(self, index: int):
        """Return the channel object referenced by ``index`` if known."""

        if index == getattr(self, "control_channel_index", None):
            return getattr(self, "control_channel", None)
        lookup = getattr(self, "_channel_lookup", None)
        if lookup is None:
            return None
        return lookup.get(index)

    def __init__(
        self,
        num_nodes: int = 10,
        num_gateways: int = 1,
        area_size: float = 1000.0,
        transmission_mode: str = "Random",
        packet_interval: float = 60.0,
        first_packet_interval: float | None = None,
        warm_up_intervals: int = 0,
        log_mean_after: int | None = None,
        interval_variation: float = 0.0,
        packets_to_send: int = 0,
        adr_node: bool = False,
        adr_server: bool = False,
        adr_method: str = "max",
        duty_cycle: float | None = 0.01,
        mobility: bool = True,
        channels=None,
        channel_distribution: str = "round-robin",
        mobility_speed: tuple[float, float] = (2.0, 10.0),
        fixed_sf: int | None = None,
        fixed_tx_power: float | None = None,
        battery_capacity_j: float | None = None,
        payload_size_bytes: int = 20,
        node_class: str = "A",
        detection_threshold_dBm: float = -float("inf"),
        energy_detection_dBm: float = -float("inf"),
        min_interference_time: float = 0.0,
        flora_mode: bool = False,
        flora_timing: bool = False,
        config_file: str | None = None,
        seed: int | None = None,
        class_c_rx_interval: float = 1.0,
        phy_model: str = "",
        flora_loss_model: str = "lognorm",
        terrain_map: str | list[list[float]] | None = None,
        path_map: str | list[list[float]] | None = None,
        dynamic_obstacles: str | list[dict] | None = None,
        mobility_model=None,
        beacon_drift: float = 0.0,
        *,
        clock_accuracy: float = 0.0,
        beacon_loss_prob: float = 0.0,
        ping_slot_interval: float = 1.0,
        ping_slot_offset: float = 2.0,
        channel_config: str | Path | None = None,
        snir_fading_std: float | None = None,
        noise_floor_std: float | None = None,
        interference_dB: float | None = None,
        sensitivity_margin_dB: float | None = None,
        capture_threshold_dB: float | None = None,
        marginal_snir_margin_db: float | None = None,
        marginal_snir_drop_prob: float | None = None,
        debug_rx: bool = False,
        dump_intervals: bool = False,
        pure_poisson_mode: bool = False,
        lock_step_poisson: bool = False,
        phase_noise_std_dB: float = 0.0,
        clock_jitter_std_s: float = 0.0,
        pa_ramp_up_s: float = 0.0,
        pa_ramp_down_s: float = 0.0,
        capture_mode: str | None = None,
        validation_mode: str | None = None,
        tick_ns: int | None = None,
    ):
        """
        Initialise la simulation LoRaFlexSim avec les entités et paramètres donnés.
        :param num_nodes: Nombre de nœuds à simuler.
        :param num_gateways: Nombre de passerelles à simuler.
        :param area_size: Taille de l'aire carrée (mètres) dans laquelle sont déployés nœuds et passerelles.
        :param transmission_mode: 'Random' pour transmissions aléatoires (Poisson) ou 'Periodic' pour périodiques.
        :param packet_interval: Intervalle moyen entre transmissions (si Random, moyenne en s; si Periodic, période fixe en s).
        :param first_packet_interval: Intervalle moyen appliqué uniquement au tout premier envoi (``None`` pour utiliser ``packet_interval``).
        :param warm_up_intervals: Nombre d'intervalles à ignorer dans les métriques (warm-up).
        :param log_mean_after: Nombre d'intervalles comptabilisés après warm-up
            avant journalisation de la moyenne empirique (``None`` pour désactiver).
        :param interval_variation: Jitter relatif appliqué à chaque intervalle
            exponentiel. La valeur par défaut ``0`` reproduit fidèlement le
            modèle aléatoire de FLoRa (aucune dispersion supplémentaire).
        :param packets_to_send: Nombre de paquets à émettre **par nœud** avant
            d'arrêter la simulation (0 = infini).
        :param adr_node: Activation de l'ADR côté nœud.
        :param adr_server: Activation de l'ADR côté serveur.
        :param adr_method: Méthode d'agrégation du SNR pour l'ADR
            (``"max"`` ou ``"avg"``).
        :param duty_cycle: Facteur de duty cycle (ex: 0.01 pour 1 %). Par
            défaut à 0.01. Si None, le duty cycle est désactivé.
        :param mobility: Active la mobilité aléatoire des nœuds lorsqu'il est
            à True.
        :param mobility_speed: Couple (min, max) définissant la plage de
            vitesses de déplacement des nœuds en m/s lorsqu'ils sont mobiles.
        :param channels: ``MultiChannel`` ou liste de fréquences/``Channel`` pour
            gérer plusieurs canaux.
        :param channel_distribution: Méthode d'affectation des canaux aux nœuds
            ("round-robin" ou "random").
        :param fixed_sf: Si défini, tous les nœuds démarrent avec ce SF.
        :param fixed_tx_power: Si défini, puissance d'émission initiale commune (dBm).
        :param battery_capacity_j: Capacité de la batterie attribuée à chaque nœud (J). ``None`` pour illimité.
        :param payload_size_bytes: Taille du payload utilisé pour calculer l'airtime (octets).
        :param node_class: Classe LoRaWAN commune à tous les nœuds ('A', 'B' ou 'C').
        :param detection_threshold_dBm: RSSI minimal requis pour qu'une
            réception soit prise en compte.
        :param energy_detection_dBm: Seuil de détection d'énergie appliqué
            avant la vérification de sensibilité. Lorsque ``flora_mode`` est
            activé, la valeur par défaut correspond à −90 dBm.
        :param min_interference_time: Chevauchement temporel toléré entre
            transmissions avant de les considérer en collision (s).
        :param flora_mode: Active automatiquement les réglages du mode FLoRa
            complet (seuil -110 dBm et 5 s d'interférence minimale).
        :param flora_timing: Utilise les temporisations du projet FLoRa
            (délai réseau de 10 ms et traitement serveur de 1,2 s).
        :param config_file: Fichier INI listant les positions des nœuds et
            passerelles à charger. Lorsque défini, ``num_nodes`` et
            ``num_gateways`` sont ignorés.
        :param seed: Graine aléatoire pour reproduire le placement des nœuds et
            l'ordre statistique des intervalles. ``None`` pour un tirage
            différent à chaque exécution.
        :param class_c_rx_interval: Période entre deux vérifications de
            downlink pour les nœuds de classe C (s).
        :param phy_model: "omnet" ou "flora" pour activer un modèle physique
            inspiré de FLoRa.
        :param flora_loss_model: Variante d'atténuation FLoRa ("lognorm",
            "oulu" ou "hata").
        :param terrain_map: Carte de terrain utilisée pour la mobilité
            aléatoire (chemin JSON/texte ou matrice). Les valeurs négatives
            indiquent les obstacles et ralentissements éventuels.
        :param path_map: Carte de type obstacle où un chemin doit être trouvé
            entre deux positions. Lorsque défini, la mobilité suit les
            plus courts chemins évitant les obstacles.
        :param dynamic_obstacles: Fichier JSON ou liste décrivant des obstacles
            mouvants pour ``PathMobility``.
        :param mobility_model: Instance personnalisée de modèle de mobilité
            (prioritaire sur ``terrain_map`` et ``path_map``).
        :param beacon_drift: Dérive relative appliquée aux beacons (ppm).
        :param clock_accuracy: Écart-type de la dérive d'horloge des nœuds
            (ppm). Chaque nœud se voit attribuer un décalage aléatoire selon
            cette précision.
        :param beacon_loss_prob: Probabilité pour un nœud de manquer un beacon.
        :param ping_slot_interval: Intervalle de base entre deux ping slots
            (s).
        :param ping_slot_offset: Décalage initial entre le beacon et le premier
            ping slot (s).
        :param channel_config: Fichier INI optionnel contenant une section
            ``[channel]`` pour surcharger les paramètres radio (fading SNIR,
            bruit, capture…). Aucun effet si absent.
        :param snir_fading_std: Écart-type du fading appliqué au signal et aux
            interférences lors du calcul SNIR (dB). ``None`` conserve la
            valeur par défaut ou celle du fichier de configuration.
        :param noise_floor_std: Dispersion logarithmique appliquée au bruit de
            fond (dB). ``None`` laisse la valeur actuelle.
        :param interference_dB: Bruit de fond moyen ajouté au plancher thermique
            (dB). ``None`` pour conserver la valeur existante.
        :param sensitivity_margin_dB: Marge ajoutée aux seuils de sensibilité
            par SF pour éviter des réceptions trop optimistes.
        :param capture_threshold_dB: Seuil de capture utilisé lorsque plusieurs
            paquets se chevauchent (dB).
        :param marginal_snir_margin_db: Marge en-dessous de laquelle un paquet
            capturé peut encore être perdu aléatoirement (dB).
        :param marginal_snir_drop_prob: Probabilité maximale associée à la
            perte marginale décrite ci-dessus.
        :param debug_rx: Active la journalisation détaillée des paquets reçus ou rejetés.
        :param dump_intervals: Exporte la série complète des intervalles dans un fichier Parquet.
        :param lock_step_poisson: Prégénère la séquence Poisson une seule fois et la réutilise.
        :param phase_noise_std_dB: Bruit de phase appliqué au SNR (écart-type en dB).
        :param clock_jitter_std_s: Gigue d'horloge ajoutée à chaque calcul (s).
        :param pa_ramp_up_s: Temps de montée du PA (s).
        :param pa_ramp_down_s: Temps de descente du PA (s).
        :param capture_mode: Force un mode de capture spécifique pour toutes
            les passerelles (``"basic"``, ``"advanced"``, ``"flora"``,
            ``"omnet"`` ou ``"aloha"``). ``None`` conserve la sélection
            automatique basée sur le canal.
        :param validation_mode: Active des réglages additionnels dédiés aux
            campagnes de validation (``"flora"`` déclenche le mode capture
            ``"aloha"`` par défaut).
        :param tick_ns: Quand défini, quantifie chaque instant à des entiers de
            ``tick_ns`` nanosecondes pour la file d'événements.
        """
        # Paramètres de simulation
        if flora_mode and packet_interval == 60.0 and first_packet_interval is None:
            packet_interval = first_packet_interval = 1000.0

        packet_interval = _validate_positive_real("packet_interval", packet_interval)
        if first_packet_interval is None:
            first_packet_interval = packet_interval
        else:
            first_packet_interval = _validate_positive_real(
                "first_packet_interval",
                first_packet_interval,
            )

        self.num_nodes = num_nodes
        self.num_gateways = num_gateways
        self.area_size = area_size
        self.transmission_mode = transmission_mode
        self.packet_interval = packet_interval
        self.first_packet_interval = first_packet_interval
        # Minimal delay before the first transmission (5 s in FLoRa mode)
        self.first_packet_min_delay = 0.0
        self.warm_up_intervals = warm_up_intervals
        self.log_mean_after = log_mean_after
        if interval_variation < 0 or interval_variation > 3:
            raise ValueError("interval_variation must be between 0 and 3")
        self.interval_variation = interval_variation
        self.packets_to_send = packets_to_send
        self.adr_node = adr_node
        self.adr_server = adr_server
        self.adr_method = adr_method
        self.fixed_sf = fixed_sf
        self.fixed_tx_power = fixed_tx_power
        self.battery_capacity_j = battery_capacity_j
        self.payload_size_bytes = payload_size_bytes
        self.node_class = node_class
        provided_channels: list[Channel] = []
        if isinstance(channels, MultiChannel):
            provided_channels = [
                ch for ch in channels.channels if isinstance(ch, Channel)
            ]
        elif isinstance(channels, (list, tuple, set)):
            provided_channels = [ch for ch in channels if isinstance(ch, Channel)]
        elif isinstance(channels, Channel):
            provided_channels = [channels]

        if flora_mode:
            if detection_threshold_dBm == -float("inf"):
                if provided_channels:
                    thresholds = [
                        ch.detection_threshold_dBm
                        for ch in provided_channels
                        if ch.detection_threshold_dBm != -float("inf")
                    ]
                    detection_threshold_dBm = (
                        min(thresholds) if thresholds else -110.0
                    )
                else:
                    detection_threshold_dBm = Channel.flora_energy_threshold(125000.0)
            if energy_detection_dBm == -float("inf"):
                if provided_channels:
                    floors = [
                        ch.energy_detection_dBm
                        if ch.energy_detection_dBm != -float("inf")
                        else Channel.flora_energy_threshold(ch.bandwidth)
                        for ch in provided_channels
                    ]
                    energy_detection_dBm = min(floors) if floors else Channel.FLORA_ENERGY_DETECTION_DBM
                else:
                    energy_detection_dBm = Channel.flora_energy_threshold(125000.0)
            if min_interference_time == 0.0:
                min_interference_time = 5.0
            if self.first_packet_min_delay == 0.0:
                self.first_packet_min_delay = 5.0
        if pure_poisson_mode:
            duty_cycle = None
            detection_threshold_dBm = -float("inf")
            min_interference_time = float("inf")
            energy_detection_dBm = -float("inf")
        self.detection_threshold_dBm = detection_threshold_dBm
        self.energy_detection_dBm = energy_detection_dBm
        self.min_interference_time = min_interference_time
        self.pure_poisson_mode = pure_poisson_mode
        self.lock_step_poisson = lock_step_poisson
        self.flora_mode = flora_mode
        self.flora_timing = flora_timing
        self.config_file = config_file
        self.phy_model = phy_model
        self.flora_loss_model = flora_loss_model
        self.phase_noise_std_dB = phase_noise_std_dB
        self.clock_jitter_std_s = clock_jitter_std_s
        self.validation_mode = validation_mode.lower() if isinstance(validation_mode, str) else validation_mode
        if isinstance(capture_mode, str):
            capture_mode = capture_mode.lower()
        if capture_mode is None and self.validation_mode == "flora":
            capture_mode = "aloha"
        self.capture_mode = capture_mode
        self.pa_ramp_up_s = pa_ramp_up_s
        self.pa_ramp_down_s = pa_ramp_down_s
        channel_overrides = _load_channel_overrides(channel_config)
        manual_overrides = {
            "snir_fading_std": snir_fading_std,
            "noise_floor_std": noise_floor_std,
            "interference_dB": interference_dB,
            "sensitivity_margin_dB": sensitivity_margin_dB,
            "capture_threshold_dB": capture_threshold_dB,
            "marginal_snir_margin_db": marginal_snir_margin_db,
            "marginal_snir_drop_prob": marginal_snir_drop_prob,
        }
        for key, value in manual_overrides.items():
            if value is not None:
                channel_overrides[key] = float(value)

        def _apply_overrides(channel: Channel) -> None:
            if not channel_overrides:
                return
            if "capture_threshold_dB" in channel_overrides:
                channel.capture_threshold_dB = channel_overrides["capture_threshold_dB"]
            if "snir_fading_std" in channel_overrides:
                channel.snir_fading_std = channel_overrides["snir_fading_std"]
            if "noise_floor_std" in channel_overrides:
                channel.noise_floor_std = channel_overrides["noise_floor_std"]
            if "interference_dB" in channel_overrides:
                channel.interference_dB = channel_overrides["interference_dB"]
            if "marginal_snir_margin_db" in channel_overrides:
                channel.marginal_snir_margin_db = channel_overrides["marginal_snir_margin_db"]
            if "marginal_snir_drop_prob" in channel_overrides:
                channel.marginal_snir_drop_prob = channel_overrides["marginal_snir_drop_prob"]
            if "sensitivity_margin_dB" in channel_overrides:
                channel.sensitivity_margin_dB = channel_overrides["sensitivity_margin_dB"]
                if hasattr(channel, "_update_sensitivity"):
                    channel._update_sensitivity()

        channel_kwargs = {
            key: value
            for key, value in channel_overrides.items()
            if key
            in {
                "snir_fading_std",
                "noise_floor_std",
                "interference_dB",
                "sensitivity_margin_dB",
                "capture_threshold_dB",
                "marginal_snir_margin_db",
                "marginal_snir_drop_prob",
            }
        }
        # Activation ou non de la mobilité des nœuds
        self.mobility_enabled = mobility
        if mobility_model is not None:
            self.mobility_model = mobility_model
        elif path_map is not None:
            if isinstance(path_map, (str, Path)):
                from .map_loader import load_map

                path_map = load_map(path_map)
            from .path_mobility import PathMobility

            self.mobility_model = PathMobility(
                area_size,
                path_map,
                min_speed=mobility_speed[0],
                max_speed=mobility_speed[1],
                dynamic_obstacles=dynamic_obstacles,
            )
        elif terrain_map is not None:
            if isinstance(terrain_map, (str, Path)):
                from .map_loader import load_map

                terrain_map = load_map(terrain_map)
            from .random_waypoint import RandomWaypoint

            self.mobility_model = RandomWaypoint(
                area_size,
                min_speed=mobility_speed[0],
                max_speed=mobility_speed[1],
                terrain=terrain_map,
            )
        else:
            self.mobility_model = SmoothMobility(
                area_size, mobility_speed[0], mobility_speed[1]
            )

        # Class B/C settings
        self.beacon_interval = 128.0
        self.ping_slot_interval = ping_slot_interval
        self.ping_slot_offset = ping_slot_offset
        self.class_c_rx_interval = class_c_rx_interval
        self.beacon_drift = beacon_drift
        self.clock_accuracy = clock_accuracy
        self.beacon_loss_prob = beacon_loss_prob
        self.debug_rx = debug_rx
        self.dump_intervals = dump_intervals

        # Gestion du duty cycle (activé par défaut à 1 %)
        self.duty_cycle_manager = DutyCycleManager(duty_cycle) if duty_cycle else None

        # Activer les courbes FLoRa lorsque la simulation est configurée en mode FLoRa.
        force_flora_curves = flora_mode or phy_model.startswith("flora")
        flora_phy_cls = None

        def _apply_flora_curves(ch: Channel) -> None:
            nonlocal flora_phy_cls
            if not force_flora_curves:
                return
            ch.use_flora_curves = True
            if getattr(ch, "flora_phy", None) is None:
                if flora_phy_cls is None:
                    from .flora_phy import FloraPHY as _FloraPHY

                    flora_phy_cls = _FloraPHY
                ch.flora_phy = flora_phy_cls(ch, loss_model=ch.flora_loss_model)

        # Initialiser la gestion multi-canaux
        if isinstance(channels, MultiChannel):
            self.multichannel = channels
            if detection_threshold_dBm != -float("inf"):
                for ch in self.multichannel.channels:
                    ch.detection_threshold_dBm = detection_threshold_dBm
            if energy_detection_dBm != -float("inf"):
                for ch in self.multichannel.channels:
                    ch.energy_detection_dBm = energy_detection_dBm
            for ch in self.multichannel.channels:
                _apply_overrides(ch)
            if flora_mode:
                for ch in self.multichannel.channels:
                    ch.phy_model = "omnet_full"
            for ch in self.multichannel.channels:
                _apply_flora_curves(ch)
            if flora_mode or phy_model.startswith("flora"):
                for ch in self.multichannel.channels:
                    if getattr(ch, "environment", None) is None:
                        ch.environment = "flora"
                        (
                            ch.path_loss_exp,
                            ch.shadowing_std,
                            ch.path_loss_d0,
                            ch.reference_distance,
                        ) = Channel.ENV_PRESETS["flora"]
                for ch in self.multichannel.channels:
                    if flora_mode and ch.multipath_taps <= 1:
                        ch.multipath_taps = 3
            for ch in self.multichannel.channels:
                ch.phase_noise_std_dB = phase_noise_std_dB
                ch.clock_jitter_std_s = clock_jitter_std_s
                ch.pa_ramp_up_s = pa_ramp_up_s
                ch.pa_ramp_down_s = pa_ramp_down_s
                if hasattr(ch, "_phase_noise"):
                    ch._phase_noise.std = phase_noise_std_dB
                if getattr(ch, "omnet_phy", None):
                    ch.omnet_phy.clock_jitter_std_s = clock_jitter_std_s
                    ch.omnet_phy.pa_ramp_up_s = pa_ramp_up_s
                    ch.omnet_phy.pa_ramp_down_s = pa_ramp_down_s
                    ch.omnet_phy._phase_noise.std = phase_noise_std_dB
        else:
            if channels is None:
                env = "flora" if (flora_mode or phy_model.startswith("flora")) else None
                ch_phy_model = "omnet_full" if flora_mode else phy_model
                ch_list = [
                    Channel(
                        detection_threshold_dBm=detection_threshold_dBm,
                        energy_detection_dBm=energy_detection_dBm,
                        phy_model=ch_phy_model,
                        environment=env,
                        flora_loss_model=flora_loss_model,
                        use_flora_curves=force_flora_curves,
                        multipath_taps=3 if flora_mode else 1,
                        phase_noise_std_dB=phase_noise_std_dB,
                        clock_jitter_std_s=clock_jitter_std_s,
                        pa_ramp_up_s=pa_ramp_up_s,
                        pa_ramp_down_s=pa_ramp_down_s,
                        **channel_kwargs,
                    )
                ]
                for ch in ch_list:
                    _apply_flora_curves(ch)
                    _apply_overrides(ch)
            else:
                ch_list = []
                for ch in channels:
                    if isinstance(ch, Channel):
                        if detection_threshold_dBm != -float("inf"):
                            ch.detection_threshold_dBm = detection_threshold_dBm
                        if energy_detection_dBm != -float("inf"):
                            ch.energy_detection_dBm = energy_detection_dBm
                        if flora_mode:
                            ch.phy_model = "omnet_full"
                        _apply_flora_curves(ch)
                        if (flora_mode or phy_model.startswith("flora")) and getattr(
                            ch, "environment", None
                        ) is None:
                            ch.environment = "flora"
                            (
                                ch.path_loss_exp,
                                ch.shadowing_std,
                                ch.path_loss_d0,
                                ch.reference_distance,
                            ) = Channel.ENV_PRESETS["flora"]
                        if flora_mode and ch.multipath_taps <= 1:
                            ch.multipath_taps = 3
                        ch.phase_noise_std_dB = phase_noise_std_dB
                        ch.clock_jitter_std_s = clock_jitter_std_s
                        ch.pa_ramp_up_s = pa_ramp_up_s
                        ch.pa_ramp_down_s = pa_ramp_down_s
                        if hasattr(ch, "_phase_noise"):
                            ch._phase_noise.std = phase_noise_std_dB
                        if getattr(ch, "omnet_phy", None):
                            ch.omnet_phy.clock_jitter_std_s = clock_jitter_std_s
                            ch.omnet_phy.pa_ramp_up_s = pa_ramp_up_s
                            ch.omnet_phy.pa_ramp_down_s = pa_ramp_down_s
                            ch.omnet_phy._phase_noise.std = phase_noise_std_dB
                        _apply_overrides(ch)
                        ch_list.append(ch)
                    else:
                        channel_obj = Channel(
                            frequency_hz=float(ch),
                            detection_threshold_dBm=detection_threshold_dBm,
                            energy_detection_dBm=energy_detection_dBm,
                            phy_model="omnet_full" if flora_mode else phy_model,
                            environment=(
                                "flora"
                                if (flora_mode or phy_model.startswith("flora"))
                                else None
                            ),
                            flora_loss_model=flora_loss_model,
                            use_flora_curves=force_flora_curves,
                            multipath_taps=3 if flora_mode else 1,
                            phase_noise_std_dB=phase_noise_std_dB,
                            clock_jitter_std_s=clock_jitter_std_s,
                            pa_ramp_up_s=pa_ramp_up_s,
                            pa_ramp_down_s=pa_ramp_down_s,
                            **channel_kwargs,
                        )
                        _apply_flora_curves(channel_obj)
                        _apply_overrides(channel_obj)
                        ch_list.append(channel_obj)
            self.multichannel = MultiChannel(ch_list, method=channel_distribution)
            if force_flora_curves:
                for ch in self.multichannel.channels:
                    _apply_flora_curves(ch)

        non_orth_required = flora_mode or phy_model.startswith("flora")
        if not non_orth_required:
            non_orth_required = any(
                getattr(ch, "use_flora_curves", False) for ch in self.multichannel.channels
            )
        if non_orth_required:
            self.multichannel.force_non_orthogonal(FLORA_NON_ORTH_DELTA)

        self._channel_lookup: dict[int, Channel] = {}
        self._channel_reverse: dict[int, int] = {}
        for index, channel in enumerate(self.multichannel.channels):
            self._channel_lookup[index] = channel
            self._channel_reverse[id(channel)] = index
            if not hasattr(channel, "channel_index") or channel.channel_index == 0:
                channel.channel_index = index

        base_channel = self.multichannel.channels[0]
        self.control_channel_index = max(self._channel_lookup.keys(), default=-1) + 1
        self.control_channel = Channel(
            frequency_hz=getattr(base_channel, "frequency_hz", 868e6),
            bandwidth=getattr(base_channel, "bandwidth", 125e3),
            region=getattr(base_channel, "region", None),
            detection_threshold_dBm=getattr(
                base_channel, "detection_threshold_dBm", -float("inf")
            ),
            energy_detection_dBm=getattr(
                base_channel, "energy_detection_dBm", -float("inf")
            ),
            channel_index=self.control_channel_index,
            **channel_kwargs,
        )
        self.control_channel.orthogonal_sf = getattr(
            base_channel, "orthogonal_sf", True
        )
        self.control_channel.non_orth_delta = getattr(
            base_channel, "non_orth_delta", None
        )
        self.control_channel.sensitivity_margin_dB = getattr(
            base_channel, "sensitivity_margin_dB", 0.0
        )
        _apply_overrides(self.control_channel)
        self._channel_lookup[self.control_channel_index] = self.control_channel
        self._channel_reverse[id(self.control_channel)] = self.control_channel_index

        # Compatibilité : premier canal par défaut
        self.channel = self.multichannel.channels[0]
        # Réglages de temporisation inspirés de FLoRa
        if flora_timing:
            proc_delay = 1.2
            net_delay = 0.01
        else:
            proc_delay = 0.0
            net_delay = 0.0
        # Traiter immédiatement les paquets reçus pour éviter un retard artificiel
        self.network_server = NetworkServer(
            simulator=self,
            process_delay=proc_delay,
            network_delay=net_delay,
            adr_method=self.adr_method,
            energy_detection_dBm=energy_detection_dBm,
            capture_mode=self.capture_mode,
        )
        self.network_server.beacon_interval = self.beacon_interval
        self.network_server.beacon_drift = self.beacon_drift
        self.network_server.ping_slot_interval = self.ping_slot_interval
        self.network_server.ping_slot_offset = self.ping_slot_offset

        # Graine commune pour reproduire FLoRa (placement et tirages aléatoires)
        self.seed = seed
        stream_hash = 3091881735
        self.rng_manager = RngManager((self.seed or 0) ^ stream_hash)
        self.pos_rng = random.Random(self.seed)
        self.interval_rng = self.rng_manager.get_stream("traffic", 0)
        if self.seed is not None:
            random.seed(self.seed)

        for idx, channel in enumerate(self.multichannel.channels):
            if hasattr(channel, "set_rng"):
                channel.set_rng(self.rng_manager.get_stream("channel", idx))
        if hasattr(self.control_channel, "set_rng"):
            self.control_channel.set_rng(
                self.rng_manager.get_stream("channel_control", 0)
            )

        # Générer les passerelles
        self.gateways = []
        reset_ids()
        cfg_nodes = None
        cfg_gateways = None
        if config_file:
            from .config_loader import load_config

            cfg_nodes, cfg_gateways, mean_interval, first_mean = load_config(
                config_file
            )
            if cfg_gateways:
                self.num_gateways = len(cfg_gateways)
            if cfg_nodes:
                self.num_nodes = len(cfg_nodes)
            if mean_interval is not None:
                self.packet_interval = mean_interval
            if first_mean is not None:
                self.first_packet_interval = first_mean

        for idx in range(self.num_gateways):
            gw_id = next_gateway_id()
            if cfg_gateways and idx < len(cfg_gateways):
                gw_x = cfg_gateways[idx]["x"]
                gw_y = cfg_gateways[idx]["y"]
                gw_power = cfg_gateways[idx].get("tx_power")
            elif self.num_gateways == 1:
                gw_x = area_size / 2.0
                gw_y = area_size / 2.0
                gw_power = None
            else:
                gw_x = self.pos_rng.random() * area_size
                gw_y = self.pos_rng.random() * area_size
                gw_power = None
            self.gateways.append(
                Gateway(
                    gw_id,
                    gw_x,
                    gw_y,
                    downlink_power_dBm=gw_power,
                    energy_detection_dBm=energy_detection_dBm,
                    rng=self.rng_manager.get_stream("gateway", gw_id),
                )
            )

        # Générer les nœuds aléatoirement dans l'aire et assigner un SF/power initiaux
        self.nodes = []
        for idx in range(self.num_nodes):
            node_id = next_node_id()
            if cfg_nodes and idx < len(cfg_nodes):
                ncfg = cfg_nodes[idx]
                x = ncfg["x"]
                y = ncfg["y"]
                sf = ncfg.get(
                    "sf",
                    (
                        self.fixed_sf
                        if self.fixed_sf is not None
                        else random.randint(7, 12)
                    ),
                )
                tx_power = ncfg.get(
                    "tx_power",
                    self.fixed_tx_power if self.fixed_tx_power is not None else 14.0,
                )
            else:
                x = self.pos_rng.random() * area_size
                y = self.pos_rng.random() * area_size
                sf = (
                    self.fixed_sf
                    if self.fixed_sf is not None
                    else random.randint(7, 12)
                )
                tx_power = (
                    self.fixed_tx_power if self.fixed_tx_power is not None else 14.0
                )
            channel = self.multichannel.select_mask(0xFFFF)
            node = Node(
                node_id,
                x,
                y,
                sf,
                tx_power,
                channel=channel,
                class_type=self.node_class,
                battery_capacity_j=self.battery_capacity_j,
                beacon_loss_prob=self.beacon_loss_prob,
                beacon_drift=(
                    random.gauss(0.0, self.clock_accuracy)
                    if self.clock_accuracy > 0.0
                    else 0.0
                ),
            )
            node.simulator = self
            node.assigned_channel_index = self.channel_index(channel)
            node._warmup_remaining = self.warm_up_intervals
            node._log_after = self.log_mean_after
            # Enregistrer les états initiaux du nœud pour rapport ultérieur
            node.initial_x = x
            node.initial_y = y
            node.initial_sf = sf
            node.initial_tx_power = tx_power
            # Attributs supplémentaires pour mobilité et ADR
            node.history = (
                []
            )  # Historique des 20 dernières transmissions (snr, delivered)
            node.in_transmission = (
                False  # Indique si le nœud est actuellement en transmission
            )
            node.current_end_time = None  # Instant de fin de la transmission en cours (si in_transmission True)
            node.last_rssi = (
                None  # Dernier meilleur RSSI mesuré pour la transmission en cours
            )
            node.last_snr = (
                None  # Dernier meilleur SNR mesuré pour la transmission en cours
            )
            if self.mobility_enabled:
                self.mobility_model.assign(node)
            node.rng = self.rng_manager.get_stream("traffic", node_id)
            self.nodes.append(node)

        # Configurer le serveur réseau avec les références pour ADR
        self.network_server.adr_enabled = self.adr_server
        self.network_server.nodes = self.nodes
        self.network_server.gateways = self.gateways
        self.network_server.channel = self.channel
        self.tick_ns = tick_ns
        if self.tick_ns is not None and self.tick_ns <= 0:
            raise ValueError("tick_ns must be positive")

        # File d'événements (min-heap)
        self.event_queue: list[Event] = []
        self.node_map = {node.id: node for node in self.nodes}
        self.current_time = 0.0
        self.event_id_counter = 0

        # Gestion automatique du rafraîchissement QoS
        self.qos_manager = None
        self.qos_algorithm = None
        self._next_qos_reconfig_time: float | None = None

        # Statistiques cumulatives
        self.packets_sent = 0
        self.packets_delivered = 0
        self.packets_lost_collision = 0
        self.packets_lost_snir = 0
        self.packets_lost_no_signal = 0
        self.total_energy_J = 0.0
        self.energy_nodes_J = 0.0
        self.energy_gateways_J = 0.0
        self.total_delay = 0.0
        self.delivered_count = 0
        # Counters for PDR computation
        self.tx_attempted = 0
        self.rx_delivered = 0
        self.retransmissions = 0

        # Gestion des transmissions simultanées pour calculer l'interférence
        self._interference_tracker = InterferenceTracker(
            self.rng_manager.get_stream("interference", 0)
        )
        # Alias rétrocompatible utilisé par certains tests pour intercepter les
        # enregistrements de transmissions et les mesures d'interférence.
        self._tx_manager = self._interference_tracker

        # Journal des événements (pour export CSV)
        self.events_log: list[dict] = []
        # Accès direct aux événements par identifiant
        self._events_log_map: dict[int, dict] = {}
        # Nœuds de classe C actuellement maintenus en écoute périodique
        self._class_c_polling_nodes: set[int] = set()
        self.out_of_service_queue: deque[tuple[int, str, float]] = deque()

        # Planifier le premier envoi de chaque nœud
        for node in self.nodes:
            if self.transmission_mode.lower() == "random":
                if self.lock_step_poisson:
                    if self.packets_to_send == 0:
                        raise ValueError(
                            "lock_step_poisson requires packets_to_send > 0"
                        )
                    node.precompute_poisson_arrivals(
                        self.packet_interval,
                        self.packets_to_send,
                        self.interval_rng,
                        variation=self.interval_variation,
                    )
                else:
                    node.ensure_poisson_arrivals(
                        node.last_tx_time,
                        self.packet_interval,
                        self.interval_rng,
                        min_interval=max(
                            node.last_airtime, self.first_packet_min_delay
                        ),
                        variation=self.interval_variation,
                        limit=(self.packets_to_send if self.packets_to_send else None),
                    )
                t0 = node.arrival_queue.pop(0)
            else:
                t0 = random.random() * self.packet_interval
                node.arrival_queue.append(t0)
                node.arrival_interval_sum += t0
                node.arrival_interval_count += 1
                node._last_arrival_time = t0
            self.schedule_event(
                node,
                t0,
                reason=(
                    "poisson"
                    if self.transmission_mode.lower() == "random"
                    else "periodic"
                ),
            )
            # Planifier le premier changement de position si la mobilité est activée
            if self.mobility_enabled:
                self.schedule_mobility(node, self.mobility_model.step)
            if node.class_type.upper() in ("B", "C"):
                if node.class_type.upper() == "C":
                    self._ensure_class_c_polling(node, 0.0)
                else:
                    eid = self.event_id_counter
                    self.event_id_counter += 1
                    self._push_event(0.0, EventType.RX_WINDOW, eid, node.id)

        # Première émission de beacon pour la synchronisation Class B
        eid = self.event_id_counter
        self.event_id_counter += 1
        self._push_event(0.0, EventType.BEACON, eid, 0)
        self.last_beacon_time = 0.0
        self.network_server.last_beacon_time = 0.0

        # Indicateur d'exécution de la simulation
        self.running = True

    # ------------------------------------------------------------------
    # Internal time helpers
    # ------------------------------------------------------------------
    def _seconds_to_ticks(self, t: float) -> int | float:
        if self.tick_ns is None:
            return t
        return int(round(t * 1_000_000_000 / self.tick_ns)) * self.tick_ns

    def _ticks_to_seconds(self, t: int | float) -> float:
        if self.tick_ns is None:
            return float(t)
        return float(t) / 1_000_000_000

    def _quantize(self, t: float) -> float:
        return self._ticks_to_seconds(self._seconds_to_ticks(t))

    def _push_event(self, time_s: float, event_type: EventType, eid: int, node_id: int) -> None:
        heapq.heappush(
            self.event_queue,
            Event(self._seconds_to_ticks(time_s), event_type, eid, node_id),
        )

    def _ensure_class_c_polling(self, node: Node | int, when: float) -> None:
        """Garantit qu'un nœud de classe C possède une fenêtre RX planifiée."""

        node_id = node if isinstance(node, int) else node.id
        scheduled_time = self._quantize(when)
        earliest: float | None = None
        for evt in self.event_queue:
            if evt.type != EventType.RX_WINDOW or evt.node_id != node_id:
                continue
            evt_time = self._ticks_to_seconds(evt.time)
            if earliest is None or evt_time < earliest:
                earliest = evt_time
        tolerance = 1e-9
        if earliest is not None and earliest <= scheduled_time + tolerance:
            self._class_c_polling_nodes.add(node_id)
            return
        eid = self.event_id_counter
        self.event_id_counter += 1
        self._push_event(scheduled_time, EventType.RX_WINDOW, eid, node_id)
        self._class_c_polling_nodes.add(node_id)

    # ------------------------------------------------------------------
    # Gestion du rafraîchissement QoS
    # ------------------------------------------------------------------
    def _cancel_qos_reconfigure_event(self) -> None:
        if not self.event_queue:
            return
        kept = [evt for evt in self.event_queue if evt.type != EventType.QOS_RECONFIG]
        if len(kept) != len(self.event_queue):
            heapq.heapify(kept)
            self.event_queue = kept
        self._next_qos_reconfig_time = None

    def _schedule_qos_reconfigure_event(self, when: float) -> None:
        self._cancel_qos_reconfigure_event()
        ticks = self._quantize(when)
        event_id = self.event_id_counter
        self.event_id_counter += 1
        self._push_event(ticks, EventType.QOS_RECONFIG, event_id, 0)
        self._next_qos_reconfig_time = when

    def _handle_qos_reconfigure_event(self) -> None:
        manager = getattr(self, "qos_manager", None)
        if manager is None:
            self._cancel_qos_reconfigure_event()
            return
        if not getattr(self, "qos_active", False):
            self._cancel_qos_reconfigure_event()
            return
        algorithm = getattr(self, "qos_algorithm", None)
        if not algorithm:
            self._cancel_qos_reconfigure_event()
            return
        self.request_qos_refresh(reason="periodic")

    def _on_qos_applied(self, manager) -> None:
        self.qos_manager = manager
        interval = getattr(manager, "reconfig_interval_s", None)
        if interval is None or interval <= 0.0:
            self._cancel_qos_reconfigure_event()
            return
        base_time = self.current_time if self.running else 0.0
        self._schedule_qos_reconfigure_event(base_time + float(interval))

    def request_qos_refresh(self, *, reason: str = "unspecified") -> None:
        manager = getattr(self, "qos_manager", None)
        if manager is None or not getattr(self, "qos_active", False):
            return
        algorithm = getattr(self, "qos_algorithm", None)
        if not algorithm:
            return
        if not manager.clusters:
            return
        try:
            needs_refresh = manager._should_refresh_context(self)
        except Exception:  # pragma: no cover - robust logging
            logger.exception("Échec de l'évaluation du rafraîchissement QoS (%s).", reason)
            return
        if not needs_refresh:
            return
        diag_logger.debug("Déclenchement QoS (%s) à t=%.3fs", reason, self.current_time)
        manager.apply(self, algorithm)

    def _notify_qos_metrics_update(self, node: Node) -> None:
        if getattr(self, "qos_manager", None) is None:
            return
        if not getattr(self, "qos_active", False):
            return
        self.request_qos_refresh(reason="metrics")

    def _apply_rx_window_energy(self, node: "Node", window_time: float) -> tuple[str, float, float]:
        """Account for the energy spent during an RX window."""

        state = "listen" if node.profile.listen_current_a > 0.0 else "rx"
        current = (
            node.profile.listen_current_a
            if node.profile.listen_current_a > 0.0
            else node.profile.rx_current_a
        )
        duration = node.profile.rx_window_duration
        energy_J = current * node.profile.voltage_v * duration
        prev_energy = node.energy_consumed
        node.add_energy(energy_J, state, duration_s=duration)
        delta = node.energy_consumed - prev_energy
        if delta > 0.0:
            self.energy_nodes_J += delta
            self.total_energy_J += delta
        node.last_state_time = max(node.last_state_time, window_time + duration)
        if node.class_type.upper() != "C":
            node.state = "sleep"
        return state, current, duration

    def _mark_out_of_service(self, node: Node, reason: str) -> None:
        if getattr(node, "out_of_service", False):
            return
        node.out_of_service = True
        self.out_of_service_queue.append((node.id, reason, self.current_time))
        diag_logger.info(
            "Nœud %s mis hors service (%s) à t=%.3fs", node.id, reason, self.current_time
        )

    def schedule_event(self, node: Node, time: float, *, reason: str = "poisson"):
        """Planifie un événement de transmission pour un nœud."""
        if not node.alive:
            return
        if getattr(node, "out_of_service", False):
            self._mark_out_of_service(node, "already_out_of_service")
            return
        if getattr(node, "_qos_blocked_channel", False):
            self._mark_out_of_service(node, "blocked_channel")
            return
        requested_time = time
        if (
            not self.pure_poisson_mode
            and node.current_end_time is not None
            and time < node.current_end_time
        ):
            time = node.current_end_time
            reason = "overlap"
        event_id = self.event_id_counter
        self.event_id_counter += 1
        if self.duty_cycle_manager and not self.pure_poisson_mode:
            enforced = self.duty_cycle_manager.enforce(node.id, time)
            if enforced > time:
                time = enforced
                reason = "duty_cycle"
        time = self._quantize(time)
        if node.channel is None:
            node.channel = self.multichannel.select_mask(getattr(node, "chmask", 0xFFFF))
        if node.channel is None:
            self._mark_out_of_service(node, "missing_channel")
            return
        node.channel.detection_threshold_dBm = Channel.flora_detection_threshold(
            node.sf, node.channel.bandwidth
        ) + node.channel.sensitivity_margin_dB
        self._push_event(time, EventType.TX_START, event_id, node.id)
        node.interval_log.append(
            {
                "poisson_time": requested_time,
                "tx_time": time,
                "reason": reason,
            }
        )
        logger.debug(
            f"Scheduled transmission {event_id} for node {node.id} at t={time:.2f}s"

        )

    def update_duty_cycle(self, node_id: int, max_duty_cycle: int) -> None:
        """Update the duty-cycle configuration following a LoRaWAN command."""
        if self.duty_cycle_manager is None:
            return

        exponent = max(0, int(max_duty_cycle))
        duty = 2.0 ** (-exponent)
        if duty <= 0.0:
            duty = float.fromhex("0x1p-1022")

        self.duty_cycle_manager.duty_cycle = duty

        node = self.node_map.get(node_id)
        if node is None:
            return

        duration = getattr(node, "last_airtime", 0.0)
        start_time = getattr(node, "last_tx_time", None)
        if duration <= 0.0 or start_time is None:
            return

        earliest = max(self.current_time, start_time + duration / duty)
        self.duty_cycle_manager.next_tx_time[node_id] = earliest

        rescheduled = False
        tick_deadline = self._seconds_to_ticks(earliest)
        deadline_seconds = self._ticks_to_seconds(tick_deadline)
        for event in self.event_queue:
            if event.node_id == node_id and event.type == EventType.TX_START:
                if event.time < tick_deadline:
                    event.time = tick_deadline
                    rescheduled = True

        if rescheduled:
            heapq.heapify(self.event_queue)
            if node.interval_log:
                node.interval_log[-1]["reason"] = "duty_cycle"
                node.interval_log[-1]["tx_time"] = deadline_seconds
        elif node.interval_log:
            node.interval_log[-1]["tx_time"] = deadline_seconds

    def schedule_mobility(self, node: Node, time: float):
        """Planifie un événement de mobilité (déplacement aléatoire) pour un nœud à l'instant donné."""
        if not node.alive:
            return
        time = self._quantize(time)
        event_id = self.event_id_counter
        self.event_id_counter += 1
        self._push_event(time, EventType.MOBILITY, event_id, node.id)
        logger.debug(
            f"Scheduled mobility {event_id} for node {node.id} at t={time:.2f}s"
        )

    def step(self) -> bool:
        """Exécute le prochain événement planifié. Retourne False si plus d'événement à traiter."""
        if not self.running or not self.event_queue:
            return False
        # Extraire le prochain événement (le plus tôt dans le temps)
        event = heapq.heappop(self.event_queue)
        time = self._ticks_to_seconds(event.time)
        priority = event.type
        event_id = event.id
        node = self.node_map.get(event.node_id)
        if node is None and priority not in {EventType.BEACON, EventType.QOS_RECONFIG}:
            return True
        # Avancer le temps de simulation et mettre à jour l'état PHY
        delta = time - self.current_time
        self.current_time = time
        if delta > 0:
            for ch in self.multichannel.channels:
                if getattr(ch, "omnet_phy", None):
                    ch.omnet_phy.update(delta)
        if node is not None:
            prev_energy = node.energy_consumed
            node.consume_until(time)
            delta_energy = node.energy_consumed - prev_energy
            if delta_energy > 0.0:
                self.energy_nodes_J += delta_energy
                self.total_energy_J += delta_energy
            if not node.alive:
                return True

        if priority == EventType.TX_START:
            # Début d'une transmission émise par 'node'
            node_id = node.id
            if node.channel is None:
                fallback = self.multichannel.channels[0] if self.multichannel.channels else None
                if fallback is None or getattr(node, "_qos_blocked_channel", False):
                    diag_logger.info(
                        "Exclusion du nœud %s sans canal au début de TX_START", node_id
                    )
                    self._mark_out_of_service(node, "tx_start_no_channel")
                    return True
                node.channel = fallback
                diag_logger.info(
                    "Affectation de secours du nœud %s sur le canal %s", node_id, fallback
                )
            node.last_tx_time = time
            if node._nb_trans_left <= 0:
                node._nb_trans_left = max(1, node.nb_trans)
            node._nb_trans_left -= 1
            if getattr(node.channel, "omnet_phy", None):
                node.channel.omnet_phy.start_tx()
            sf = node.sf
            tx_power = node.tx_power
            # Durée de la transmission
            duration = node.channel.airtime(sf, payload_size=self.payload_size_bytes)
            node.last_airtime = duration
            node.total_airtime += duration
            end_time = time + duration
            if self.duty_cycle_manager and not self.pure_poisson_mode:
                self.duty_cycle_manager.update_after_tx(node_id, time, duration)
            # Mettre à jour les compteurs de paquets émis
            self.packets_sent += 1
            self.tx_attempted += 1
            node.increment_sent()
            # Énergie consommée par la transmission (E = I * V * t)
            current_a = node.profile.get_tx_current(tx_power)
            energy_J = current_a * node.profile.voltage_v * duration
            prev = node.energy_consumed
            node.add_energy(energy_J, "tx", duration_s=duration)
            delta = node.energy_consumed - prev
            self.total_energy_J += delta
            self.energy_nodes_J += delta
            if not node.alive:
                return True
            node.state = "tx"
            node.last_state_time = end_time
            # Marquer le nœud comme en cours de transmission
            node.in_transmission = True
            node.current_end_time = end_time

            # Actualiser les offsets temps/fréquence utilisés pour cette émission
            if hasattr(node, "update_offsets"):
                node.update_offsets()

            heard_by_any = False
            best_rssi = None
            # Propagation du paquet vers chaque passerelle
            best_snr = None
            best_snir = None
            for gw in self.gateways:
                distance = node.distance_to(gw)
                kwargs = {
                    "freq_offset_hz": getattr(node, "current_freq_offset", 0.0),
                    "sync_offset_s": getattr(node, "current_sync_offset", 0.0),
                    "tx_pos": (node.x, node.y, getattr(node, "altitude", 0.0)),
                    "rx_pos": (gw.x, gw.y, getattr(gw, "altitude", 0.0)),
                }
                if hasattr(node.channel, "_obstacle_loss"):
                    kwargs["tx_angle"] = getattr(
                        node, "orientation_az", getattr(node, "direction", 0.0)
                    )
                    kwargs["rx_angle"] = getattr(
                        gw, "orientation_az", getattr(gw, "direction", 0.0)
                    )
                else:
                    kwargs["tx_angle"] = (
                        getattr(node, "orientation_az", 0.0),
                        getattr(node, "orientation_el", 0.0),
                    )
                    kwargs["rx_angle"] = (
                        getattr(gw, "orientation_az", 0.0),
                        getattr(gw, "orientation_el", 0.0),
                    )
                rssi, snr = node.channel.compute_rssi(
                    tx_power,
                    distance,
                    sf,
                    **kwargs,
                )
                noise_dBm = node.channel.last_noise_dBm
                noise_lin = 10 ** (noise_dBm / 10.0)
                freq_hz = getattr(node.channel, "last_freq_hz", node.channel.frequency_hz)
                interference_mw = self._interference_tracker.total_interference(
                    gw.id,
                    freq_hz,
                    sf,
                    time,
                    end_time,
                    base_noise_mW=noise_lin,
                    alpha_isf=getattr(node.channel, "alpha_isf", 0.0),
                    fading_std=getattr(node.channel, "snir_fading_std", 0.0),
                )
                if interference_mw < 0.0:
                    interference_mw = 0.0
                fading_std = getattr(node.channel, "snir_fading_std", 0.0)
                if fading_std > 0.0:
                    rng = getattr(node.channel, "rng", None) or create_generator()
                    rssi += float(rng.normal(0.0, fading_std))
                    interference_mw *= 10 ** (
                        float(rng.normal(0.0, fading_std)) / 10.0
                    )
                total_noise_mw = noise_lin + interference_mw
                node.channel.last_interference_mW = interference_mw
                node.channel.current_interference_mW = interference_mw
                gain_dB = getattr(gw, "rx_gain_dB", 0.0)
                gain_lin = 10 ** (gain_dB / 10.0)
                rssi += gain_dB
                total_noise_mw *= gain_lin
                snr_effective = rssi - 10 * math.log10(total_noise_mw)
                energy_threshold = max(
                    node.channel.energy_detection_dBm,
                    getattr(gw, "energy_detection_dBm", -float("inf")),
                )
                if rssi < energy_threshold:
                    continue
                # Enregistrer la transmission pour l'interférence future
                self._interference_tracker.add(
                    gw.id,
                    freq_hz,
                    sf,
                    rssi,
                    end_time,
                    event_id,
                    start_time=time,
                )
                if not self.pure_poisson_mode:
                    if rssi < node.channel.detection_threshold_dBm:
                        continue  # trop faible pour être détecté
                    snr_threshold = (
                        node.channel.sensitivity_dBm.get(sf, -float("inf"))
                        - noise_dBm
                    )
                    if snr_effective < snr_threshold:
                        continue  # signal trop faible pour être reçu
                heard_by_any = True
                if best_rssi is None or rssi > best_rssi:
                    best_rssi = rssi
                if best_snr is None or snr > best_snr:
                    best_snr = snr
                if best_snir is None or snr_effective > best_snir:
                    best_snir = snr_effective
                # Démarrer la réception à la passerelle (gestion des collisions et capture)
                if self.capture_mode is not None:
                    capture_mode = self.capture_mode
                else:
                    capture_mode = (
                        "omnet"
                        if node.channel.phy_model == "omnet"
                        else (
                            "flora"
                            if node.channel.phy_model.startswith("flora")
                            else (
                                "advanced" if node.channel.advanced_capture else "basic"
                            )
                        )
                    )

                gw.start_reception(
                    event_id,
                    node_id,
                    sf,
                    rssi,
                    end_time,
                    node.channel.capture_threshold_dB,
                    self.current_time,
                    freq_hz,
                    self.min_interference_time,
                    freq_offset=getattr(node, "current_freq_offset", 0.0),
                    sync_offset=getattr(node, "current_sync_offset", 0.0),
                    bandwidth=node.channel.bandwidth,
                    noise_floor=noise_dBm,
                    snir=snr_effective,
                    capture_mode=capture_mode,
                    flora_phy=(
                        node.channel.flora_phy
                        if node.channel.phy_model.startswith("flora")
                        else None
                    ),
                    orthogonal_sf=node.channel.orthogonal_sf,
                    capture_window_symbols=node.channel.capture_window_symbols,
                    non_orth_delta=getattr(node.channel, "non_orth_delta", None),
                    snir_fading_std=getattr(node.channel, "snir_fading_std", 0.0),
                    marginal_snir_db=getattr(node.channel, "marginal_snir_margin_db", 0.0),
                    marginal_drop_prob=getattr(
                        node.channel, "marginal_snir_drop_prob", 0.0
                    ),
                )

            # Retenir le meilleur RSSI/SNR mesuré pour cette transmission
            node.last_rssi = best_rssi if heard_by_any else None
            quality_metric = best_snir if best_snir is not None else best_snr
            node.last_snr = quality_metric if heard_by_any else None
            # Planifier l'événement de fin de transmission correspondant
            end_time = self._quantize(end_time)
            self._push_event(end_time, EventType.TX_END, event_id, node.id)
            # Planifier les fenêtres de réception LoRaWAN
            rx1, rx2 = node.schedule_receive_windows(end_time)
            rx1 = self._quantize(rx1)
            rx2 = self._quantize(rx2)
            ev1 = self.event_id_counter
            self.event_id_counter += 1
            self._push_event(rx1, EventType.RX_WINDOW, ev1, node.id)
            ev2 = self.event_id_counter
            self.event_id_counter += 1
            self._push_event(rx2, EventType.RX_WINDOW, ev2, node.id)

            # Journaliser l'événement de transmission (résultat inconnu à ce stade)
            log_entry = {
                "event_id": event_id,
                "node_id": node_id,
                "sf": sf,
                "start_time": time,
                "end_time": end_time,
                "frequency_hz": node.channel.frequency_hz,
                "energy_J": energy_J,
                "heard": heard_by_any,
                "rssi_dBm": best_rssi,
                "snr_dB": best_snr,
                "snir_dB": best_snir,
                "result": None,
                "gateway_id": None,
            }
            self.events_log.append(log_entry)
            self._events_log_map[event_id] = log_entry
            return True

        elif priority == EventType.TX_END:
            # Fin d'une transmission – traitement de la réception/perte
            self._interference_tracker.remove(event_id)
            node_id = node.id
            # Marquer la fin de transmission du nœud
            if getattr(node.channel, "omnet_phy", None):
                node.channel.omnet_phy.stop_tx()
            node.in_transmission = False
            node.current_end_time = None
            node.state = "rx" if node.class_type.upper() == "C" else "processing"
            node.channel.current_interference_mW = 0.0
            # Notifier chaque passerelle de la fin de réception
            for gw in self.gateways:
                gw.end_reception(event_id, self.network_server, node_id)
            # Vérifier si le paquet a été reçu par au moins une passerelle
            delivered = event_id in self.network_server.received_events
            if delivered:
                self.packets_delivered += 1
                self.rx_delivered += 1
                node.increment_success()
                # Délai = temps de fin - temps de début de l'émission
                start_time = self._events_log_map[event_id]["start_time"]
                delay = self.current_time - start_time
                self.total_delay += delay
                self.delivered_count += 1
            else:
                # Identifier la cause de perte: collision ou absence de couverture
                log_entry = self._events_log_map[event_id]
                heard = log_entry["heard"]
                if heard:
                    self.packets_lost_collision += 1
                    reason = self.network_server.collision_reasons.get(event_id)
                    if reason in {"snir_below_threshold", "snir_marginal"}:
                        self.packets_lost_snir += 1
                    node.increment_collision(
                        snir_limit=reason in {"snir_below_threshold", "snir_marginal"}
                    )
                else:
                    self.packets_lost_no_signal += 1
            # Mettre à jour le résultat et la passerelle du log de l'événement
            entry = self._events_log_map[event_id]
            entry["result"] = (
                "Success"
                if delivered
                else ("CollisionLoss" if entry["heard"] else "NoCoverage")
            )
            entry["gateway_id"] = (
                self.network_server.event_gateway.get(event_id, None)
                if delivered
                else None
            )

            snr_value = entry.get("snr_dB")
            rssi_value = entry.get("rssi_dBm")

            if event_id in self.network_server.event_snir:
                snr_value = self.network_server.event_snir[event_id]
            if event_id in self.network_server.event_rssi:
                rssi_value = self.network_server.event_rssi[event_id]

            if snr_value is None:
                snr_value = float("nan")
            if rssi_value is None:
                rssi_value = float("nan")

            entry["snr_dB"] = snr_value
            entry["rssi_dBm"] = rssi_value

            if self.debug_rx:
                if delivered:
                    gw_id = self.network_server.event_gateway.get(event_id, None)
                    logger.debug(
                        f"t={self.current_time:.2f} Packet {event_id} from node {node_id} reçu via GW {gw_id}"
                    )
                else:
                    reason = "Collision" if log_entry["heard"] else "NoCoverage"
                    logger.debug(
                        f"t={self.current_time:.2f} Packet {event_id} from node {node_id} perdu ({reason})"
                    )

            # Mettre à jour l'historique du nœud pour calculer les statistiques
            # récentes et éventuellement déclencher l'ADR.
            snr_value = None
            rssi_value = None
            if delivered and node.last_snr is not None:
                snr_value = node.last_snr
            if delivered and node.last_rssi is not None:
                rssi_value = node.last_rssi
            node.history.append(
                {"snr": snr_value, "rssi": rssi_value, "delivered": delivered}
            )
            if len(node.history) > 20:
                node.history.pop(0)
            self._notify_qos_metrics_update(node)

            # Gestion Adaptive Data Rate (ADR)
            if self.adr_node:
                # Only track history here; the actual adaptation now relies on
                # the standard adr_ack_cnt mechanism implemented in :class:`Node`.
                pass
                
            # Planifier retransmissions restantes ou prochaine émission
            if node._nb_trans_left > 0:
                self.retransmissions += 1
                self.schedule_event(
                    node, self.current_time + 1.0, reason="retransmission"
                )
            else:
                if (
                    self.packets_to_send == 0
                    or node.packets_sent < self.packets_to_send
                ):
                    if self.transmission_mode.lower() == "random":
                        if not self.lock_step_poisson:
                            node.ensure_poisson_arrivals(
                                node.last_tx_time,
                                self.packet_interval,
                                self.interval_rng,
                                variation=self.interval_variation,
                                limit=(
                                    self.packets_to_send
                                    if self.packets_to_send
                                    else None
                                ),
                            )
                        next_time = node.arrival_queue.pop(0)
                    else:
                        next_time = node._last_arrival_time + self.packet_interval
                        node.arrival_interval_sum += self.packet_interval
                        node.arrival_interval_count += 1
                        node._last_arrival_time = next_time
                    self.schedule_event(
                        node,
                        next_time if self.pure_poisson_mode else max(next_time, self.current_time),
                        reason=(
                            "poisson"
                            if self.transmission_mode.lower() == "random"
                            else "periodic"
                        ),
                    )
                else:
                    logger.debug(
                        "Packet limit reached for node %s – no more events for this node.",
                        node.id,
                    )

                if self.packets_to_send != 0 and all(
                    n.packets_sent >= self.packets_to_send for n in self.nodes
                ):
                    new_queue: list[Event] = []
                    class_c_cleanup: set[int] = set()
                    rx_windows_kept: dict[int, int] = {}
                    for evt in self.event_queue:
                        if evt.type == EventType.PING_SLOT:
                            # Les créneaux ping de classe B ne doivent pas
                            # prolonger la simulation une fois la limite de
                            # paquets atteinte.
                            continue
                        if evt.type == EventType.TX_END:
                            new_queue.append(evt)
                        elif evt.type == EventType.RX_WINDOW:
                            node = self.node_map[evt.node_id]
                            if (
                                node.packets_sent < self.packets_to_send
                                or node.downlink_pending > 0
                            ):
                                new_queue.append(evt)
                            elif node.class_type.upper() == "C":
                                class_c_cleanup.add(node.id)
                                event_time = self._ticks_to_seconds(evt.time)
                                prev_energy = node.energy_consumed
                                node.consume_until(event_time)
                                state = (
                                    "listen"
                                    if node.profile.listen_current_a > 0.0
                                    else "rx"
                                )
                                current = (
                                    node.profile.listen_current_a
                                    if node.profile.listen_current_a > 0.0
                                    else node.profile.rx_current_a
                                )
                                duration = node.profile.rx_window_duration
                                energy_j = current * node.profile.voltage_v * duration
                                node.add_energy(energy_j, state, duration_s=duration)
                                delta = node.energy_consumed - prev_energy
                                if delta > 0.0:
                                    self.energy_nodes_J += delta
                                    self.total_energy_J += delta
                                node.last_state_time = max(
                                    node.last_state_time, event_time + duration
                                )
                                node.state = "rx"
                            else:
                                kept = rx_windows_kept.get(node.id, 0)
                                # Conserver au moins les deux prochaines
                                # fenêtres RX pour permettre au serveur de
                                # planifier un downlink tant que l'événement
                                # n'a pas été traité. Sans cela, la dernière
                                # transmission d'un nœud de classe A pouvait
                                # se voir retirer ses fenêtres RX avant même
                                # que le test ait l'occasion d'y injecter un
                                # downlink.
                                if kept < 2:
                                    new_queue.append(evt)
                                    rx_windows_kept[node.id] = kept + 1
                                else:
                                    event_time = self._ticks_to_seconds(evt.time)
                                    self._apply_rx_window_energy(node, event_time)
                    heapq.heapify(new_queue)
                    self.event_queue = new_queue
                    for node_id in class_c_cleanup:
                        self._class_c_polling_nodes.discard(node_id)
                    # Stop scheduling further mobility events once every node
                    # reached the packet limit to ensure the simulation
                    # completes when using fast forward.
                    self.mobility_enabled = False
                    logger.debug(
                        "Packet limit reached – no more new events will be scheduled."
                    )

            return True

        elif priority == EventType.RX_WINDOW:
            # Fenêtre de réception RX1/RX2 pour un nœud
            state = "rx"
            current = node.profile.rx_current_a
            duration = node.profile.rx_window_duration
            if node.class_type.upper() != "C":
                state, current, duration = self._apply_rx_window_energy(node, time)
            if not node.alive:
                return True
            self.network_server.deliver_scheduled(node.id, time)
            delivered = False
            for gw in self.gateways:
                downlink = gw.pop_downlink(node.id)
                if not downlink:
                    continue
                delivered = True
                if len(downlink) == 4:
                    frame, data_rate, tx_power, dl_channel = downlink
                else:
                    frame, data_rate, tx_power = downlink
                    dl_channel = None
                payload_len = 0
                if hasattr(frame, "payload"):
                    try:
                        payload_len = len(frame.payload)
                    except Exception:
                        pass
                elif hasattr(frame, "to_bytes"):
                    try:
                        payload_len = len(frame.to_bytes())
                    except Exception:
                        pass
                sf = node.sf
                if data_rate is not None:
                    from .lorawan import DR_TO_SF

                    sf = DR_TO_SF.get(data_rate, node.sf)
                channel_obj = dl_channel if dl_channel is not None else node.channel
                duration_dl = channel_obj.airtime(sf, payload_len)
                tx_power_dl = (
                    tx_power if tx_power is not None else gw.select_downlink_power(node)
                )
                current_gw = gw.profile.get_tx_current(tx_power_dl)
                energy_tx = current_gw * gw.profile.voltage_v * duration_dl
                include_transients = getattr(
                    gw.profile, "include_transients", True
                )
                ramp = 0.0
                if include_transients:
                    ramp = current_gw * gw.profile.voltage_v * (
                        gw.profile.ramp_up_s + gw.profile.ramp_down_s
                    )
                total_tx = energy_tx + ramp
                self.energy_gateways_J += total_tx
                self.total_energy_J += total_tx
                gw.add_energy(energy_tx, "tx")
                if ramp > 0.0:
                    gw.add_energy(ramp, "ramp")
                preamble_J = 0.0
                if include_transients:
                    preamble_J = (
                        gw.profile.preamble_current_a
                        * gw.profile.voltage_v
                        * gw.profile.preamble_time_s
                    )
                if preamble_J > 0.0:
                    self.energy_gateways_J += preamble_J
                    self.total_energy_J += preamble_J
                    gw.add_energy(preamble_J, "preamble")
                if node.class_type.upper() != "C":
                    extra_time = max(duration_dl - duration, 0.0)
                    if extra_time > 0.0:
                        extra_energy = current * node.profile.voltage_v * extra_time
                        prev_energy = node.energy_consumed
                        node.add_energy(
                            extra_energy,
                            state,
                            duration_s=extra_time,
                        )
                        delta = node.energy_consumed - prev_energy
                        if delta > 0.0:
                            self.energy_nodes_J += delta
                            self.total_energy_J += delta
                distance = node.distance_to(gw)
                kwargs = {
                    "freq_offset_hz": 0.0,
                    "sync_offset_s": 0.0,
                    "tx_pos": (gw.x, gw.y, getattr(gw, "altitude", 0.0)),
                    "rx_pos": (node.x, node.y, getattr(node, "altitude", 0.0)),
                }
                if hasattr(node.channel, "_obstacle_loss"):
                    kwargs["tx_angle"] = getattr(gw, "orientation_az", getattr(gw, "direction", 0.0))
                    kwargs["rx_angle"] = getattr(node, "orientation_az", getattr(node, "direction", 0.0))
                else:
                    kwargs["tx_angle"] = (
                        getattr(gw, "orientation_az", 0.0),
                        getattr(gw, "orientation_el", 0.0),
                    )
                    kwargs["rx_angle"] = (
                        getattr(node, "orientation_az", 0.0),
                        getattr(node, "orientation_el", 0.0),
                    )
                reference_power = tx_power_dl if tx_power_dl is not None else node.tx_power
                rssi, snr = node.channel.compute_rssi(
                    reference_power,
                    distance,
                    sf,
                    **kwargs,
                )
                noise_dBm = node.channel.last_noise_dBm
                if not self.pure_poisson_mode:
                    if rssi < node.channel.detection_threshold_dBm:
                        node.downlink_pending = max(0, node.downlink_pending - 1)
                        continue
                    snr_threshold = (
                        node.channel.sensitivity_dBm.get(sf, -float("inf"))
                        - noise_dBm
                    )
                    if snr >= snr_threshold:
                        node.handle_downlink(frame)
                        delivered = True
                    else:
                        node.downlink_pending = max(0, node.downlink_pending - 1)
                else:
                    node.handle_downlink(frame)
                    delivered = True
                break
            # Replanifier selon la classe du nœud
            if node.class_type.upper() == "C":
                scheduler = getattr(self.network_server, "scheduler", None)
                next_time: float | None = None
                if scheduler is not None and hasattr(scheduler, "next_time"):
                    try:
                        next_time = scheduler.next_time(node.id)
                    except Exception:
                        next_time = None
                pending_gateway = any(
                    bool(getattr(gw, "downlink_buffer", {}).get(node.id))
                    for gw in self.gateways
                )
                if (
                    not delivered
                    and node.downlink_pending > 0
                    and next_time is None
                    and not pending_gateway
                ):
                    node.downlink_pending = max(0, node.downlink_pending - 1)
                needs_polling = (
                    node.downlink_pending > 0
                    or next_time is not None
                    or pending_gateway
                )
                if needs_polling:
                    nxt = time + self.class_c_rx_interval
                    self._ensure_class_c_polling(node.id, nxt)
                else:
                    # Arrêt explicite du polling : aucun downlink n'est attendu
                    # tant que le serveur ne reprogramme pas une fenêtre.
                    self._class_c_polling_nodes.discard(node.id)
            return True

        elif priority == EventType.BEACON:
            nxt = self._quantize(self.network_server.next_beacon_time(time))
            eid = self.event_id_counter
            self.event_id_counter += 1
            self._push_event(nxt, EventType.BEACON, eid, 0)
            self.last_beacon_time = time
            self.network_server.notify_beacon(time)
            end_of_cycle = nxt
            for n in self.nodes:
                if n.class_type.upper() == "B":
                    prev_energy = n.energy_consumed
                    n.consume_until(time)
                    delta = n.energy_consumed - prev_energy
                    if delta > 0.0:
                        self.energy_nodes_J += delta
                        self.total_energy_J += delta
                    if not n.alive:
                        continue
                    state = "listen" if n.profile.listen_current_a > 0.0 else "rx"
                    duration = getattr(n.profile, "beacon_listen_duration", 0.0)
                    if duration <= 0.0:
                        duration = n.profile.rx_window_duration
                    current = (
                        n.profile.listen_current_a
                        if n.profile.listen_current_a > 0.0
                        else n.profile.rx_current_a
                    )
                    energy_J = current * n.profile.voltage_v * duration
                    prev_energy = n.energy_consumed
                    n.add_energy(energy_J, state, duration_s=duration)
                    delta = n.energy_consumed - prev_energy
                    if delta > 0.0:
                        self.energy_nodes_J += delta
                        self.total_energy_J += delta
                    n.last_state_time = time + duration
                    n.state = "sleep"
                    if not n.alive:
                        continue
                    received = random.random() >= getattr(n, "beacon_loss_prob", 0.0)
                    if received:
                        n.register_beacon(time)
                    else:
                        n.miss_beacon(self.beacon_interval)
                    periodicity_value = getattr(n, "ping_slot_periodicity", 0) or 0
                    periodicity_value = max(0, min(7, periodicity_value))
                    slots_per_period = 2 ** (7 - periodicity_value)
                    interval = self.ping_slot_interval * slots_per_period
                    slot = self._quantize(
                        n.next_ping_slot_time(
                            time,
                            self.beacon_interval,
                            self.ping_slot_interval,
                            self.ping_slot_offset,
                        )
                    )
                    while slot < end_of_cycle:
                        eid = self.event_id_counter
                        self.event_id_counter += 1
                        self._push_event(slot, EventType.PING_SLOT, eid, n.id)
                        slot = self._quantize(slot + interval)
            return True

        elif priority == EventType.PING_SLOT:
            if node.class_type.upper() != "B":
                return True
            current = (
                node.profile.listen_current_a
                if node.profile.listen_current_a > 0.0
                else node.profile.rx_current_a
            )
            state = "listen" if node.profile.listen_current_a > 0.0 else "rx"
            energy_J = (
                current
                * node.profile.voltage_v
                * node.profile.rx_window_duration
            )
            prev_energy = node.energy_consumed
            node.add_energy(
                energy_J,
                state,
                duration_s=node.profile.rx_window_duration,
            )
            delta = node.energy_consumed - prev_energy
            if delta > 0.0:
                self.energy_nodes_J += delta
                self.total_energy_J += delta
            if not node.alive:
                return True
            node.last_state_time = time + node.profile.rx_window_duration
            node.state = "sleep"
            self.network_server.deliver_scheduled(node.id, time)
            for gw in self.gateways:
                downlink = gw.pop_downlink(node.id)
                if not downlink:
                    continue
                if len(downlink) == 4:
                    frame, data_rate, tx_power, dl_channel = downlink
                else:
                    frame, data_rate, tx_power = downlink
                    dl_channel = None
                payload_len = 0
                if hasattr(frame, "payload"):
                    try:
                        payload_len = len(frame.payload)
                    except Exception:
                        pass
                elif hasattr(frame, "to_bytes"):
                    try:
                        payload_len = len(frame.to_bytes())
                    except Exception:
                        pass
                sf = node.sf
                effective_dr = data_rate if data_rate is not None else node.ping_slot_dr
                if effective_dr is not None:
                    from .lorawan import DR_TO_SF

                    sf = DR_TO_SF.get(effective_dr, node.sf)
                channel_obj = dl_channel if dl_channel is not None else node.channel
                duration_dl = channel_obj.airtime(sf, payload_len)
                tx_power_dl = (
                    tx_power if tx_power is not None else gw.select_downlink_power(node)
                )
                current_gw = gw.profile.get_tx_current(tx_power_dl)
                energy_tx = current_gw * gw.profile.voltage_v * duration_dl
                include_transients = getattr(
                    gw.profile, "include_transients", True
                )
                ramp = 0.0
                if include_transients:
                    ramp = current_gw * gw.profile.voltage_v * (
                        gw.profile.ramp_up_s + gw.profile.ramp_down_s
                    )
                total_tx = energy_tx + ramp
                self.energy_gateways_J += total_tx
                self.total_energy_J += total_tx
                gw.add_energy(energy_tx, "tx")
                if ramp > 0.0:
                    gw.add_energy(ramp, "ramp")
                preamble_J = 0.0
                if include_transients:
                    preamble_J = (
                        gw.profile.preamble_current_a
                        * gw.profile.voltage_v
                        * gw.profile.preamble_time_s
                    )
                if preamble_J > 0.0:
                    self.energy_gateways_J += preamble_J
                    self.total_energy_J += preamble_J
                    gw.add_energy(preamble_J, "preamble")
                extra_time = max(duration_dl - node.profile.rx_window_duration, 0.0)
                if extra_time > 0.0:
                    extra_energy = current * node.profile.voltage_v * extra_time
                    prev_energy = node.energy_consumed
                    node.add_energy(
                        extra_energy,
                        state,
                        duration_s=extra_time,
                    )
                    delta = node.energy_consumed - prev_energy
                    if delta > 0.0:
                        self.energy_nodes_J += delta
                        self.total_energy_J += delta
                distance = node.distance_to(gw)
                kwargs = {"freq_offset_hz": 0.0, "sync_offset_s": 0.0}
                if hasattr(node.channel, "_obstacle_loss"):
                    kwargs["tx_pos"] = (gw.x, gw.y, getattr(gw, "altitude", 0.0))
                    kwargs["rx_pos"] = (node.x, node.y, getattr(node, "altitude", 0.0))
                reference_power = (
                    tx_power_dl if tx_power_dl is not None else node.tx_power
                )
                rssi, snr = node.channel.compute_rssi(
                    reference_power,
                    distance,
                    sf,
                    **kwargs,
                )
                noise_dBm = node.channel.last_noise_dBm
                if not self.pure_poisson_mode:
                    if rssi < node.channel.detection_threshold_dBm:
                        node.downlink_pending = max(0, node.downlink_pending - 1)
                        continue
                    snr_threshold = (
                        node.channel.sensitivity_dBm.get(sf, -float("inf"))
                        - noise_dBm
                    )
                    if snr >= snr_threshold:
                        node.handle_downlink(frame)
                    else:
                        node.downlink_pending = max(0, node.downlink_pending - 1)
                else:
                    node.handle_downlink(frame)
                break
            return True

        elif priority == EventType.SERVER_RX:
            self.network_server._handle_network_arrival(event_id)
            return True

        elif priority == EventType.SERVER_PROCESS:
            self.network_server._process_scheduled(event_id)
            return True

        elif priority == EventType.QOS_RECONFIG:
            self._handle_qos_reconfigure_event()
            return True

        elif priority == EventType.MOBILITY:
            # Événement de mobilité (changement de position du nœud)
            if not self.mobility_enabled:
                return True
            node_id = node.id
            if node.in_transmission:
                # Si le nœud est en cours de transmission, reporter le déplacement à la fin de celle-ci
                next_move_time = (
                    node.current_end_time
                    if node.current_end_time is not None
                    else self.current_time
                )
                self.schedule_mobility(node, next_move_time)
            else:
                # Déplacer le nœud de manière progressive
                self.mobility_model.move(node, self.current_time)
                log_entry = {
                    "event_id": event_id,
                    "node_id": node_id,
                    "sf": node.sf,
                    "start_time": time,
                    "end_time": time,
                    "frequency_hz": node.channel.frequency_hz,
                    "heard": None,
                    "result": "Mobility",
                    "energy_J": 0.0,
                    "gateway_id": None,
                    "rssi_dBm": None,
                    "snr_dB": None,
                }
                self.events_log.append(log_entry)
                self._events_log_map[event_id] = log_entry
                if self.mobility_enabled and (
                    self.packets_to_send == 0
                    or node.packets_sent < self.packets_to_send
                ):
                    self.schedule_mobility(node, time + self.mobility_model.step)
            return True

        # Si autre type d'événement (non prévu)
        return True

    def run(
        self,
        max_steps: int | None = None,
        *,
        max_time: float | None = None,
    ):
        """Exécute la simulation en traitant les événements jusqu'à épuisement ou jusqu'à une limite optionnelle."""
        step_count = 0
        while self.event_queue and self.running:
            if max_time is not None:
                next_time = self.event_queue[0].time
                if next_time > max_time:
                    break
            self.step()
            step_count += 1
            if max_steps and step_count >= max_steps:
                break
        if self.dump_intervals:
            self.dump_interval_logs()

    def stop(self):
        """Arrête la simulation en cours."""
        self.running = False

    def get_metrics(self) -> dict:
        """Retourne un dictionnaire des métriques actuelles de la simulation."""
        total_sent = self.tx_attempted
        delivered = self.rx_delivered
        pdr = delivered / total_sent if total_sent > 0 else 0.0
        avg_delay = (
            self.total_delay / self.delivered_count if self.delivered_count > 0 else 0.0
        )
        sim_time = self.current_time
        throughput_bps = (
            self.packets_delivered * self.payload_size_bytes * 8 / sim_time
            if sim_time > 0
            else 0.0
        )
        pdr_by_node = {node.id: node.pdr for node in self.nodes}
        recent_pdr_by_node = {node.id: node.recent_pdr for node in self.nodes}
        pdr_by_sf: dict[int, float] = {}
        for sf in range(7, 13):
            nodes_sf = [n for n in self.nodes if n.sf == sf]
            sent_sf = sum(n.tx_attempted for n in nodes_sf)
            delivered_sf = sum(n.rx_delivered for n in nodes_sf)
            pdr_by_sf[sf] = delivered_sf / sent_sf if sent_sf > 0 else 0.0

        gateway_counts = {gw.id: 0 for gw in self.gateways}
        for gw_id in self.network_server.event_gateway.values():
            if gw_id in gateway_counts:
                gateway_counts[gw_id] += 1
        pdr_by_gateway = {
            gw_id: count / total_sent if total_sent > 0 else 0.0
            for gw_id, count in gateway_counts.items()
        }

        pdr_by_class: dict[str, float] = {}
        class_types = {n.class_type for n in self.nodes}
        for ct in class_types:
            nodes_cls = [n for n in self.nodes if n.class_type == ct]
            sent_cls = sum(n.tx_attempted for n in nodes_cls)
            delivered_cls = sum(n.rx_delivered for n in nodes_cls)
            pdr_by_class[ct] = delivered_cls / sent_cls if sent_cls > 0 else 0.0

        energy_by_class = {
            ct: sum(n.energy_consumed for n in self.nodes if n.class_type == ct)
            for ct in class_types
        }

        energy_by_node = {n.id: n.energy_consumed for n in self.nodes}
        airtime_by_node = {n.id: n.total_airtime for n in self.nodes}
        energy_by_gateway = {gw.id: gw.energy_consumed for gw in self.gateways}
        energy_breakdown_by_node = {n.id: n.get_energy_breakdown() for n in self.nodes}
        energy_breakdown_by_gateway = {
            gw.id: gw.get_energy_breakdown() for gw in self.gateways
        }

        interval_sum = 0.0
        interval_count = 0
        for n in self.nodes:
            times = sorted(entry["tx_time"] for entry in n.interval_log)
            if self.warm_up_intervals:
                times = times[self.warm_up_intervals :]
            if len(times) > 1:
                for t0, t1 in zip(times, times[1:]):
                    interval_sum += t1 - t0
                    interval_count += 1
            else:
                interval_sum += n.arrival_interval_sum
                interval_count += n.arrival_interval_count

        avg_arrival_interval = (
            interval_sum / interval_count if interval_count > 0 else 0.0
        )

        tx_power_distribution: dict[float, int] = {}
        for node in self.nodes:
            p = node.tx_power
            tx_power_distribution[p] = tx_power_distribution.get(p, 0) + 1

        metrics = {
            "PDR": pdr,
            "tx_attempted": total_sent,
            "delivered": delivered,
            "collisions": self.packets_lost_collision,
            "collisions_snir": getattr(self, "packets_lost_snir", 0),
            "duplicates": self.network_server.duplicate_packets,
            "energy_J": self.total_energy_J,
            "energy_nodes_J": self.energy_nodes_J,
            "energy_gateways_J": self.energy_gateways_J,
            "avg_delay_s": avg_delay,
            "avg_arrival_interval_s": avg_arrival_interval,
            "throughput_bps": throughput_bps,
            "sf_distribution": {
                sf: sum(1 for node in self.nodes if node.sf == sf)
                for sf in range(7, 13)
            },
            "tx_power_distribution": tx_power_distribution,
            "pdr_by_node": pdr_by_node,
            "recent_pdr_by_node": recent_pdr_by_node,
            "pdr_by_sf": pdr_by_sf,
            "pdr_by_gateway": pdr_by_gateway,
            "pdr_by_class": pdr_by_class,
            "energy_by_node": energy_by_node,
            "airtime_by_node": airtime_by_node,
            "energy_by_gateway": energy_by_gateway,
            "energy_breakdown_by_node": energy_breakdown_by_node,
            "energy_breakdown_by_gateway": energy_breakdown_by_gateway,
            **{f"energy_class_{ct}_J": energy_by_class[ct] for ct in energy_by_class},
            "retransmissions": self.retransmissions,
        }

        qos_clusters_config = getattr(self, "qos_clusters_config", {}) or {}
        qos_node_clusters = getattr(self, "qos_node_clusters", {}) or {}
        metrics.setdefault("qos_cluster_throughput_bps", {})
        metrics.setdefault("qos_cluster_pdr", {})
        metrics.setdefault("qos_cluster_targets", {})
        metrics.setdefault("qos_cluster_node_counts", {})
        metrics.setdefault("qos_cluster_pdr_gap", {})
        metrics.setdefault("qos_cluster_sf_channel", {})
        metrics.setdefault("qos_throughput_gini", 0.0)
        if qos_clusters_config:
            payload_bits = float(self.payload_size_bytes) * 8.0
            cluster_node_counts: dict[int, int] = {
                cluster_id: 0 for cluster_id in qos_clusters_config
            }
            cluster_attempts: dict[int, int] = {
                cluster_id: 0 for cluster_id in qos_clusters_config
            }
            cluster_delivered: dict[int, int] = {
                cluster_id: 0 for cluster_id in qos_clusters_config
            }
            cluster_sf_channel: dict[int, dict[int, dict[int, int]]] = {
                cluster_id: {} for cluster_id in qos_clusters_config
            }

            for node in self.nodes:
                node_id = getattr(node, "id", None)
                if node_id is None:
                    continue
                cluster_id = qos_node_clusters.get(node_id)
                if cluster_id is None:
                    cluster_id = getattr(node, "qos_cluster_id", None)
                if cluster_id is None:
                    continue
                if cluster_id not in qos_clusters_config:
                    continue
                cluster_node_counts.setdefault(cluster_id, 0)
                cluster_attempts.setdefault(cluster_id, 0)
                cluster_delivered.setdefault(cluster_id, 0)
                cluster_sf_channel.setdefault(cluster_id, {})
                cluster_node_counts[cluster_id] += 1
                cluster_attempts[cluster_id] += getattr(node, "tx_attempted", 0)
                cluster_delivered[cluster_id] += getattr(node, "rx_delivered", 0)
                sf_value = int(getattr(node, "sf", 0) or 0)
                channel_idx = self.channel_index(getattr(node, "channel", None))
                sf_map = cluster_sf_channel[cluster_id].setdefault(sf_value, {})
                sf_map[channel_idx] = sf_map.get(channel_idx, 0) + 1

            cluster_pdr: dict[int, float] = {}
            cluster_targets: dict[int, float] = {}
            cluster_throughput: dict[int, float] = {}

            for cluster_id, config in qos_clusters_config.items():
                attempts = cluster_attempts.get(cluster_id, 0)
                delivered_cluster = cluster_delivered.get(cluster_id, 0)
                if attempts > 0:
                    cluster_pdr[cluster_id] = delivered_cluster / attempts
                else:
                    cluster_pdr[cluster_id] = 0.0
                target = config.get("pdr_target", 0.0)
                cluster_targets[cluster_id] = float(target) if target is not None else 0.0
                if sim_time > 0.0 and payload_bits > 0.0:
                    throughput_value = (
                        delivered_cluster * payload_bits / sim_time
                    )
                else:
                    throughput_value = 0.0
                cluster_throughput[cluster_id] = throughput_value

            cluster_ids = sorted(qos_clusters_config)
            throughput_values = [cluster_throughput.get(cid, 0.0) for cid in cluster_ids]
            total_throughput = sum(throughput_values)
            gini_index = 0.0
            if cluster_ids and total_throughput > 0.0:
                numerator = 0.0
                for value_i in throughput_values:
                    for value_j in throughput_values:
                        numerator += abs(value_i - value_j)
                gini_index = numerator / (2.0 * len(cluster_ids) * total_throughput)

            metrics["qos_cluster_throughput_bps"] = cluster_throughput
            metrics["qos_cluster_pdr"] = cluster_pdr
            metrics["qos_cluster_targets"] = cluster_targets
            metrics["qos_cluster_node_counts"] = cluster_node_counts
            metrics["qos_cluster_pdr_gap"] = {
                cluster_id: cluster_pdr.get(cluster_id, 0.0) - cluster_targets.get(cluster_id, 0.0)
                for cluster_id in cluster_ids
            }
            metrics["qos_cluster_sf_channel"] = cluster_sf_channel
            metrics["qos_throughput_gini"] = gini_index

        return metrics

    def get_events_dataframe(self) -> "pd.DataFrame | None":
        """
        Retourne un DataFrame pandas contenant le log de tous les événements de
        transmission enrichi des états initiaux et finaux des nœuds.
        """
        if pd is None:
            raise RuntimeError("pandas is required for this feature")
        if not self.events_log:
            return pd.DataFrame()
        df = pd.DataFrame(self.events_log)
        # Construire un dictionnaire id->nœud pour récupérer les états initiaux/finaux
        node_dict = {node.id: node for node in self.nodes}
        # Ajouter colonnes d'état initial et final du nœud pour chaque événement
        df["initial_x"] = df["node_id"].apply(lambda nid: node_dict[nid].initial_x)
        df["initial_y"] = df["node_id"].apply(lambda nid: node_dict[nid].initial_y)
        df["final_x"] = df["node_id"].apply(lambda nid: node_dict[nid].x)
        df["final_y"] = df["node_id"].apply(lambda nid: node_dict[nid].y)
        df["initial_sf"] = df["node_id"].apply(lambda nid: node_dict[nid].initial_sf)
        df["final_sf"] = df["node_id"].apply(lambda nid: node_dict[nid].sf)
        df["initial_tx_power"] = df["node_id"].apply(
            lambda nid: node_dict[nid].initial_tx_power
        )
        df["final_tx_power"] = df["node_id"].apply(lambda nid: node_dict[nid].tx_power)
        df["packets_sent"] = df["node_id"].apply(
            lambda nid: node_dict[nid].packets_sent
        )
        df["packets_success"] = df["node_id"].apply(
            lambda nid: node_dict[nid].packets_success
        )
        df["packets_collision"] = df["node_id"].apply(
            lambda nid: node_dict[nid].packets_collision
        )
        df["packets_collision_snir"] = df["node_id"].apply(
            lambda nid: node_dict[nid].packets_collision_snir
        )
        df["tx_attempted"] = df["node_id"].apply(
            lambda nid: node_dict[nid].tx_attempted
        )
        df["rx_delivered"] = df["node_id"].apply(
            lambda nid: node_dict[nid].rx_delivered
        )
        df["energy_tx_J_node"] = df["node_id"].apply(
            lambda nid: node_dict[nid].energy_tx
        )
        df["energy_rx_J_node"] = df["node_id"].apply(
            lambda nid: node_dict[nid].energy_rx
        )
        df["energy_sleep_J_node"] = df["node_id"].apply(
            lambda nid: node_dict[nid].energy_sleep
        )
        df["energy_processing_J_node"] = df["node_id"].apply(
            lambda nid: node_dict[nid].energy_processing
        )
        df["energy_ramp_J_node"] = df["node_id"].apply(
            lambda nid: node_dict[nid].energy_ramp
        )
        df["energy_consumed_J_node"] = df["node_id"].apply(
            lambda nid: node_dict[nid].energy_consumed
        )
        df["battery_capacity_J"] = df["node_id"].apply(
            lambda nid: node_dict[nid].battery_capacity_j
        )
        df["battery_remaining_J"] = df["node_id"].apply(
            lambda nid: node_dict[nid].battery_remaining_j
        )
        df["downlink_pending"] = df["node_id"].apply(
            lambda nid: node_dict[nid].downlink_pending
        )
        df["acks_received"] = df["node_id"].apply(
            lambda nid: node_dict[nid].acks_received
        )
        # Colonnes d'intérêt dans un ordre lisible
        columns_order = [
            "event_id",
            "node_id",
            "initial_x",
            "initial_y",
            "final_x",
            "final_y",
            "initial_sf",
            "final_sf",
            "initial_tx_power",
            "final_tx_power",
            "packets_sent",
            "packets_success",
            "packets_collision",
            "tx_attempted",
            "rx_delivered",
            "energy_tx_J_node",
            "energy_rx_J_node",
            "energy_sleep_J_node",
            "energy_processing_J_node",
            "energy_ramp_J_node",
            "energy_consumed_J_node",
            "battery_capacity_J",
            "battery_remaining_J",
            "downlink_pending",
            "acks_received",
            "start_time",
            "end_time",
            "energy_J",
            "rssi_dBm",
            "snr_dB",
            "result",
            "gateway_id",
        ]
        for col in columns_order:
            if col not in df.columns:
                df[col] = None
        return df[columns_order]

    def dump_interval_logs(self, dest: str | Path = ".") -> None:
        """Écrit les intervalles théoriques et réels de chaque nœud en Parquet."""
        if not self.dump_intervals:
            return
        if pd is None:
            raise RuntimeError("pandas is required for this feature")
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        for node in self.nodes:
            if not node.interval_log:
                continue
            df = pd.DataFrame(node.interval_log)
            df.to_parquet(dest_path / f"intervals_node_{node.id}.parquet", index=False)
