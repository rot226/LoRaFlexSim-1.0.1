from __future__ import annotations

import logging
from collections import defaultdict
import math

from typing import TYPE_CHECKING
from .downlink_scheduler import DownlinkScheduler
from .join_server import JoinServer  # re-export
from .channel import Channel

__all__ = ["NetworkServer", "JoinServer"]

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .lorawan import LoRaWANFrame, JoinAccept

logger = logging.getLogger(__name__)

# Paramètres ADR (valeurs issues de la spécification LoRaWAN)
REQUIRED_SNR = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
ADR_WINDOW_SIZE = 20
MARGIN_DB = 15.0
# Typical payload size (in bytes) used when balancing airtime for EXPLoRa-AT
EXPLORA_AT_PAYLOAD_SIZE = 20


class NetworkServer:
    """Représente le serveur de réseau LoRa (collecte des paquets reçus)."""

    def __init__(
        self,
        join_server=None,
        *,
        simulator=None,
        process_delay: float = 0.0,
        network_delay: float = 0.0,
        adr_method: str = "max",
        energy_detection_dBm: float = -float("inf"),
        capture_mode: str | None = None,
    ):
        """Initialise le serveur réseau.

        :param join_server: Instance facultative de serveur d'activation OTAA.
        :param simulator: Référence au :class:`Simulator` pour planifier
            éventuellement certains événements (classe C).
        :param process_delay: Délai de traitement du serveur (s).
        :param network_delay: Délai de propagation des messages (s).
        :param adr_method: Méthode d'agrégation du SNR pour l'ADR
            (``"max"`` ou ``"avg"``).
        :param energy_detection_dBm: Seuil de détection d'énergie appliqué aux
            configurations FLoRa (−90 dBm par défaut lorsque fourni par le
            simulateur).
        :param capture_mode: Mode de capture imposé par le simulateur
            (``None`` pour laisser chaque passerelle décider).
        """
        # Ensemble des identifiants d'événements déjà reçus (pour éviter les doublons)
        self.received_events = set()
        # Stockage optionnel d'infos sur les réceptions (par ex : via quelle passerelle)
        self.event_gateway = {}
        self.event_snir: dict[int, float] = {}
        self.event_rssi: dict[int, float] = {}
        # Compteur de paquets reçus
        self.packets_received = 0
        # Nombre de doublons ignorés
        self.duplicate_packets = 0
        # Indicateur ADR serveur
        self.adr_enabled = False
        # Références pour ADR serveur
        self.nodes = []
        self.gateways = []
        self.channel = None
        self.net_id = 0
        self.next_devaddr = 1
        self.scheduler = DownlinkScheduler(link_delay=network_delay)
        if simulator is not None and hasattr(simulator, "_quantize"):
            self.scheduler.quantize = simulator._quantize
        self.join_server = join_server
        self.simulator = simulator
        self.process_delay = process_delay
        self.network_delay = network_delay
        self.adr_method = adr_method
        self.energy_detection_dBm = energy_detection_dBm
        self.capture_mode = capture_mode
        self.pending_process: dict[
            int,
            tuple[int, int, int, float | None, float | None, object, float | None],
        ] = {}
        self.gateway_snr_samples: dict[int, dict[int, float]] = defaultdict(dict)
        self.gateway_rssi_samples: dict[int, dict[int, float]] = defaultdict(dict)
        self.beacon_interval = 128.0
        self.beacon_drift = 0.0
        self.ping_slot_interval = 1.0
        self.ping_slot_offset = 2.0
        self.last_beacon_time: float | None = None
        # Ensure EXPLoRa-AT grouping is only performed once after initial SNRs
        self.explora_at_groups_assigned = False

    def next_beacon_time(self, after_time: float) -> float:
        """Return the next beacon time after ``after_time``."""
        from .lorawan import next_beacon_time

        return next_beacon_time(
            after_time,
            self.beacon_interval,
            last_beacon=self.last_beacon_time,
            drift=self.beacon_drift,
        )

    def notify_beacon(self, time: float) -> None:
        """Record that a beacon was emitted at ``time``."""
        self.last_beacon_time = time

    def _adr_margin_db(self) -> float:
        """Return the ADR device margin configured on the server."""

        return getattr(self, "MARGIN_DB", MARGIN_DB)

    @staticmethod
    def _update_node_snr_history(node, gateway_id: int | None, snr_value: float) -> None:
        """Enregistrer ``snr_value`` pour ``gateway_id`` dans l'historique nœud."""

        if node is None:
            return

        node.snr_history.append((gateway_id, snr_value))
        if len(node.snr_history) > ADR_WINDOW_SIZE:
            old_gateway, old_snr = node.snr_history.pop(0)
            if old_gateway is not None:
                history = node.gateway_snr_history.get(old_gateway)
                if history:
                    try:
                        history.remove(old_snr)
                    except ValueError:
                        pass

    @staticmethod
    def _gateway_snr_statistics(node) -> tuple[dict[int, dict[str, float]], int]:
        """Retourne les statistiques SNIR par passerelle et le total d'échantillons."""

        stats: dict[int, dict[str, float]] = {}
        total = 0
        if node is None:
            return stats, total

        for gateway_id, history in node.gateway_snr_history.items():
            if not history:
                continue
            count = len(history)
            total_snr = float(sum(history))
            stats[gateway_id] = {
                "count": count,
                "total": total_snr,
                "avg": total_snr / count,
                "max": max(history),
            }
            total += count

        # En cas d'incohérence (par exemple après un retrait manuel), on
        # reconstruit à partir de ``node.snr_history`` pour rester robuste.
        if node.snr_history and total != len(node.snr_history):
            stats.clear()
            total = 0
            for gateway_id, snr in node.snr_history:
                if gateway_id is None:
                    continue
                entry = stats.setdefault(
                    gateway_id,
                    {"count": 0, "total": 0.0, "avg": snr, "max": snr},
                )
                entry["count"] += 1
                entry["total"] += snr
                if snr > entry["max"]:
                    entry["max"] = snr
                total += 1
            for entry in stats.values():
                entry["avg"] = entry["total"] / entry["count"]

        return stats, total

    @staticmethod
    def _round_half_away_from_zero(value: float) -> int:
        """Match ``std::round`` semantics used by FLoRa (half away from zero)."""

        if not math.isfinite(value):
            logger.debug(
                "_round_half_away_from_zero reçu une valeur non finie %s, retour 0",
                value,
            )
            return 0

        return int(math.copysign(math.floor(abs(value) + 0.5), value))

    def assign_explora_sf_groups(self) -> None:
        """Assign nodes to spreading factor groups based on last RSSI."""
        nodes = [n for n in self.nodes if getattr(n, "last_rssi", None) is not None]
        if not nodes:
            return
        nodes.sort(key=lambda n: n.last_rssi, reverse=True)
        base, extra = divmod(len(nodes), 6)
        index = 0
        for i, sf in enumerate(range(7, 13)):
            size = base + (1 if i < extra else 0)
            group = nodes[index : index + size]
            for node in group:
                if node.sf != sf:
                    node.sf = sf
                    node.channel.detection_threshold_dBm = (
                        Channel.flora_detection_threshold(
                            node.sf, node.channel.bandwidth
                        )
                        + node.channel.sensitivity_margin_dB
                    )
                    self.send_downlink(
                        node,
                        adr_command=(sf, node.tx_power, node.chmask, node.nb_trans),
                    )
            index += size

    def assign_explora_at_groups(self) -> None:
        """Assign nodes to SF groups balancing total airtime.

        Nodes are first ordered by their last RSSI (strongest to weakest).
        Group sizes are derived so that the aggregate airtime used by each
        spreading factor is roughly identical.  After assigning the target SF
        for each group, the individual SNR of every node is checked and, if the
        margin is insufficient, the node is moved to a more robust SF and the
        transmit power adjusted.  A ``LinkADRReq`` is issued whenever a node
        needs new parameters.
        """

        nodes = [n for n in self.nodes if getattr(n, "last_rssi", None) is not None]
        if not nodes:
            return

        # Sort from highest RSSI (closest) to lowest (furthest)
        nodes.sort(key=lambda n: n.last_rssi, reverse=True)

        # Determine group sizes so that group airtimes are uniform
        airtimes = {
            sf: self.channel.airtime(sf, EXPLORA_AT_PAYLOAD_SIZE) for sf in range(7, 13)
        }
        inv = {sf: 1.0 / t for sf, t in airtimes.items()}
        total = sum(inv.values())
        n_nodes = len(nodes)
        raw = {sf: n_nodes * inv[sf] / total for sf in range(7, 13)}
        sizes = {sf: int(raw[sf]) for sf in range(7, 13)}
        remaining = n_nodes - sum(sizes.values())
        # Distribute remaining nodes to SFs with largest fractional parts,
        # favouring lower SFs when remainders are equal.
        order = sorted(range(7, 13), key=lambda s: (-(raw[s] - sizes[s]), s))
        for sf in order[:remaining]:
            sizes[sf] += 1

        from .lorawan import DBM_TO_TX_POWER_INDEX, TX_POWER_INDEX_TO_DBM

        noise = self.channel.noise_floor_dBm()
        index = 0
        for sf in range(7, 13):
            group = nodes[index : index + sizes.get(sf, 0)]
            for node in group:
                snr = node.last_rssi - noise
                target_sf = sf
                required = REQUIRED_SNR.get(target_sf, -20.0)
                margin = snr - required - self._adr_margin_db()

                p_idx = DBM_TO_TX_POWER_INDEX.get(int(node.tx_power), 0)
                max_idx = max(TX_POWER_INDEX_TO_DBM.keys())

                # Increase power first if the margin is negative
                while margin < 0 and p_idx > 0:
                    p_idx -= 1
                    margin += 3.0

                # If still negative margin, move to a higher SF
                while margin < 0 and target_sf < 12:
                    target_sf += 1
                    required = REQUIRED_SNR.get(target_sf, -20.0)
                    margin = snr - required - self._adr_margin_db()

                # Reduce power if we have excess margin
                while margin >= 3.0 and p_idx < max_idx:
                    p_idx += 1
                    margin -= 3.0

                power = TX_POWER_INDEX_TO_DBM.get(p_idx, node.tx_power)

                if node.sf != target_sf or node.tx_power != power:
                    node.sf = target_sf
                    node.channel.detection_threshold_dBm = (
                        Channel.flora_detection_threshold(
                            node.sf, node.channel.bandwidth
                        )
                        + node.channel.sensitivity_margin_dB
                    )
                    node.tx_power = power
                    self.send_downlink(
                        node,
                        adr_command=(
                            target_sf,
                            power,
                            node.chmask,
                            node.nb_trans,
                        ),
                    )
            index += sizes.get(sf, 0)

    # ------------------------------------------------------------------
    # Downlink management
    # ------------------------------------------------------------------
    def send_downlink(
        self,
        node,
        payload: bytes | LoRaWANFrame | JoinAccept = b"",
        confirmed: bool = False,
        adr_command: tuple | None = None,
        request_ack: bool = False,
        at_time: float | None = None,
        gateway=None,
        *,
        channel=None,
    ):
        """Queue a downlink frame for a node via ``gateway`` or the first one."""
        from .lorawan import (
            LoRaWANFrame,
            LinkADRReq,
            SF_TO_DR,
            DBM_TO_TX_POWER_INDEX,
            JoinAccept,
        )

        gw = gateway or (self.gateways[0] if self.gateways else None)
        if gw is None:
            return
        if (
            self.simulator is not None
            and getattr(self.scheduler, "quantize", None)
            is getattr(self.scheduler, "_identity_quantize", None)
            and hasattr(self.simulator, "_quantize")
        ):
            self.scheduler.quantize = self.simulator._quantize
        fctrl = 0x20 if request_ack else 0
        frame: LoRaWANFrame | JoinAccept
        if isinstance(payload, JoinAccept):
            frame = payload
        elif isinstance(payload, LoRaWANFrame):
            frame = payload
        else:
            raw = payload.to_bytes() if hasattr(payload, "to_bytes") else bytes(payload)
            frame = LoRaWANFrame(
                mhdr=0x60,
                fctrl=fctrl,
                fcnt=node.fcnt_down,
                payload=raw,
                confirmed=confirmed,
            )
        priority = -1 if confirmed or adr_command or request_ack or isinstance(frame, JoinAccept) else 0
        if adr_command and isinstance(frame, LoRaWANFrame):
            if len(adr_command) == 2:
                sf, power = adr_command
                chmask = node.chmask
                nbtrans = node.nb_trans
            else:
                sf, power, chmask, nbtrans = adr_command
            dr = SF_TO_DR.get(sf, 5)
            p_idx = DBM_TO_TX_POWER_INDEX.get(int(power), 0)
            frame.payload = LinkADRReq(dr, p_idx, chmask, nbtrans).to_bytes()
        adr_frames_before: int | None = None
        if adr_command and hasattr(node, "frames_since_last_adr_command"):
            adr_frames_before = node.frames_since_last_adr_command
            node.frames_since_last_adr_command = 0
        if node.security_enabled and isinstance(frame, LoRaWANFrame):
            from .lorawan import encrypt_payload, compute_mic

            enc = encrypt_payload(
                node.appskey, node.devaddr, node.fcnt_down, 1, frame.payload
            )
            frame.encrypted_payload = enc
            frame.mic = compute_mic(node.nwkskey, node.devaddr, node.fcnt_down, 1, enc)
        frame_counter_start = node.fcnt_down
        node.fcnt_down += 1
        scheduled_time: float | None = None
        previous_pending = getattr(node, "downlink_pending", 0)
        scheduled = False

        def _schedule(method, *args, **kwargs):
            if channel is None:
                return method(*args, **kwargs)
            return method(*args, channel=channel, **kwargs)

        if at_time is None:
            if node.class_type.upper() == "B":
                after = self.simulator.current_time if self.simulator else 0.0
                beacon_reference = getattr(node, "last_beacon_time", None)
                if beacon_reference is not None:
                    beacon_reference += getattr(node, "clock_offset", 0.0)
                scheduled_time = _schedule(
                    self.scheduler.schedule_class_b,
                    node,
                    after,
                    frame,
                    gw,
                    self.beacon_interval,
                    self.ping_slot_interval,
                    self.ping_slot_offset,
                    last_beacon_time=beacon_reference,
                    priority=priority,
                )
                scheduled = True
            elif node.class_type.upper() == "C":
                after = self.simulator.current_time if self.simulator else 0.0
                scheduled_time = _schedule(
                    self.scheduler.schedule_class_c,
                    node,
                    after,
                    frame,
                    gw,
                    priority=priority,
                )
                scheduled = scheduled_time is not None
            else:
                end = getattr(node, "last_uplink_end_time", None)
                if end is not None:
                    from .lorawan import compute_rx1, compute_rx2

                    after = self.simulator.current_time if self.simulator else 0.0
                    rx1 = compute_rx1(end, node.rx_delay)
                    rx2 = compute_rx2(end, node.rx_delay)
                    scheduled_time = _schedule(
                        self.scheduler.schedule_class_a,
                        node,
                        after,
                        rx1,
                        rx2,
                        frame,
                        gw,
                        priority=priority,
                    )
                    scheduled = scheduled_time is not None
                else:
                    gw.buffer_downlink(node.id, frame, channel=channel)
                    scheduled = True
        else:
            if node.class_type.upper() == "B":
                scheduled_time = _schedule(
                    self.scheduler.schedule_class_b,
                    node,
                    at_time,
                    frame,
                    gw,
                    self.beacon_interval,
                    self.ping_slot_interval,
                    self.ping_slot_offset,
                    last_beacon_time=getattr(node, "last_beacon_time", None),
                    priority=priority,
                )
                scheduled = True
            elif node.class_type.upper() == "C":
                scheduled_time = _schedule(
                    self.scheduler.schedule_class_c,
                    node,
                    at_time,
                    frame,
                    gw,
                    priority=priority,
                )
                scheduled = scheduled_time is not None
            else:
                _schedule(
                    self.scheduler.schedule,
                    node.id,
                    at_time,
                    frame,
                    gw,
                    priority=priority,
                )
                scheduled_time = at_time
                scheduled = True
        if not scheduled:
            node.fcnt_down = frame_counter_start
            if adr_frames_before is not None:
                node.frames_since_last_adr_command = adr_frames_before
            logger.debug(
                "Downlink non planifié pour le nœud %s (classe %s)",
                node.id,
                node.class_type,
            )
            return
        try:
            node.downlink_pending += 1
        except AttributeError:
            pass
        else:
            if (
                self.simulator is not None
                and node.class_type.upper() == "C"
                and hasattr(self.simulator, "_ensure_class_c_polling")
            ):
                trigger_time: float | None = None
                if at_time is not None:
                    trigger_time = at_time
                elif previous_pending == 0 and node.downlink_pending > 0:
                    current_time = self.simulator.current_time
                    if scheduled_time is None and hasattr(self.scheduler, "next_time"):
                        try:
                            scheduled_time = self.scheduler.next_time(node.id)
                        except Exception:
                            scheduled_time = None
                    if scheduled_time is not None:
                        trigger_time = min(current_time, scheduled_time)
                    else:
                        trigger_time = current_time
                if trigger_time is not None:
                    self.simulator._ensure_class_c_polling(node.id, trigger_time)

    def _derive_keys(
        self, appkey: bytes, devnonce: int, appnonce: int
    ) -> tuple[bytes, bytes]:
        from .lorawan import derive_session_keys

        return derive_session_keys(appkey, devnonce, appnonce, self.net_id)

    # ------------------------------------------------------------------
    # Event scheduling helpers
    # ------------------------------------------------------------------
    def schedule_receive(
        self,
        event_id: int,
        node_id: int,
        gateway_id: int,
        rssi: float | None = None,
        frame=None,
        at_time: float | None = None,
        *,
        snir: float | None = None,
    ) -> None:
        """Planifie le traitement serveur d'une trame."""
        if self.simulator is None:
            self.receive(
                event_id,
                node_id,
                gateway_id,
                rssi,
                frame,
                end_time=at_time,
                snir=snir,
            )
            return

        from .simulator import EventType

        arrival_time = (
            (at_time if at_time is not None else self.simulator.current_time)
            + self.network_delay
        )

        if arrival_time <= self.simulator.current_time and self.process_delay <= 0:
            self.receive(event_id, node_id, gateway_id, rssi, frame, snir=snir)
            return

        eid = self.simulator.event_id_counter
        self.simulator.event_id_counter += 1
        self.pending_process[eid] = (
            event_id,
            node_id,
            gateway_id,
            rssi,
            snir,
            frame,
            at_time,
        )
        if hasattr(self.simulator, "_push_event"):
            self.simulator._push_event(arrival_time, EventType.SERVER_RX, eid, node_id)
        else:
            from .simulator import Event
            import heapq

            heapq.heappush(
                self.simulator.event_queue,
                Event(arrival_time, EventType.SERVER_RX, eid, node_id),
            )

    def _handle_network_arrival(self, eid: int) -> None:
        """Planifie le traitement d'un paquet arrivé au serveur."""
        info = self.pending_process.pop(eid, None)
        if not info:
            return
        from .simulator import EventType

        process_time = self.simulator.current_time + self.process_delay
        new_id = self.simulator.event_id_counter
        self.simulator.event_id_counter += 1
        self.pending_process[new_id] = info
        node_id = info[1]
        if hasattr(self.simulator, "_push_event"):
            self.simulator._push_event(process_time, EventType.SERVER_PROCESS, new_id, node_id)
        else:
            from .simulator import Event
            import heapq

            heapq.heappush(
                self.simulator.event_queue,
                Event(process_time, EventType.SERVER_PROCESS, new_id, node_id),
            )

    def _process_scheduled(self, eid: int) -> None:
        """Exécute le traitement différé d'un paquet."""
        info = self.pending_process.pop(eid, None)
        if not info:
            return
        event_id, node_id, gateway_id, rssi, snir, frame, end_time = info
        self.receive(
            event_id,
            node_id,
            gateway_id,
            rssi,
            frame,
            end_time=end_time,
            snir=snir,
        )

    def deliver_scheduled(self, node_id: int, current_time: float) -> None:
        """Move ready scheduled frames to the gateway buffer."""
        tolerance = 0.1
        nxt = self.scheduler.next_time(node_id)
        if nxt is not None and nxt < current_time - tolerance:
            entry = self.scheduler.pop_ready(node_id, nxt)
            if entry:
                entry.gateway.buffer_downlink(
                    node_id,
                    entry.frame,
                    data_rate=entry.data_rate,
                    tx_power=entry.tx_power,
                    channel=entry.channel,
                )
        entry = self.scheduler.pop_ready(node_id, current_time)
        while entry:
            entry.gateway.buffer_downlink(
                node_id,
                entry.frame,
                data_rate=entry.data_rate,
                tx_power=entry.tx_power,
                channel=entry.channel,
            )
            entry = self.scheduler.pop_ready(node_id, current_time)

    def _apply_best_gateway_selection(
        self,
        event_id: int,
        node,
        gateway_id: int | None,
        snr: float | None,
        rssi: float | None,
    ) -> None:
        """Record the best gateway/SNR pair for ``event_id`` and update history."""

        prev_gateway = self.event_gateway.get(event_id)
        prev_snr = self.event_snir.get(event_id)

        replace_previous = (
            node is not None
            and prev_gateway is not None
            and prev_snr is not None
            and gateway_id is not None
            and snr is not None
            and (prev_gateway != gateway_id or prev_snr != snr)
        )

        if replace_previous:
            history = node.gateway_snr_history.get(prev_gateway)
            if history:
                try:
                    history.remove(prev_snr)
                except ValueError:
                    pass

        if gateway_id is not None:
            self.event_gateway[event_id] = gateway_id

        if snr is not None:
            self.event_snir[event_id] = snr
            if node is not None:
                node.last_snr = snr

        if rssi is not None:
            self.event_rssi[event_id] = rssi
            if node is not None:
                node.last_rssi = rssi

        if (
            node is not None
            and gateway_id is not None
            and snr is not None
            and (prev_gateway != gateway_id or prev_snr != snr or prev_gateway is None)
        ):
            history = node.gateway_snr_history.setdefault(gateway_id, [])
            history.append(snr)
            if len(history) > ADR_WINDOW_SIZE:
                history.pop(0)

    def _activate(self, node, gateway=None):
        from .lorawan import JoinAccept, encrypt_join_accept

        appnonce = self.next_devaddr & 0xFFFFFF
        devaddr = self.next_devaddr
        self.next_devaddr += 1
        devnonce = (node.devnonce - 1) & 0xFFFF
        nwk_skey, app_skey = self._derive_keys(node.appkey, devnonce, appnonce)
        # Store derived keys server-side but send only join parameters
        frame = JoinAccept(appnonce, self.net_id, devaddr)
        if node.security_enabled:
            enc, mic = encrypt_join_accept(node.appkey, frame)
            frame.encrypted = enc
            frame.mic = mic
        node.nwkskey = nwk_skey
        node.appskey = app_skey
        self.send_downlink(node, frame, gateway=gateway)

    def receive(
        self,
        event_id: int,
        node_id: int,
        gateway_id: int,
        rssi: float | None = None,
        frame=None,
        end_time: float | None = None,
        *,
        snir: float | None = None,
    ):
        """
        Traite la réception d'un paquet par le serveur.
        Évite de compter deux fois le même paquet s'il arrive via plusieurs passerelles.
        :param event_id: Identifiant unique de l'événement de transmission du paquet.
        :param node_id: Identifiant du nœud source.
        :param gateway_id: Identifiant de la passerelle ayant reçu le paquet.
        :param rssi: RSSI mesuré par la passerelle pour ce paquet (optionnel).
        :param frame: Trame LoRaWAN associée pour vérification de sécurité
            (optionnelle).
        """
        node = next((n for n in self.nodes if n.id == node_id), None)
        gw = next((g for g in self.gateways if g.id == gateway_id), None)

        snr_value = None
        noise_floor = None
        if snir is not None:
            snr_value = snir
        else:
            if node is not None and getattr(node, "channel", None) is not None:
                noise_floor = node.channel.noise_floor_dBm()
            elif self.channel is not None:
                noise_floor = self.channel.noise_floor_dBm()
            if noise_floor is not None and rssi is not None:
                snr_value = rssi - noise_floor

        if snr_value is not None:
            self.gateway_snr_samples[gateway_id][event_id] = snr_value
        if rssi is not None:
            self.gateway_rssi_samples[gateway_id][event_id] = rssi

        samples = [
            (gw_id, gw_samples[event_id])
            for gw_id, gw_samples in self.gateway_snr_samples.items()
            if event_id in gw_samples
        ]
        best_gateway_id = None
        best_snr = None
        if samples:
            best_gateway_id, best_snr = max(samples, key=lambda item: item[1])

        already_processed = event_id in self.received_events
        previous_gateway_id = self.event_gateway.get(event_id)
        previous_snr = self.event_snir.get(event_id)
        previous_rssi = self.event_rssi.get(event_id)

        selected_gateway_id = (
            best_gateway_id
            if best_gateway_id is not None
            else previous_gateway_id if previous_gateway_id is not None else gateway_id
        )
        selected_snr = (
            best_snr
            if best_snr is not None
            else previous_snr if previous_snr is not None else snr_value
        )
        best_rssi = None
        if selected_gateway_id is not None:
            best_rssi = self.gateway_rssi_samples.get(selected_gateway_id, {}).get(event_id)
        selected_rssi = (
            best_rssi
            if best_rssi is not None
            else previous_rssi if previous_rssi is not None else rssi
        )

        selection_changed = (
            selected_gateway_id != previous_gateway_id
            or (selected_snr is not None and selected_snr != previous_snr)
            or (selected_snr is None and previous_snr is not None)
        )

        self._apply_best_gateway_selection(
            event_id,
            node,
            selected_gateway_id,
            selected_snr,
            selected_rssi,
        )

        gw = next((g for g in self.gateways if g.id == selected_gateway_id), None)
        if gw is None:
            gw = next((g for g in self.gateways if g.id == gateway_id), None)

        snr_value = selected_snr
        rssi = selected_rssi

        if already_processed:
            # Doublon (déjà reçu via une autre passerelle)
            if selection_changed:
                if (
                    node is not None
                    and previous_snr is not None
                    and selected_snr is not None
                ):
                    for idx in range(len(node.snr_history) - 1, -1, -1):
                        gw_id, snr_sample = node.snr_history[idx]
                        if gw_id == previous_gateway_id and snr_sample == previous_snr:
                            del node.snr_history[idx]
                            break
                logger.debug(
                    "NetworkServer: duplicate packet event %s from node %s updated to gateway %s.",
                    event_id,
                    node_id,
                    selected_gateway_id,
                )
            else:
                logger.debug(
                    "NetworkServer: duplicate packet event %s from node %s (ignored).",
                    event_id,
                    node_id,
                )
            self.duplicate_packets += 1
            if not selection_changed:
                return
        else:
            # Nouveau paquet reçu
            self.received_events.add(event_id)
            self.packets_received += 1
            logger.debug(
                "NetworkServer: packet event %s from node %s received via gateway %s.",
                event_id,
                node_id,
                selected_gateway_id,
            )

        if node is not None and not already_processed:
            if hasattr(node, "frames_since_last_adr_command"):
                node.frames_since_last_adr_command += 1

        if node is not None:
            node.last_uplink_end_time = end_time
            if rssi is not None:
                node.last_rssi = rssi
            if snr_value is not None:
                node.last_snr = snr_value
        from .lorawan import JoinRequest

        if node and isinstance(frame, JoinRequest) and self.join_server:
            try:
                accept, nwk_skey, app_skey = self.join_server.handle_join(frame)
            except Exception:
                return
            node.nwkskey = nwk_skey
            node.appskey = app_skey
            node.devaddr = accept.dev_addr
            node.activated = True
            self.send_downlink(node, accept, gateway=gw)
            return

        if node and frame is not None and node.security_enabled:
            from .lorawan import validate_frame, LoRaWANFrame

            if isinstance(frame, LoRaWANFrame) and not validate_frame(
                frame,
                node.nwkskey,
                node.appskey,
                node.devaddr,
                0,
            ):
                return

        if node and not getattr(node, "activated", True):
            self._activate(node, gateway=gw)

        adr_ack_req = bool(node and getattr(node, "last_adr_ack_req", False))

        if node and node.last_adr_ack_req:
            # Device requested an ADR acknowledgement -> reply with current parameters
            self.send_downlink(node, adr_command=(node.sf, node.tx_power))
            node.last_adr_ack_req = False

        if self.simulator is not None and hasattr(self.simulator, "_events_log_map"):
            entry = self.simulator._events_log_map.get(event_id)
            if entry is not None:
                if rssi is not None:
                    entry["rssi_dBm"] = rssi
                if snr_value is not None:
                    entry["snr_dB"] = snr_value

        # Appliquer ADR complet au niveau serveur
        if self.adr_enabled and snr_value is not None and node is not None:
            if self.adr_method == "explora-sf":
                if all(getattr(n, "last_rssi", None) is not None for n in self.nodes):
                    self.assign_explora_sf_groups()
            elif self.adr_method == "explora-at":
                if (
                    not self.explora_at_groups_assigned
                    and all(getattr(n, "last_rssi", None) is not None for n in self.nodes)
                ):
                    self.assign_explora_at_groups()
                    self.explora_at_groups_assigned = True
            elif self.adr_method == "adr-max":
                from .lorawan import (
                    DBM_TO_TX_POWER_INDEX,
                    TX_POWER_INDEX_TO_DBM,
                )

                if selected_gateway_id is not None:
                    self._update_node_snr_history(node, selected_gateway_id, snr_value)
                if len(node.snr_history) >= ADR_WINDOW_SIZE:
                    stats, total = self._gateway_snr_statistics(node)
                    if not stats or total <= 0:
                        return
                    snr_max = max(entry["max"] for entry in stats.values())
                    required = REQUIRED_SNR.get(node.sf, -20.0)
                    margin = snr_max - required - self._adr_margin_db()
                    nstep = self._round_half_away_from_zero(margin / 3.0)

                    sf = node.sf
                    p_idx = DBM_TO_TX_POWER_INDEX.get(int(node.tx_power), 0)
                    power_order = sorted(
                        TX_POWER_INDEX_TO_DBM.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    idx_to_rank = {
                        idx: rank for rank, (idx, _) in enumerate(power_order)
                    }
                    rank_to_idx = {
                        rank: idx for idx, rank in idx_to_rank.items()
                    }
                    power_rank = idx_to_rank.get(p_idx, 0)
                    max_power_rank = len(power_order) - 1

                    if nstep > 0:
                        while nstep > 0 and sf > 7:
                            sf -= 1
                            nstep -= 1
                        while nstep > 0 and power_rank < max_power_rank:
                            power_rank += 1
                            nstep -= 1
                    elif nstep < 0:
                        while nstep < 0 and power_rank > 0:
                            power_rank -= 1
                            nstep += 1
                        while nstep < 0 and sf < 12:
                            sf += 1
                            nstep += 1

                    p_idx = rank_to_idx.get(power_rank, p_idx)
                    power = TX_POWER_INDEX_TO_DBM.get(p_idx, node.tx_power)

                    if sf != node.sf or power != node.tx_power:
                        node.sf = sf
                        node.channel.detection_threshold_dBm = (
                            Channel.flora_detection_threshold(
                                node.sf, node.channel.bandwidth
                            )
                            + node.channel.sensitivity_margin_dB
                        )
                        node.tx_power = power
                        self.send_downlink(
                            node,
                            adr_command=(
                                sf,
                                power,
                                node.chmask,
                                node.nb_trans,
                            ),
                            gateway=gw,
                        )
            elif self.adr_method == "adr-lite":
                optimal_sf = 12
                for sf in range(7, 13):
                    required = REQUIRED_SNR.get(sf, -20.0) + self._adr_margin_db()
                    if snr_value >= required:
                        optimal_sf = sf
                        break
                if optimal_sf != node.sf:
                    node.sf = optimal_sf
                    node.channel.detection_threshold_dBm = (
                        Channel.flora_detection_threshold(
                            node.sf, node.channel.bandwidth
                        )
                        + node.channel.sensitivity_margin_dB
                    )
                    self.send_downlink(
                        node,
                        adr_command=(
                            optimal_sf,
                            node.tx_power,
                            node.chmask,
                            node.nb_trans,
                        ),
                    )
            else:
                from .lorawan import (
                    DBM_TO_TX_POWER_INDEX,
                    TX_POWER_INDEX_TO_DBM,
                )

                if selected_gateway_id is not None:
                    self._update_node_snr_history(node, selected_gateway_id, snr_value)
                if len(node.snr_history) < ADR_WINDOW_SIZE:
                    return

                if (
                    getattr(node, "frames_since_last_adr_command", 0) < ADR_WINDOW_SIZE
                    and not adr_ack_req
                ):
                    return

                stats, total = self._gateway_snr_statistics(node)
                if not stats or total <= 0:
                    return

                if self.adr_method == "avg":
                    snr_m = sum(entry["total"] for entry in stats.values()) / total
                else:
                    snr_m = max(entry["max"] for entry in stats.values())
                required = REQUIRED_SNR.get(node.sf, -20.0)
                margin = snr_m - required - self._adr_margin_db()
                nstep = self._round_half_away_from_zero(margin / 3.0)

                sf = node.sf
                p_idx = DBM_TO_TX_POWER_INDEX.get(int(node.tx_power), 0)
                max_power_index = max(TX_POWER_INDEX_TO_DBM.keys())

                if nstep > 0:
                    while nstep > 0 and sf > 7:
                        sf -= 1
                        nstep -= 1
                    while nstep > 0 and p_idx < max_power_index:
                        p_idx += 1
                        nstep -= 1
                elif nstep < 0:
                    while nstep < 0 and p_idx > 0:
                        p_idx -= 1
                        nstep += 1
                    while nstep < 0 and sf < 12:
                        sf += 1
                        nstep += 1

                power = TX_POWER_INDEX_TO_DBM.get(p_idx, node.tx_power)

                if sf != node.sf or power != node.tx_power:
                    node.sf = sf
                    node.channel.detection_threshold_dBm = Channel.flora_detection_threshold(
                        node.sf, node.channel.bandwidth
                    ) + node.channel.sensitivity_margin_dB
                    node.tx_power = power
                    self.send_downlink(
                        node,
                        adr_command=(sf, power, node.chmask, node.nb_trans),
                        gateway=gw,
                    )
