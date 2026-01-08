import logging
import math
from traffic.numpy_compat import create_generator

from .non_orth_delta import (
    DEFAULT_NON_ORTH_DELTA as FLORA_NON_ORTH_DELTA,
    load_non_orth_delta,
)
from .energy_profiles import EnergyProfile, FLORA_PROFILE, EnergyAccumulator

# Default energy profile for gateways (same as nodes by default)
DEFAULT_ENERGY_PROFILE = FLORA_PROFILE

logger = logging.getLogger(__name__)
diag_logger = logging.getLogger("diagnostics")

class Gateway:
    """Représente une passerelle LoRa recevant les paquets des nœuds."""

    def __init__(
        self,
        gateway_id: int,
        x: float,
        y: float,
        altitude: float = 0.0,
        *,
        rx_gain_dB: float = 0.0,
        orientation_az: float = 0.0,
        orientation_el: float = 0.0,
        energy_profile: EnergyProfile | str | None = None,
        downlink_power_dBm: float | None = None,
        energy_detection_dBm: float = -float("inf"),
        rng=None,
    ):
        """
        Initialise une passerelle LoRa.

        ``rx_gain_dB`` correspond au gain d'antenne supplémentaire appliqué
        au signal reçu.

        :param gateway_id: Identifiant de la passerelle.
        :param x: Position X (mètres).
        :param y: Position Y (mètres).
        :param altitude: Altitude de l'antenne (mètres).
        :param rx_gain_dB: Gain d’antenne additionnel (dB).
        :param energy_profile: Profil énergétique utilisé pour la consommation.
        :param energy_detection_dBm: Seuil minimal de détection d'énergie.
        """
        self.id = gateway_id
        self.x = x
        self.y = y
        self.altitude = altitude
        self.rx_gain_dB = rx_gain_dB
        self.orientation_az = orientation_az
        self.orientation_el = orientation_el
        self.downlink_power_dBm = downlink_power_dBm
        if isinstance(energy_profile, str):
            from .energy_profiles import get_profile

            self.profile = get_profile(energy_profile)
        else:
            self.profile = (
                DEFAULT_ENERGY_PROFILE if energy_profile is None else energy_profile
            )
        self.energy_consumed = 0.0
        self.energy_tx = 0.0
        self.energy_rx = 0.0
        self.energy_listen = 0.0
        self.energy_preamble = 0.0
        self.energy_sleep = 0.0
        self.energy_processing = 0.0
        self.energy_ramp = 0.0
        self.energy_detection_dBm = energy_detection_dBm
        self.rng = rng or create_generator()
        # Accumulateur d'énergie par état
        self.energy = EnergyAccumulator()
        # Transmissions en cours indexées par (sf, frequency)
        self.active_map: dict[tuple[int, float], list[dict]] = {}
        # Mapping event_id -> (key, dict) for quick removal
        self.active_by_event: dict[int, tuple[tuple[int, float], dict]] = {}
        # Downlink frames waiting for the corresponding node receive windows
        self.downlink_buffer: dict[int, list] = {}

    def select_downlink_power(self, node=None) -> float:
        """Return the TX power (dBm) to use for downlinks to ``node``."""

        if self.downlink_power_dBm is not None:
            return self.downlink_power_dBm
        if node is not None and getattr(node, "tx_power", None) is not None:
            return float(node.tx_power)
        return 14.0

    def add_energy(self, energy_joules: float, state: str = "tx") -> None:
        """Ajoute de l'énergie consommée par la passerelle."""
        self.energy.add(state, energy_joules)
        self.energy_consumed += energy_joules
        if state == "tx":
            self.energy_tx += energy_joules
        elif state == "rx":
            self.energy_rx += energy_joules
        elif state == "listen":
            self.energy_listen += energy_joules
        elif state == "preamble":
            self.energy_preamble += energy_joules
        elif state == "sleep":
            self.energy_sleep += energy_joules
        elif state == "processing":
            self.energy_processing += energy_joules
        elif state == "ramp":
            self.energy_ramp += energy_joules

    def get_energy_breakdown(self) -> dict[str, float]:
        """Retourne la consommation d'énergie par état."""
        return self.energy.to_dict()

    def start_reception(
        self,
        event_id: int,
        node_id: int,
        sf: int,
        rssi: float,
        end_time: float,
        capture_threshold: float,
        current_time: float,
        frequency: float,
        min_interference_time: float = 0.0,
        *,
        freq_offset: float = 0.0,
        sync_offset: float = 0.0,
        bandwidth: float = 125e3,
        noise_floor: float | None = None,
        snir: float | None = None,
        capture_mode: str = "basic",
        flora_phy=None,
        orthogonal_sf: bool = True,
        capture_window_symbols: int = 5,
        non_orth_delta: list[list[float]] | None = None,
        snir_fading_std: float = 0.0,
        marginal_snir_db: float = 0.0,
        marginal_drop_prob: float = 0.0,
        residual_collision_prob: float = 0.0,
        residual_collision_load_scale: float = 1.0,
        baseline_loss_rate: float = 0.0,
        baseline_collision_rate: float = 0.0,
        use_snir: bool = True,
        snir_off_noise_prob: float = 0.0,
    ):
        """
        Tente de démarrer la réception d'une nouvelle transmission sur cette passerelle.
        Gère les collisions et le capture effect.
        :param event_id: Identifiant de l'événement de transmission du nœud.
        :param node_id: Identifiant du nœud émetteur.
        :param sf: Spreading Factor de la transmission.
        :param rssi: Puissance du signal reçu (RSSI) en dBm.
        :param end_time: Temps (simulation) auquel la transmission se termine.
        :param capture_threshold: Seuil de capture en dB pour considérer qu'un signal plus fort peut être décodé malgré les interférences.
        :param frequency: Fréquence radio de la transmission (Hz).
        :param min_interference_time: Durée d'interférence tolérée (s). Les
            transmissions qui ne se chevauchent pas plus longtemps que cette
            valeur ne sont pas considérées comme en collision.
        :param noise_floor: Niveau de bruit pour le calcul du SNR (mode avancé).
        :param snir: SNIR déjà calculé pour ce paquet (optionnel).
        :param capture_mode: "basic" pour l'ancien comportement, "advanced" pour
            un calcul basé sur le SNR.
        :param flora_phy: Instance ``FloraPHY`` lorsque ``capture_mode`` vaut
            "flora".
        :param orthogonal_sf: Si ``True``, les transmissions de SF différents
            sont ignorées pour la détection de collision.
        :param capture_window_symbols: Nombre de symboles de préambule exigés
            avant qu'un paquet puisse capturer la réception.
        :param non_orth_delta: Matrice 6×6 des seuils de capture utilisés lorsque
            ``orthogonal_sf`` vaut ``False``. Chaque cellule représente le
            ΔRSSI (dB) minimal entre ``SF_signal`` (ligne) et
            ``SF_interférence`` (colonne) pour autoriser la capture.
        :param snir_fading_std: Écart-type (dB) appliqué aux RSSI pour refléter
            un fading rapide lors du calcul de capture SNIR.
        :param marginal_snir_db: Marge en-dessous de laquelle un paquet capturé
            peut encore échouer aléatoirement malgré un SNIR légèrement au-dessus
            du seuil.
        :param marginal_drop_prob: Probabilité maximale de déclencher cette
            perte marginale lorsque ``margin=0``.
        :param residual_collision_prob: Probabilité maximale d'une collision
            résiduelle lorsque la charge atteint ``residual_collision_load_scale``.
        :param residual_collision_load_scale: Nombre de transmissions
            concurrentes servant de référence pour saturer la collision résiduelle.
        :param baseline_loss_rate: Perte résiduelle minimale appliquée même sans
            collision explicite, et qui croît avec la charge.
        :param baseline_collision_rate: Perte résiduelle additionnelle qui
            augmente avec la charge (jusqu'à ``residual_collision_load_scale``).
        :param use_snir: Indique si le calcul SNIR est actif sur ce canal.
        :param snir_off_noise_prob: Probabilité minimale de perte aléatoire
            lorsque ``use_snir`` est désactivé.
        """
        if rssi < getattr(self, "energy_detection_dBm", -float("inf")):
            logger.debug(
                "Gateway %s: ignore un paquet sous le seuil d'énergie (RSSI=%.1f dBm)",
                self.id,
                rssi,
            )
            return

        if callable(capture_threshold):
            threshold_fn = lambda sf_val: float(capture_threshold(sf_val))
        elif isinstance(capture_threshold, dict):
            threshold_fn = lambda sf_val: float(
                capture_threshold.get(sf_val, capture_threshold.get("default", 0.0))
            )
        else:
            threshold_fn = lambda _sf: float(capture_threshold)

        key = (sf, frequency)
        symbol_duration = (2 ** sf) / bandwidth
        # Gather all active transmissions that share the same frequency. When
        # ``orthogonal_sf`` is ``True`` we only consider the same SF. Otherwise
        # we must look across all SFs on this frequency.
        if orthogonal_sf:
            candidates = self.active_map.get(key, [])
        else:
            candidates = []
            for (sf_k, freq_k), txs in self.active_map.items():
                if freq_k == frequency:
                    candidates.extend(txs)
        concurrent_transmissions = [t for t in candidates if t['end_time'] > current_time]

        # Filtrer les transmissions dont le chevauchement est significatif
        interfering_transmissions = []
        for t in concurrent_transmissions:
            if orthogonal_sf and t.get('sf') != sf:
                continue
            overlap = min(t['end_time'], end_time) - current_time
            if overlap > min_interference_time:
                interfering_transmissions.append(t)

        # Liste des transmissions en collision potentielles (y compris la nouvelle)
        colliders = interfering_transmissions.copy()
        if noise_floor is not None:
            for t in interfering_transmissions:
                if t.get('snir') is None:
                    t['snir'] = t['rssi'] - noise_floor
        # Ajouter la nouvelle transmission elle-même
        new_transmission = {
            'event_id': event_id,
            'node_id': node_id,
            'sf': sf,
            'frequency': frequency,
            'rssi': rssi,
            'end_time': end_time,
            'start_time': current_time,
            'freq_offset': freq_offset,
            'sync_offset': sync_offset,
            'bandwidth': bandwidth,
            'symbol_duration': symbol_duration,
            'lost_flag': False,
            'snir': snir,
            'collision_reason': None,
        }
        if new_transmission['snir'] is None and noise_floor is not None:
            new_transmission['snir'] = rssi - noise_floor
        colliders.append(new_transmission)

        if capture_mode == "aloha" and interfering_transmissions:
            logger.debug(
                "Gateway %s: pure ALOHA collision detected with packets %s and new event %s.",
                self.id,
                [t["event_id"] for t in interfering_transmissions],
                event_id,
            )
            diag_logger.info(
                f"t={current_time:.2f} gw={self.id} aloha_collision="
                f"{[t['event_id'] for t in interfering_transmissions] + [event_id]}"
            )
            for t in interfering_transmissions:
                t["lost_flag"] = True
            new_transmission["lost_flag"] = True
            self.active_map.setdefault(key, []).append(new_transmission)
            self.active_by_event[event_id] = (key, new_transmission)
            return

        def _load_factor(count: int) -> float:
            scale = max(residual_collision_load_scale, 1.0)
            return min(count / scale, 1.0)

        def _baseline_drop_prob(count: int) -> float:
            load_factor = _load_factor(count)
            base = max(baseline_loss_rate, 0.0)
            min_loss = base * (1.0 + load_factor)
            extra = max(baseline_collision_rate, 0.0) * load_factor
            return min_loss + extra

        if not interfering_transmissions:
            # Aucun paquet actif (ou chevauchement inférieur au seuil)
            drop_prob = _baseline_drop_prob(len(concurrent_transmissions))
            if drop_prob > 0.0 and self.rng.random() < drop_prob:
                new_transmission["lost_flag"] = True
                new_transmission["collision_reason"] = "baseline_loss"
            self.active_map.setdefault(key, []).append(new_transmission)
            self.active_by_event[event_id] = (key, new_transmission)
            logger.debug(
                f"Gateway {self.id}: new transmission {event_id} from node {node_id} "
                f"(SF{sf}, {frequency/1e6:.3f} MHz) started, RSSI={rssi:.1f} dBm."
            )
            return

        # Sinon, on a une collision potentielle: déterminer le capture effect
        # Tri décroissant selon la puissance ou le SNR
        def _penalty(tx1, tx2):
            freq_diff = tx1.get('freq_offset', 0.0) - tx2.get('freq_offset', 0.0)
            time_diff = (tx1.get('start_time', 0.0) + tx1.get('sync_offset', 0.0)) - (
                tx2.get('start_time', 0.0) + tx2.get('sync_offset', 0.0)
            )
            bw = tx1.get('bandwidth', bandwidth)
            freq_factor = abs(freq_diff) / (bw / 2.0)
            symbol_time = (2 ** tx1.get('sf', sf)) / bw
            time_factor = abs(time_diff) / symbol_time
            if freq_factor >= 1.0 and time_factor >= 1.0:
                return float('inf')
            return 10 * math.log10(1.0 + freq_factor ** 2 + time_factor ** 2)

        def _enough_preamble(winner, others) -> bool:
            """Return ``True`` if ``winner`` collects enough clean preamble."""

            sym_time = winner.get('symbol_duration')
            if sym_time is None:
                bw_w = winner.get('bandwidth', bandwidth)
                sf_w = winner.get('sf', sf)
                sym_time = (2 ** sf_w) / bw_w

            default_preamble = capture_window_symbols + 6 if capture_window_symbols is not None else 8
            n_preamble = winner.get('preamble_symbols', default_preamble)
            try:
                n_preamble_val = float(n_preamble)
            except (TypeError, ValueError):
                n_preamble_val = float(default_preamble)
            n_preamble_val = max(n_preamble_val, 0.0)
            cs_begin = winner['start_time'] + sym_time * max(n_preamble_val - 6.0, 0.0)

            for other in others:
                if other is winner:
                    continue

                overlap_start = max(winner['start_time'], other.get('start_time', -float('inf')))
                overlap_end = min(winner['end_time'], other.get('end_time', float('inf')))
                if overlap_end <= overlap_start:
                    continue
                if overlap_start < cs_begin and overlap_end > cs_begin:
                    return False

            return True

        flora_mode = False
        if snir_fading_std > 0.0:
            for t in colliders:
                t['rssi'] += float(self.rng.normal(0.0, snir_fading_std))

        if capture_mode in {"advanced", "omnet"} and noise_floor is not None:
            def _snr(i: int) -> float:
                rssi_i = colliders[i]['rssi']
                total = 10 ** (noise_floor / 10)
                start_i = colliders[i]['start_time']
                end_i = colliders[i]['end_time']
                duration_i = max(end_i - start_i, 1e-9)
                for j, other in enumerate(colliders):
                    if j == i:
                        continue
                    pen = _penalty(colliders[i], other)
                    if pen == float('inf'):
                        continue
                    overlap = min(end_i, other['end_time']) - max(start_i, other['start_time'])
                    if overlap <= 0.0:
                        continue
                    weight = overlap / duration_i
                    total += weight * 10 ** ((other['rssi'] - pen) / 10)
                return rssi_i - 10 * math.log10(total)

            snrs = [_snr(i) for i in range(len(colliders))]
            for idx, snr_val in enumerate(snrs):
                colliders[idx]['snir'] = snr_val
            metrics = snrs
            indices = sorted(range(len(colliders)), key=lambda i: snrs[i], reverse=True)
            strongest = colliders[indices[0]]
            strongest_metric = snrs[indices[0]]
        elif capture_mode == "flora" and flora_phy is not None:
            colliders.sort(key=lambda t: t['rssi'], reverse=True)
            sf_list = [t['sf'] for t in colliders]
            rssi_list = [t['rssi'] for t in colliders]
            start_list = [t['start_time'] for t in colliders]
            end_list = [t['end_time'] for t in colliders]
            freq_list = [t['frequency'] for t in colliders]
            winners = flora_phy.capture(rssi_list, sf_list, start_list, end_list, freq_list)
            capture = any(winners)
            if capture:
                win_idx = winners.index(True)
                strongest = colliders[win_idx]
            else:
                strongest = colliders[0]
        elif (
            capture_mode == "omnet"
            and getattr(self, "omnet_phy", None) is not None
            and getattr(self.omnet_phy, "flora_capture", False)
        ):
            colliders.sort(key=lambda t: t['rssi'], reverse=True)
            sf_list = [t['sf'] for t in colliders]
            rssi_list = [t['rssi'] for t in colliders]
            start_list = [t['start_time'] for t in colliders]
            end_list = [t['end_time'] for t in colliders]
            freq_list = [t['frequency'] for t in colliders]
            bandwidth_list = [t.get('bandwidth', bandwidth) for t in colliders]
            winners = self.omnet_phy.capture(
                rssi_list,
                start_list=start_list,
                end_list=end_list,
                sf_list=sf_list,
                freq_list=freq_list,
                bandwidth_list=bandwidth_list,
            )
            capture = any(winners)
            if capture:
                win_idx = winners.index(True)
                strongest = colliders[win_idx]
            else:
                strongest = colliders[0]
            flora_mode = True
        else:
            colliders.sort(key=lambda t: t['rssi'], reverse=True)
            strongest = colliders[0]
            strongest_metric = strongest['rssi']
            metrics = []
            for t in colliders:
                if t is strongest:
                    metrics.append(strongest_metric)
                else:
                    metrics.append(t['rssi'] - _penalty(strongest, t))

        if capture_mode != "flora" and not flora_mode:
            capture = True
            matrix = non_orth_delta if non_orth_delta is not None else FLORA_NON_ORTH_DELTA
            snir_failure = False
            for t, metric in zip(colliders, metrics):
                if t is strongest:
                    continue
                threshold = threshold_fn(strongest.get('sf', sf))
                if not orthogonal_sf and matrix is not None:
                    sf_w = strongest.get('sf', sf)
                    sf_i = t.get('sf', sf)
                    if (
                        sf_w != sf_i
                        and 7 <= sf_w <= 12
                        and 7 <= sf_i <= 12
                    ):
                        threshold = matrix[sf_w - 7][sf_i - 7]
                if strongest_metric - metric < threshold:
                    capture = False
                    break
            if capture and not _enough_preamble(strongest, colliders):
                capture = False

        strongest_snir = strongest.get('snir')
        if strongest_snir is None and noise_floor is not None:
            strongest_snir = strongest.get('rssi')
            if strongest_snir is not None:
                strongest_snir -= noise_floor

        snir_threshold = threshold_fn(strongest.get('sf', sf))
        snir_failure = False
        failure_reason = None
        if strongest_snir is not None and strongest_snir < snir_threshold:
            snir_failure = True
            failure_reason = "snir_below_threshold"
            capture = False

        if capture and strongest_snir is not None and marginal_drop_prob > 0.0:
            margin = strongest_snir - snir_threshold
            if margin < marginal_snir_db:
                drop_prob = marginal_drop_prob * (1.0 - max(margin, 0.0) / max(marginal_snir_db, 1e-9))
                if self.rng.random() < drop_prob:
                    capture = False
                    failure_reason = "snir_marginal"
                    snir_failure = True

        if capture and residual_collision_prob > 0.0 and interfering_transmissions:
            scale = max(residual_collision_load_scale, 1.0)
            load_factor = min(len(interfering_transmissions) / scale, 1.0)
            drop_prob = residual_collision_prob * load_factor
            if drop_prob > 0.0 and self.rng.random() < drop_prob:
                capture = False
                if failure_reason is None:
                    failure_reason = "residual_load"

        if capture and not use_snir and snir_off_noise_prob > 0.0:
            if self.rng.random() < snir_off_noise_prob:
                capture = False
                if failure_reason is None:
                    failure_reason = "snir_off_noise"

        if capture and (baseline_loss_rate > 0.0 or baseline_collision_rate > 0.0):
            drop_prob = _baseline_drop_prob(len(concurrent_transmissions))
            if drop_prob > 0.0 and self.rng.random() < drop_prob:
                capture = False
                if failure_reason is None:
                    failure_reason = "baseline_loss"

        if capture:
            # Apply preamble rule: the winning packet must have started
            # at least ``capture_window_symbols`` symbols before the new one.
            capture_allowed = True
            if strongest is not new_transmission:
                winner_sym_time = strongest.get('symbol_duration')
                if winner_sym_time is None:
                    bw_w = strongest.get('bandwidth', bandwidth)
                    sf_w = strongest.get('sf', sf)
                    winner_sym_time = (2 ** sf_w) / bw_w
                elapsed = current_time - strongest.get('start_time', current_time)
                if elapsed < capture_window_symbols * winner_sym_time:
                    capture_allowed = False
            if not capture_allowed:
                capture = False

        if capture:
            # Le signal le plus fort sera décodé, les autres sont perdus
            for t in colliders:
                if t is strongest:
                    t['lost_flag'] = False  # gagnant
                else:
                    t['lost_flag'] = True   # perdants
            if strongest.get("collision_reason") is None:
                strongest["collision_reason"] = "capture"
            # Ajouter la transmission la plus forte si c'est la nouvelle (sinon elle est déjà dans active_transmissions)
            if strongest is new_transmission:
                new_transmission['lost_flag'] = False
                self.active_map.setdefault(key, []).append(new_transmission)
                self.active_by_event[event_id] = (key, new_transmission)
            else:
                # La nouvelle transmission est perdue mais doit rester marquée active
                new_transmission['lost_flag'] = True
                self.active_map.setdefault(key, []).append(new_transmission)
                self.active_by_event[event_id] = (key, new_transmission)
            # Les transmissions perdantes restent suivies jusqu'à leur fin simulée
            logger.debug(
                f"Gateway {self.id}: collision avec capture – paquet {strongest['event_id']} capturé, autres perdus.")
            diag_logger.info(
                f"t={current_time:.2f} gw={self.id} capture winner={strongest['event_id']} "
                f"losers={[t['event_id'] for t in colliders if t is not strongest]}"
            )
        else:
            # Aucun signal ne peut être décodé (collision totale)
            for t in colliders:
                t['lost_flag'] = True
                if failure_reason is not None:
                    t['collision_reason'] = failure_reason
            # Conserver la nouvelle transmission marquée comme perdue pour bloquer le canal
            self.active_map.setdefault(key, []).append(new_transmission)
            self.active_by_event[event_id] = (key, new_transmission)
            logger.debug(
                f"Gateway {self.id}: collision sans capture – toutes les transmissions en collision sont perdues.")
            diag_logger.info(
                f"t={current_time:.2f} gw={self.id} collision={[t['event_id'] for t in colliders]} none"
            )
            return

    def end_reception(self, event_id: int, network_server, node_id: int):
        """
        Termine la réception d'une transmission sur cette passerelle si elle est active.
        Cette méthode est appelée lorsque l'heure de fin d'une transmission est atteinte.
        Elle supprime la transmission de la liste active et notifie le serveur réseau en cas de succès.
        :param event_id: Identifiant de l'événement de transmission terminé.
        :param network_server: L'objet NetworkServer pour notifier la réception d'un paquet décodé.
        :param node_id: Identifiant du nœud ayant transmis.
        """
        key, t = self.active_by_event.pop(event_id, (None, None))
        if t is not None and key is not None:
            try:
                self.active_map[key].remove(t)
            except (ValueError, KeyError):
                pass
            if t.get('collision_reason'):
                network_server.register_collision_reason(event_id, t['collision_reason'])
            if not t['lost_flag']:
                network_server.schedule_receive(
                    event_id,
                    node_id,
                    self.id,
                    t['rssi'],
                    snir=t.get('snir'),
                    at_time=t['end_time'],
                )
                logger.debug(
                    f"Gateway {self.id}: successfully received event {event_id} from node {node_id}."
                )
            else:
                logger.debug(
                    f"Gateway {self.id}: event {event_id} from node {node_id} was lost and not received."
                )

    # ------------------------------------------------------------------
    # Downlink handling
    # ------------------------------------------------------------------
    def buffer_downlink(
        self,
        node_id: int,
        frame,
        *,
        data_rate: int | None = None,
        tx_power: float | None = None,
        channel=None,
    ) -> None:
        """Store a downlink frame for a node until its RX window."""
        self.downlink_buffer.setdefault(node_id, []).append(
            (frame, data_rate, tx_power, channel)
        )

    def pop_downlink(self, node_id: int):
        """Retrieve the next pending downlink for a node."""
        queue = self.downlink_buffer.get(node_id)
        if queue:
            return queue.pop(0)
        return None

    def __repr__(self):
        return f"Gateway(id={self.id}, pos=({self.x:.1f},{self.y:.1f}))"
