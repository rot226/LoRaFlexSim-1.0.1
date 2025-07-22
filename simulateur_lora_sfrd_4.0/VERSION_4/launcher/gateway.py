import logging
import math

logger = logging.getLogger(__name__)

class Gateway:
    """Représente une passerelle LoRa recevant les paquets des nœuds."""

    def __init__(self, gateway_id: int, x: float, y: float, altitude: float = 0.0):
        """
        Initialise une passerelle LoRa.
        :param gateway_id: Identifiant de la passerelle.
        :param x: Position X (mètres).
        :param y: Position Y (mètres).
        """
        self.id = gateway_id
        self.x = x
        self.y = y
        self.altitude = altitude
        # Transmissions en cours indexées par (sf, frequency)
        self.active_map: dict[tuple[int, float], list[dict]] = {}
        # Mapping event_id -> (key, dict) for quick removal
        self.active_by_event: dict[int, tuple[tuple[int, float], dict]] = {}
        # Downlink frames waiting for the corresponding node receive windows
        self.downlink_buffer: dict[int, list] = {}

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
        capture_mode: str = "basic",
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
        :param capture_mode: "basic" pour l'ancien comportement, "advanced" pour
            un calcul basé sur le SNR.
        """
        key = (sf, frequency)
        concurrent_transmissions = [
            t for t in self.active_map.get(key, []) if t['end_time'] > current_time
        ]

        # Filtrer les transmissions dont le chevauchement est significatif
        interfering_transmissions = []
        for t in concurrent_transmissions:
            overlap = min(t['end_time'], end_time) - current_time
            if overlap > min_interference_time:
                interfering_transmissions.append(t)

        # Liste des transmissions en collision potentielles (y compris la nouvelle)
        colliders = interfering_transmissions.copy()
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
            'lost_flag': False,
        }
        colliders.append(new_transmission)

        if not interfering_transmissions:
            # Aucun paquet actif (ou chevauchement inférieur au seuil)
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

        if capture_mode in {"advanced", "omnet"} and noise_floor is not None:
            snrs = [t['rssi'] - noise_floor for t in colliders]
            indices = sorted(range(len(colliders)), key=lambda i: snrs[i], reverse=True)
            strongest = colliders[indices[0]]
            strongest_metric = snrs[indices[0]]
            second = None
            for idx in indices[1:]:
                metric = snrs[idx] - _penalty(strongest, colliders[idx])
                if second is None or metric > second:
                    second = metric
        else:
            colliders.sort(key=lambda t: t['rssi'], reverse=True)
            strongest = colliders[0]
            strongest_metric = strongest['rssi']
            second = None
            for t in colliders[1:]:
                metric = t['rssi'] - _penalty(strongest, t)
                if second is None or metric > second:
                    second = metric

        capture = False
        if second is not None:
            if strongest_metric - second >= capture_threshold:
                capture = True
        else:
            capture = True

        if capture:
            # Le signal le plus fort sera décodé, les autres sont perdus
            for t in colliders:
                if t is strongest:
                    t['lost_flag'] = False  # gagnant
                else:
                    t['lost_flag'] = True   # perdants
            # Retirer toutes les transmissions concurrentes actives qui sont perdantes
            for t in interfering_transmissions:
                if t['lost_flag']:
                    try:
                        self.active_map[key].remove(t)
                        self.active_by_event.pop(t['event_id'], None)
                    except (ValueError, KeyError):
                        pass
            # Ajouter la transmission la plus forte si c'est la nouvelle (sinon elle est déjà dans active_transmissions)
            if strongest is new_transmission:
                new_transmission['lost_flag'] = False
                self.active_map.setdefault(key, []).append(new_transmission)
                self.active_by_event[event_id] = (key, new_transmission)
            # Sinon, la nouvelle transmission est perdue (on ne l'ajoute pas)
            logger.debug(f"Gateway {self.id}: collision avec capture – paquet {strongest['event_id']} capturé, autres perdus.")
        else:
            # Aucun signal ne peut être décodé (collision totale)
            for t in colliders:
                t['lost_flag'] = True
            # Retirer tous les paquets concurrents actifs (ils ne seront pas décodés finalement)
            for t in interfering_transmissions:
                try:
                    self.active_map[key].remove(t)
                    self.active_by_event.pop(t['event_id'], None)
                except (ValueError, KeyError):
                    pass
            # Ne pas ajouter la nouvelle transmission car tout est perdu (pas de décodage possible)
            logger.debug(f"Gateway {self.id}: collision sans capture – toutes les transmissions en collision sont perdues.")
            # **Simplification** : après une collision totale, on considère le canal libre (les signaux brouillés ne sont pas conservés).
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
            if not t['lost_flag']:
                network_server.receive(event_id, node_id, self.id, t['rssi'])
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
    def buffer_downlink(self, node_id: int, frame):
        """Store a downlink frame for a node until its RX window."""
        self.downlink_buffer.setdefault(node_id, []).append(frame)

    def pop_downlink(self, node_id: int):
        """Retrieve the next pending downlink for a node."""
        queue = self.downlink_buffer.get(node_id)
        if queue:
            return queue.pop(0)
        return None

    def __repr__(self):
        return f"Gateway(id={self.id}, pos=({self.x:.1f},{self.y:.1f}))"
