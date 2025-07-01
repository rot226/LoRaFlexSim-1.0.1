import logging

logger = logging.getLogger(__name__)

class Gateway:
    """Représente une passerelle LoRa recevant les paquets des nœuds."""
    def __init__(self, gateway_id: int, x: float, y: float):
        """
        Initialise une passerelle LoRa.
        :param gateway_id: Identifiant de la passerelle.
        :param x: Position X (mètres).
        :param y: Position Y (mètres).
        """
        self.id = gateway_id
        self.x = x
        self.y = y
        # Liste des transmissions actuellement en cours de réception sur cette passerelle
        # Chaque élément est un dictionnaire avec 'event_id', 'node_id', 'sf',
        # 'frequency', 'rssi', 'end_time' et 'lost_flag'
        self.active_transmissions: list[dict] = []
        # Downlink frames waiting for the corresponding node receive windows
        self.downlink_buffer: dict[int, list] = {}

    def start_reception(self, event_id: int, node_id: int, sf: int, rssi: float,
                        end_time: float, capture_threshold: float, current_time: float,
                        frequency: float):
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
        """
        # Ne considérer que les transmissions de même SF et de même fréquence
        # pour les collisions. Les transmissions sur d'autres fréquences sont
        # traitées indépendamment.
        # Récupérer les transmissions actives sur le même SF et la même
        # fréquence qui ne sont pas terminées au current_time.
        concurrent_transmissions = [
            t for t in self.active_transmissions
            if t['sf'] == sf and t['frequency'] == frequency and t['end_time'] > current_time
        ]

        # Liste des transmissions en collision potentielles (y compris la nouvelle)
        colliders = concurrent_transmissions.copy()
        # Ajouter la nouvelle transmission elle-même
        new_transmission = {
            'event_id': event_id,
            'node_id': node_id,
            'sf': sf,
            'frequency': frequency,
            'rssi': rssi,
            'end_time': end_time,
            'lost_flag': False
        }
        colliders.append(new_transmission)

        if not concurrent_transmissions:
            # Aucun paquet actif sur cette SF: on peut recevoir normalement (pas de collision)
            self.active_transmissions.append(new_transmission)
            logger.debug(
                f"Gateway {self.id}: new transmission {event_id} from node {node_id} "
                f"(SF{sf}, {frequency/1e6:.3f} MHz) started, RSSI={rssi:.1f} dBm."
            )
            return

        # Sinon, on a une collision potentielle: déterminer le capture effect
        # Trouver la transmission la plus forte en RSSI dans colliders
        colliders.sort(key=lambda t: t['rssi'], reverse=True)
        strongest = colliders[0]
        second_strongest_rssi = colliders[1]['rssi'] if len(colliders) > 1 else None

        # Vérifier si le plus fort est suffisamment au-dessus du second (et des autres)
        capture = False
        if second_strongest_rssi is not None:
            if strongest['rssi'] - second_strongest_rssi >= capture_threshold:
                capture = True

        if capture:
            # Le signal le plus fort sera décodé, les autres sont perdus
            for t in colliders:
                if t is strongest:
                    t['lost_flag'] = False  # gagnant
                else:
                    t['lost_flag'] = True   # perdants
            # Retirer toutes les transmissions concurrentes actives qui sont perdantes
            for t in concurrent_transmissions:
                if t['lost_flag']:
                    try:
                        self.active_transmissions.remove(t)
                    except ValueError:
                        pass
            # Ajouter la transmission la plus forte si c'est la nouvelle (sinon elle est déjà dans active_transmissions)
            if strongest is new_transmission:
                new_transmission['lost_flag'] = False
                self.active_transmissions.append(new_transmission)
            # Sinon, la nouvelle transmission est perdue (on ne l'ajoute pas)
            logger.debug(f"Gateway {self.id}: collision avec capture – paquet {strongest['event_id']} capturé, autres perdus.")
        else:
            # Aucun signal ne peut être décodé (collision totale)
            for t in colliders:
                t['lost_flag'] = True
            # Retirer tous les paquets concurrents actifs (ils ne seront pas décodés finalement)
            for t in concurrent_transmissions:
                try:
                    self.active_transmissions.remove(t)
                except ValueError:
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
        # Rechercher la transmission correspondante dans la liste active
        for t in list(self.active_transmissions):
            if t['event_id'] == event_id:
                # Retirer de la liste active
                self.active_transmissions.remove(t)
                # Si elle n'était pas marquée perdue, on considère le paquet reçu avec succès
                if not t['lost_flag']:
                    network_server.receive(event_id, node_id, self.id, t['rssi'])
                    logger.debug(f"Gateway {self.id}: successfully received event {event_id} from node {node_id}.")
                else:
                    # Paquet perdu sur cette passerelle (collision ou signal trop faible), on ne notifie pas le serveur
                    logger.debug(f"Gateway {self.id}: event {event_id} from node {node_id} was lost and not received.")
                break  # event_id unique traité, on peut sortir de la boucle

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
