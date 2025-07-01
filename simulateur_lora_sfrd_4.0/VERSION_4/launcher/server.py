import logging

logger = logging.getLogger(__name__)

# Paramètres ADR (valeurs issues de la spécification LoRaWAN)
REQUIRED_SNR = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
MARGIN_DB = 10.0

class NetworkServer:
    """Représente le serveur de réseau LoRa (collecte des paquets reçus)."""
    def __init__(self):
        """Initialise le serveur réseau."""
        # Ensemble des identifiants d'événements déjà reçus (pour éviter les doublons)
        self.received_events = set()
        # Stockage optionnel d'infos sur les réceptions (par ex : via quelle passerelle)
        self.event_gateway = {}
        # Compteur de paquets reçus
        self.packets_received = 0
        # Indicateur ADR serveur
        self.adr_enabled = False
        # Références pour ADR serveur
        self.nodes = []
        self.gateways = []
        self.channel = None

    # ------------------------------------------------------------------
    # Downlink management
    # ------------------------------------------------------------------
    def send_downlink(
        self,
        node,
        payload: bytes = b"",
        confirmed: bool = False,
        adr_command: tuple | None = None,
        request_ack: bool = False,
    ):
        """Queue a downlink frame for a node via the first gateway."""
        from .lorawan import (
            LoRaWANFrame,
            LinkADRReq,
            SF_TO_DR,
            DBM_TO_TX_POWER_INDEX,
        )

        gw = self.gateways[0] if self.gateways else None
        if gw is None:
            return
        fctrl = 0x20 if request_ack else 0
        frame = LoRaWANFrame(
            mhdr=0x60, fctrl=fctrl, fcnt=node.fcnt_down, payload=payload, confirmed=confirmed
        )
        if adr_command:
            if len(adr_command) == 2:
                sf, power = adr_command
                chmask = node.chmask
                nbtrans = node.nb_trans
            else:
                sf, power, chmask, nbtrans = adr_command
            dr = SF_TO_DR.get(sf, 5)
            p_idx = DBM_TO_TX_POWER_INDEX.get(int(power), 0)
            frame.payload = LinkADRReq(dr, p_idx, chmask, nbtrans).to_bytes()
        node.fcnt_down += 1
        gw.buffer_downlink(node.id, frame)
        try:
            node.downlink_pending += 1
        except AttributeError:
            pass

    def receive(self, event_id: int, node_id: int, gateway_id: int, rssi: float | None = None):
        """
        Traite la réception d'un paquet par le serveur.
        Évite de compter deux fois le même paquet s'il arrive via plusieurs passerelles.
        :param event_id: Identifiant unique de l'événement de transmission du paquet.
        :param node_id: Identifiant du nœud source.
        :param gateway_id: Identifiant de la passerelle ayant reçu le paquet.
        :param rssi: RSSI mesuré par la passerelle pour ce paquet (optionnel).
        """
        if event_id in self.received_events:
            # Doublon (déjà reçu via une autre passerelle)
            logger.debug(f"NetworkServer: duplicate packet event {event_id} from node {node_id} (ignored).")
            return
        # Nouveau paquet reçu
        self.received_events.add(event_id)
        self.event_gateway[event_id] = gateway_id
        self.packets_received += 1
        logger.debug(f"NetworkServer: packet event {event_id} from node {node_id} received via gateway {gateway_id}.")

        node = next((n for n in self.nodes if n.id == node_id), None)
        if node and node.last_adr_ack_req:
            # Device requested an ADR acknowledgement
            self.send_downlink(node)
            node.last_adr_ack_req = False

        # Appliquer ADR complet au niveau serveur
        if self.adr_enabled and rssi is not None:
            from .lorawan import SF_TO_DR, DBM_TO_TX_POWER_INDEX, TX_POWER_INDEX_TO_DBM

            node = next((n for n in self.nodes if n.id == node_id), None)
            if node:
                snr = rssi - self.channel.noise_floor_dBm()
                node.snr_history.append(snr)
                if len(node.snr_history) > 20:
                    node.snr_history.pop(0)
                if len(node.snr_history) >= 20:
                    max_snr = max(node.snr_history)
                    required = REQUIRED_SNR.get(node.sf, -20.0)
                    margin = max_snr - required - MARGIN_DB
                    nstep = int(round(margin / 3.0))

                    sf = node.sf
                    power = node.tx_power
                    p_idx = DBM_TO_TX_POWER_INDEX.get(int(power), 0)

                    while nstep > 0:
                        if sf > 7:
                            sf -= 1
                        elif p_idx < max(TX_POWER_INDEX_TO_DBM.keys()):
                            p_idx += 1
                            power = TX_POWER_INDEX_TO_DBM[p_idx]
                        nstep -= 1

                    while nstep < 0:
                        if p_idx > 0:
                            p_idx -= 1
                            power = TX_POWER_INDEX_TO_DBM[p_idx]
                        elif sf < 12:
                            sf += 1
                        nstep += 1

                    if sf != node.sf or power != node.tx_power:
                        self.send_downlink(node, adr_command=(sf, power, node.chmask, node.nb_trans))
                        node.snr_history.clear()
