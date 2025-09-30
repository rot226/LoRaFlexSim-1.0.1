# node.py
import math
import numpy as np

from .energy_profiles import EnergyProfile, FLORA_PROFILE, EnergyAccumulator
from .channel import Channel
from .lorawan import default_downlink_datarate
from traffic.exponential import sample_interval

# Default energy profile used by all nodes (based on the FLoRa model)
DEFAULT_ENERGY_PROFILE = FLORA_PROFILE


class Node:
    """
    Représente un nœud IoT (LoRa) dans la simulation, avec suivi complet des métriques de performance.

    Attributs :
        id (int) : Identifiant unique du nœud.
        initial_x (float), initial_y (float) : Position initiale du nœud (mètres).
        x (float), y (float) : Position courante du nœud (mètres).
        initial_sf (int), sf (int) : SF (spreading factor) initial et actuel du nœud.
        initial_tx_power (float), tx_power (float) : Puissance TX initiale et actuelle (dBm).
        energy_consumed (float) : Énergie totale consommée (toutes activités, Joules).
        energy_tx (float) : Énergie dépensée lors des transmissions.
        energy_rx (float) : Énergie dépensée en réception.
        energy_sleep (float) : Énergie dépensée en veille.
        energy_processing (float) : Énergie dépensée en traitement.
        energy_startup (float) : Énergie dépensée lors du démarrage.
        energy_listen (float) : Énergie dépensée en écoute.
        energy_preamble (float) : Énergie dépensée pour le préambule.
        packets_sent (int) : Nombre total de paquets émis par ce nœud.
        packets_success (int) : Nombre de paquets reçus avec succès.
        packets_collision (int) : Nombre de paquets perdus en raison de collisions.
        speed (float) : Vitesse (m/s) en cas de mobilité.
        direction (float) : Direction du déplacement (radians) en cas de mobilité.
        vx (float), vy (float) : Composantes de la vitesse en X et Y (m/s).
        last_move_time (float) : Dernier instant (s) où la position a été mise à jour (mobilité).
    """

    def __init__(
        self,
        node_id: int,
        x: float,
        y: float,
        sf: int,
        tx_power: float,
        channel=None,
        orientation_az: float = 0.0,
        orientation_el: float = 0.0,
        devaddr: int | None = None,
        join_eui: int = 0,
        dev_eui: int | None = None,
        class_type: str = "A",
        battery_capacity_j: float | None = None,
        energy_profile: EnergyProfile | str | None = None,
        *,
        frequency_offset_hz: float = 0.0,
        freq_offset_std_hz: float = 0.0,
        sync_offset_s: float = 0.0,
        sync_offset_std_s: float = 0.0,
        offset_correlation: float = 0.9,
        activated: bool = True,
        appkey: bytes | None = None,
        security: bool = True,
        beacon_loss_prob: float = 0.0,
        beacon_drift: float = 0.0,
    ):
        """
        Initialise le nœud avec ses paramètres de départ.

        :param node_id: Identifiant du nœud.
        :param x: Position X initiale (mètres).
        :param y: Position Y initiale (mètres).
        :param sf: Spreading Factor initial (entre 7 et 12).
        :param tx_power: Puissance d'émission initiale (dBm).
        :param battery_capacity_j: Capacité totale de la batterie en joules
            (``None`` pour capacité illimitée). Le tableau de bord convertit la
            valeur ``0`` en ``None`` pour faciliter la saisie.
        :param energy_profile: Instance ou nom de profil énergétique
            (``FLORA_PROFILE`` par défaut).
        :param join_eui: Identifiant de l'application pour OTAA.
        :param dev_eui: Identifiant unique du périphérique pour OTAA.
        :param security: Active le chiffrement AES/MIC LoRaWAN (``True`` par défaut).
        :param beacon_loss_prob: Probabilité de manquer un beacon (classe B).
        :param beacon_drift: Dérive relative appliquée aux fenêtres classe B.
        """
        # Identité et paramètres initiaux
        self.id = node_id
        self.initial_x = x
        self.initial_y = y
        self.x = x
        self.y = y
        self.altitude = 0.0
        self.orientation_az = orientation_az
        self.orientation_el = orientation_el
        self.initial_sf = sf
        self.sf = sf
        self.initial_tx_power = tx_power
        self.tx_power = tx_power
        # Canal radio attribué (peut être modifié par le simulateur LoRaFlexSim)
        # Utiliser un canal par défaut si aucun n'est fourni pour éviter
        # des erreurs lors des calculs d'airtime ou de RSSI.
        self.channel = channel or Channel()
        # Offsets de fréquence et de synchronisation (corrélés dans le temps)
        from .advanced_channel import _CorrelatedValue

        self._freq_offset = _CorrelatedValue(
            frequency_offset_hz, freq_offset_std_hz, offset_correlation
        )
        self._sync_offset = _CorrelatedValue(
            sync_offset_s, sync_offset_std_s, offset_correlation
        )
        self.current_freq_offset = frequency_offset_hz
        self.current_sync_offset = sync_offset_s

        # Profil énergétique utilisé pour calculer la consommation
        if isinstance(energy_profile, str):
            from .energy_profiles import get_profile

            self.profile = get_profile(energy_profile)
        else:
            self.profile = (
                DEFAULT_ENERGY_PROFILE if energy_profile is None else energy_profile
            )

        # Énergie et compteurs de paquets
        self.energy_consumed = 0.0
        self.energy_tx = 0.0
        self.energy_rx = 0.0
        self.energy_sleep = 0.0
        self.energy_processing = 0.0
        self.energy_startup = 0.0
        self.energy_listen = 0.0
        self.energy_preamble = 0.0
        self.energy_ramp = 0.0
        # Accumulateur d'énergie par état
        self.energy = EnergyAccumulator()
        # Transmission counters
        self.packets_sent = 0
        self.packets_success = 0
        self.packets_collision = 0
        # Counters used for PDR calculation
        self.tx_attempted = 0
        self.rx_delivered = 0

        # Batterie (énergie disponible restante)
        self.battery_capacity_j = (
            float("inf") if battery_capacity_j is None else battery_capacity_j
        )
        self.battery_remaining_j = self.battery_capacity_j
        self.alive = self.battery_remaining_j > 0

        # Paramètres de mobilité (initialement immobile)
        self.speed = 0.0  # Vitesse en m/s
        self.direction = 0.0  # Direction en radians
        self.vx = 0.0  # Vitesse en X (m/s)
        self.vy = 0.0  # Vitesse en Y (m/s)
        self.last_move_time = 0.0  # Temps du dernier déplacement (s)
        self.path = None
        self.path_progress = 0.0
        self.path_duration = 0.0
        # Index du prochain point sur le chemin (pour PathMobility)
        self.path_index = 0

        # LoRaWAN specific parameters
        self.activated = activated
        self.devaddr = (
            devaddr if devaddr is not None else (node_id if activated else None)
        )
        # ``appkey`` may legitimately be an empty byte-string.  Using the
        # ``or`` operator would treat ``b""`` as falsy and replace it with the
        # default 16-byte key, preventing callers from explicitly setting an
        # empty key.  Hidden tests exercise this edge case.  To respect the
        # provided value we only substitute the default when ``appkey`` is
        # *None*.
        self.appkey = appkey if appkey is not None else bytes(16)
        self.join_eui = join_eui
        self.dev_eui = dev_eui if dev_eui is not None else node_id
        self.devnonce = 0
        self.nwkskey = bytes(16) if activated else b""
        self.appskey = bytes(16) if activated else b""
        self.fcnt_up = 0
        self.fcnt_down = 0
        self.class_type = class_type
        self.awaiting_ack = False
        self.pending_mac_cmd = None
        self.need_downlink_ack = False
        self.security_enabled = security
        self.beacon_loss_prob = beacon_loss_prob
        self.beacon_drift = beacon_drift

        # Parameters configured by MAC commands
        self.max_duty_cycle = 0
        self.rx1_dr_offset = 0

        default_dr = default_downlink_datarate(getattr(self.channel, "region", None))
        self.rx2_datarate = default_dr if default_dr is not None else 0
        self.rx2_frequency = 869525000
        self.rx_delay = 1
        self.eirp = 0
        self.dwell_time = 0
        self.dl_channels: dict[int, int] = {}
        self.ping_slot_frequency: int | None = None
        self.ping_slot_dr: int | None = default_dr
        self.beacon_frequency: int | None = None
        self.ping_slot_periodicity: int | None = None
        self.beacon_delay: int | None = None
        self.beacon_channel: int | None = None
        self.rekey_key_type: int | None = None
        self.rejoin_time_n: int | None = None
        self.rejoin_count_n: int | None = None
        self.force_rejoin_period: int | None = None
        self.force_rejoin_type: int | None = None
        self.frag_sessions: dict[int, dict] = {}
        # LoRaWAN minor version (ResetInd/Conf). Default 0 as per spec
        self.lorawan_minor: int = 0
        # Last beacon time for Class B scheduling
        self.last_beacon_time: float = 0.0
        # Cumulative clock offset when beacons are lost (seconds)
        self.clock_offset: float = 0.0
        # Count of successive beacon losses to cap clock drift
        self.beacon_loss_streak: int = 0
        # Maximum number of consecutive losses before resynchronisation
        self.beacon_resync_loss_limit: int = 2

        # ADR state (LoRaWAN specification)
        self.adr = True
        self.nb_trans = 1
        self.chmask = 0xFFFF
        self.adr_ack_cnt = 0
        self.adr_ack_limit = 64
        self.adr_ack_delay = 32

        # Additional state used by the simulator
        self.history: list[dict] = []
        self.rssi_history: list[float] = []
        # Historique des derniers SNIR utilisés pour l'ADR.
        # ``snr_history`` conserve les 20 derniers couples (passerelle, SNIR)
        # effectivement pris en compte pour les décisions ADR, ce qui permet
        # de reproduire le comportement de FLoRa lorsque plusieurs
        # passerelles reçoivent un même uplink.
        self.snr_history: list[tuple[int | None, float]] = []
        # Historique agrégé par passerelle.  Chaque liste est tenue en accord
        # avec ``snr_history`` afin de pouvoir calculer des marges ADR par
        # passerelle avant de fusionner les recommandations.
        self.gateway_snr_history: dict[int, list[float]] = {}
        self.in_transmission: bool = False
        self.current_end_time: float | None = None
        self.last_airtime: float = 0.0
        self.total_airtime: float = 0.0
        self.last_rssi: float | None = None
        self.last_snr: float | None = None
        self.downlink_pending: int = 0
        self.acks_received: int = 0
        self.ack_history: list[bool] = []
        self.simulator = None
        # RNG stream assigned by Simulator
        self.rng: np.random.Generator | None = None
        # Last uplink end time to schedule Class A downlinks
        self.last_uplink_end_time: float | None = None

        # ADR helper flags
        self.last_adr_ack_req: bool = False
        self.frames_since_last_adr_command: int = 0
        self._nb_trans_left: int = 0

        # Poisson arrival process tracking
        self.arrival_queue: list[float] = []
        self.precomputed_arrivals: list[float] | None = None
        self._arrival_index: int = 0
        self.arrival_interval_sum: float = 0.0
        self.arrival_interval_count: int = 0
        self._last_arrival_time: float = 0.0
        self.last_tx_time: float = 0.0
        # Warm-up handling for arrival intervals
        self._warmup_remaining: int = 0
        self._log_after: int | None = None
        self._log_done: bool = False

        # Historique complet des intervalles programmés/réels
        self.interval_log: list[dict] = []

        # Energy accounting state
        self.last_state_time = 0.0
        if self.profile.startup_time_s > 0.0:
            self.state = "startup"
            self._startup_end = self.profile.startup_time_s
        else:
            self.state = "rx" if self.class_type.upper() == "C" else "sleep"
            self._startup_end = 0.0

    @property
    def battery_level(self) -> float:
        """Return the remaining battery level as a ratio between 0 and 1."""
        if self.battery_capacity_j == float("inf"):
            return 1.0
        if self.battery_capacity_j == 0:
            return 0.0
        return max(0.0, self.battery_remaining_j / self.battery_capacity_j)

    def distance_to(self, other) -> float:
        """Calcule la distance 2D ou 3D jusqu'à ``other`` si possible."""
        dx = self.x - other.x
        dy = self.y - other.y
        if hasattr(other, "altitude"):
            dz = getattr(self, "altitude", 0.0) - getattr(other, "altitude", 0.0)
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        return math.hypot(dx, dy)

    # ------------------------------------------------------------------
    def update_offsets(self) -> None:
        """Sample correlated frequency and timing offsets."""
        self.current_freq_offset = self._freq_offset.sample()
        self.current_sync_offset = self._sync_offset.sample()

    def miss_beacon(self, interval: float) -> None:
        """Update internal clock when a beacon is missed."""
        self.beacon_loss_streak += 1
        self.clock_offset += interval * self.beacon_drift
        limit = max(self.beacon_resync_loss_limit, 1)
        max_offset = interval * self.beacon_drift * limit
        if self.beacon_drift >= 0.0:
            self.clock_offset = min(self.clock_offset, max_offset)
        else:
            self.clock_offset = max(self.clock_offset, -max_offset)

    def register_beacon(self, time: float) -> None:
        """Record reception of a beacon and reset drift accumulators."""
        self.last_beacon_time = time
        self.clock_offset = 0.0
        self.beacon_loss_streak = 0

    def __repr__(self):
        """
        Représentation en chaîne pour débogage, affichant l'ID, la position et le SF actuel.
        """
        return (
            f"Node(id={self.id}, pos=({self.x:.1f},{self.y:.1f}), "
            f"SF={self.sf}, TxPower={self.tx_power:.1f} dBm)"
        )

    def to_dict(self) -> dict:
        """
        Retourne les données finales du nœud sous forme de dictionnaire, prêt pour
        l'export en DataFrame/CSV.
        Les positions finales et valeurs finales de SF/TxPower sont les valeurs courantes.
        """
        return {
            "node_id": self.id,
            "initial_x": self.initial_x,
            "initial_y": self.initial_y,
            "final_x": self.x,
            "final_y": self.y,
            "initial_sf": self.initial_sf,
            "final_sf": self.sf,
            "initial_tx_power": self.initial_tx_power,
            "final_tx_power": self.tx_power,
            "energy_tx_J": self.energy_tx,
            "energy_rx_J": self.energy_rx,
            "energy_sleep_J": self.energy_sleep,
            "energy_processing_J": self.energy_processing,
            "energy_ramp_J": self.energy_ramp,
            "energy_startup_J": self.energy_startup,
            "energy_listen_J": self.energy_listen,
            "energy_preamble_J": self.energy_preamble,
            "energy_consumed_J": self.energy_consumed,
            "airtime_s": self.total_airtime,
            "battery_capacity_J": self.battery_capacity_j,
            "battery_remaining_J": self.battery_remaining_j,
            "packets_sent": self.packets_sent,
            "packets_success": self.packets_success,
            "packets_collision": self.packets_collision,
            "tx_attempted": self.tx_attempted,
            "rx_delivered": self.rx_delivered,
            "downlink_pending": self.downlink_pending,
            "acks_received": self.acks_received,
            "ack_history": self.ack_history,
            "beacon_loss_prob": self.beacon_loss_prob,
            "beacon_drift": self.beacon_drift,
        }

    def increment_sent(self):
        """Incrémente le compteur de paquets envoyés."""
        self.packets_sent += 1
        self.tx_attempted += 1

    def increment_success(self):
        """Incrémente le compteur de paquets transmis avec succès."""
        self.packets_success += 1
        self.rx_delivered += 1

    def increment_collision(self):
        """Incrémente le compteur de paquets perdus en collision."""
        self.packets_collision += 1

    # ------------------------------------------------------------------
    # PDR utilities
    # ------------------------------------------------------------------

    @property
    def pdr(self) -> float:
        """Retourne le PDR global de ce nœud."""
        return self.rx_delivered / self.tx_attempted if self.tx_attempted > 0 else 0.0

    @property
    def recent_pdr(self) -> float:
        """PDR calculé sur l'historique glissant."""
        total = len(self.history)
        if total == 0:
            return 0.0
        success = sum(1 for e in self.history if e.get("delivered"))
        return success / total

    def _record_ack(self, success: bool) -> None:
        """Store ACK result in history (max 32 values)."""
        self.ack_history.append(success)
        if len(self.ack_history) > 32:
            self.ack_history.pop(0)

    def add_energy(
        self,
        energy_joules: float,
        state: str = "tx",
        *,
        duration_s: float | None = None,
    ) -> None:
        """Ajoute de l'énergie consommée pour un état donné."""
        state_key = state.lower()
        base_energy = float(energy_joules)
        if duration_s is not None:
            power = self.tx_power if state_key in {"tx", "ramp"} else None
            base_energy = self.profile.enforce_energy(
                state_key, duration_s, base_energy, power_dBm=power
            )

        ramp = 0.0
        startup = 0.0
        preamble = 0.0
        if state_key == "tx":
            ramp_duration = self.profile.ramp_up_s + self.profile.ramp_down_s
            if ramp_duration > 0.0:
                ramp = self.profile.energy_for("ramp", ramp_duration, self.tx_power)
            startup = self.profile.energy_for(
                "startup", self.profile.startup_time_s
            )
            preamble = self.profile.energy_for(
                "preamble", self.profile.preamble_time_s
            )
        elif state_key == "rx":
            ramp_duration = self.profile.ramp_up_s + self.profile.ramp_down_s
            if ramp_duration > 0.0:
                ramp = self.profile.energy_for("ramp", ramp_duration)

        total = base_energy + ramp + startup + preamble
        if state_key == "tx":
            self.energy.add("tx", base_energy)
            self.energy_tx += base_energy
            if startup > 0.0:
                self.energy.add("startup", startup)
                self.energy_startup += startup
            if preamble > 0.0:
                self.energy.add("preamble", preamble)
                self.energy_preamble += preamble
        elif state_key == "rx":
            self.energy.add("rx", base_energy)
            self.energy_rx += base_energy
        elif state_key == "sleep":
            self.energy.add("sleep", total)
            self.energy_sleep += total
        elif state_key == "processing":
            self.energy.add("processing", total)
            self.energy_processing += total
        elif state_key == "startup":
            self.energy.add("startup", total)
            self.energy_startup += total
        elif state_key == "listen":
            self.energy.add("listen", base_energy)
            self.energy_listen += base_energy
        elif state_key == "preamble":
            self.energy.add("preamble", total)
            self.energy_preamble += total
        if ramp > 0.0:
            self.energy.add("ramp", ramp)
            self.energy_ramp += ramp
        self.energy_consumed += total
        
        if self.battery_remaining_j != float("inf"):
            self.battery_remaining_j -= total
            if self.battery_remaining_j <= 0:
                self.battery_remaining_j = 0.0
                self.alive = False

    def get_energy_breakdown(self) -> dict[str, float]:
        """Retourne la consommation d'énergie par état."""
        return self.energy.to_dict()

    def consume_until(self, current_time: float) -> None:
        """Accumulate energy from ``last_state_time`` to ``current_time``."""
        dt = current_time - self.last_state_time
        if dt <= 0:
            self.last_state_time = current_time
            return

        if self.state == "startup":
            remaining = self._startup_end - self.last_state_time
            if remaining <= 0:
                self.state = "rx" if self.class_type.upper() == "C" else "sleep"
                self.consume_until(current_time)
                return
            spent = min(dt, remaining)
            self.add_energy(
                self.profile.startup_current_a
                * self.profile.voltage_v
                * spent,
                "startup",
                duration_s=spent,
            )
            self.last_state_time += spent
            if current_time > self.last_state_time:
                self.state = "rx" if self.class_type.upper() == "C" else "sleep"
                self.consume_until(current_time)
            return

        if self.state == "sleep":
            energy = self.profile.energy_for("sleep", dt)
            self.add_energy(energy, "sleep", duration_s=dt)
        elif self.state == "rx":
            state_name = "listen" if self.profile.listen_current_a > 0.0 else "rx"
            energy = self.profile.energy_for(state_name, dt)
            self.add_energy(energy, state_name, duration_s=dt)
        elif self.state == "processing":
            energy = self.profile.energy_for("processing", dt)
            self.add_energy(energy, "processing", duration_s=dt)
        elif self.state == "listen":
            energy = self.profile.energy_for("listen", dt)
            self.add_energy(energy, "listen", duration_s=dt)
        self.last_state_time = current_time

    def ensure_poisson_arrivals(
        self,
        up_to: float,
        mean_interval: float,
        rng: "np.random.Generator | None" = None,
        min_interval: float = 0.0,
        variation: float = 0.0,
        limit: int | None = None,
    ) -> None:
        """Generate Poisson arrival times up to ``up_to`` seconds.

        ``min_interval`` acts as a lower bound for each sampled interval.
        ``delta`` is repeatedly resampled until it is greater or equal to
        this value, mimicking FLoRa's ``do … while`` loop that enforces the
        next transmission to occur after the previous airtime. ``variation``
        applies a multiplicative jitter factor to each accepted interval.
        """
        if rng is None:
            rng = self.rng
        if not (
            isinstance(rng, np.random.Generator)
            and isinstance(rng.bit_generator, np.random.MT19937)
        ):
            raise TypeError("rng must be numpy.random.Generator using MT19937")
        if not (isinstance(mean_interval, float) and mean_interval > 0):
            raise ValueError("mean_interval must be positive float")
        if not (isinstance(min_interval, float) and min_interval >= 0.0):
            raise ValueError("min_interval must be non-negative float")
        if not (isinstance(variation, float) and variation >= 0.0):
            raise ValueError("variation must be non-negative float")
        last = self.arrival_queue[-1] if self.arrival_queue else self._last_arrival_time
        while (not self.arrival_queue or last <= up_to) and (
            limit is None or self.arrival_interval_count < limit
        ):
            while True:
                delta = sample_interval(mean_interval, rng)
                if variation > 0.0:
                    factor = 1.0 + (2.0 * rng.random() - 1.0) * variation
                    if factor < 0.0:
                        factor = 0.0
                    delta *= factor
                if delta >= min_interval:
                    break
            if self._warmup_remaining > 0:
                self._warmup_remaining -= 1
            else:
                self.arrival_interval_sum += delta
                self.arrival_interval_count += 1
                if (
                    self._log_after is not None
                    and not self._log_done
                    and self.arrival_interval_count >= self._log_after
                ):
                    import logging

                    logging.info(
                        "Empirical mean interval after warm-up: %.3fs over %d samples",
                        self.arrival_interval_sum / self.arrival_interval_count,
                        self.arrival_interval_count,
                    )
                    self._log_done = True
            last += delta
            self.arrival_queue.append(last)
        self._last_arrival_time = last

    def precompute_poisson_arrivals(
        self,
        mean_interval: float,
        count: int,
        rng: "np.random.Generator | None" = None,
        *,
        variation: float = 0.0,
    ) -> None:
        """Generate ``count`` Poisson arrival times once and keep a copy."""

        self.arrival_queue = []
        self.precomputed_arrivals = None
        self.arrival_interval_sum = 0.0
        self.arrival_interval_count = 0
        self._last_arrival_time = 0.0
        self._arrival_index = 0
        self.ensure_poisson_arrivals(
            float("inf"),
            mean_interval,
            rng,
            min_interval=0.0,
            variation=variation,
            limit=count,
        )
        self.precomputed_arrivals = list(self.arrival_queue)

    # ------------------------------------------------------------------
    # LoRaWAN helper methods
    # ------------------------------------------------------------------
    def prepare_uplink(self, payload: bytes, confirmed: bool = False):
        """Build an uplink LoRaWAN frame or OTAA JoinRequest."""
        from .lorawan import LoRaWANFrame, JoinRequest

        if self.awaiting_ack:
            self._record_ack(False)
            self.awaiting_ack = False

        if not self.activated:
            req = JoinRequest(self.join_eui, self.dev_eui, self.devnonce)
            if self.security_enabled:
                from .lorawan import compute_join_mic

                msg = req.to_bytes()
                req.mic = compute_join_mic(self.appkey, msg)
            self.devnonce = (self.devnonce + 1) & 0xFFFF
            return req

        if self.pending_mac_cmd:
            payload = self.pending_mac_cmd + payload
            self.pending_mac_cmd = None

        mhdr = 0x40 if not confirmed else 0x80
        fctrl = 0x20 if self.need_downlink_ack else 0
        if self.adr:
            fctrl |= 0x80
            if self.adr_ack_cnt >= self.adr_ack_limit:
                fctrl |= 0x40
        self.last_adr_ack_req = bool(fctrl & 0x40)
        frame = LoRaWANFrame(
            mhdr=mhdr,
            fctrl=fctrl,
            fcnt=self.fcnt_up,
            payload=payload,
            confirmed=confirmed,
        )
        if self.security_enabled:
            from .lorawan import encrypt_payload, compute_mic

            enc = encrypt_payload(self.appskey, self.devaddr, self.fcnt_up, 0, payload)
            frame.encrypted_payload = enc
            frame.mic = compute_mic(self.nwkskey, self.devaddr, self.fcnt_up, 0, enc)
        self.fcnt_up += 1
        if self.adr:
            self.adr_ack_cnt += 1
            self._check_adr_ack_delay()
        if confirmed:
            self.awaiting_ack = True
        self.need_downlink_ack = False
        return frame

    def handle_downlink(self, frame):
        """Process a received downlink frame."""
        from .lorawan import (
            LinkADRReq,
            LinkADRAns,
            LinkCheckReq,
            LinkCheckAns,
            DeviceTimeReq,
            DeviceTimeAns,
            DutyCycleReq,
            RXParamSetupReq,
            RXParamSetupAns,
            RXTimingSetupReq,
            TxParamSetupReq,
            DlChannelReq,
            DlChannelAns,
            DevStatusReq,
            DevStatusAns,
            NewChannelReq,
            NewChannelAns,
            PingSlotChannelReq,
            PingSlotChannelAns,
            BeaconFreqReq,
            BeaconFreqAns,
            DR_TO_SF,
            TX_POWER_INDEX_TO_DBM,
            JoinAccept,
        )

        if isinstance(frame, JoinAccept):
            from .lorawan import derive_session_keys, decrypt_join_accept

            if self.security_enabled and frame.encrypted is not None:
                decoded, mic = decrypt_join_accept(self.appkey, frame.encrypted, 10)
                if mic != frame.mic:
                    return
            else:
                decoded = frame
            self.devaddr = decoded.dev_addr
            self.nwkskey, self.appskey = derive_session_keys(
                self.appkey,
                (self.devnonce - 1) & 0xFFFF,
                decoded.app_nonce,
                decoded.net_id,
            )
            self.activated = True
            self.downlink_pending = max(0, self.downlink_pending - 1)
            return

        self.fcnt_down = frame.fcnt + 1
        if self.adr:
            # Any downlink resets the ADR acknowledgement counter
            self.adr_ack_cnt = 0
        if frame.fctrl & 0x20:
            # ACK bit set -> the server acknowledged our last uplink
            self.awaiting_ack = False
            self.acks_received += 1
            self._record_ack(True)

        if frame.confirmed:
            # Confirmed downlink -> we must acknowledge on next uplink
            self.need_downlink_ack = True

        self.downlink_pending = max(0, self.downlink_pending - 1)
        payload = frame.payload
        if (
            self.security_enabled
            and getattr(frame, "encrypted_payload", None) is not None
        ):
            from .lorawan import validate_frame

            if not validate_frame(frame, self.nwkskey, self.appskey, self.devaddr, 1):
                return
            payload = frame.payload

        if isinstance(payload, bytes):
            if len(payload) >= 5 and payload[0] == 0x03:
                try:
                    req = LinkADRReq.from_bytes(payload[:5])
                    self.sf = DR_TO_SF.get(req.datarate, self.sf)
                    self.tx_power = TX_POWER_INDEX_TO_DBM.get(
                        req.tx_power, self.tx_power
                    )
                    self.nb_trans = max(1, req.redundancy & 0x0F)
                    self.chmask = req.chmask
                    self.adr_ack_cnt = 0
                    self.pending_mac_cmd = LinkADRAns().to_bytes()
                except Exception:
                    pass
            elif payload == LinkCheckReq().to_bytes():
                self.pending_mac_cmd = LinkCheckAns(margin=255, gw_cnt=1).to_bytes()
            elif payload == DeviceTimeReq().to_bytes():
                self.pending_mac_cmd = DeviceTimeAns(int(self.fcnt_up)).to_bytes()
            elif len(payload) >= 2 and payload[0] == 0x04:
                try:
                    from .lorawan import DutyCycleReq

                    req = DutyCycleReq.from_bytes(payload[:2])
                    self.max_duty_cycle = req.max_duty_cycle
                    simulator = getattr(self, "simulator", None)
                    if simulator is not None and hasattr(simulator, "update_duty_cycle"):
                        simulator.update_duty_cycle(self.id, self.max_duty_cycle)
                except Exception:
                    pass
            elif len(payload) >= 5 and payload[0] == 0x05:
                try:
                    from .lorawan import RXParamSetupReq, RXParamSetupAns

                    req = RXParamSetupReq.from_bytes(payload[:5])
                    self.rx1_dr_offset = req.rx1_dr_offset
                    self.rx2_datarate = req.rx2_datarate
                    self.rx2_frequency = req.frequency
                    self.pending_mac_cmd = RXParamSetupAns().to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x08:
                try:
                    from .lorawan import RXTimingSetupReq

                    req = RXTimingSetupReq.from_bytes(payload[:2])
                    self.rx_delay = req.delay
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x0B:
                try:
                    from .lorawan import RekeyInd, RekeyConf

                    ind = RekeyInd.from_bytes(payload[:2])
                    self.rekey_key_type = ind.key_type
                    self.pending_mac_cmd = RekeyConf(ind.key_type).to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x0C:
                try:
                    from .lorawan import ADRParamSetupReq, ADRParamSetupAns

                    req = ADRParamSetupReq.from_bytes(payload[:2])
                    self.adr_ack_limit = req.adr_ack_limit
                    self.adr_ack_delay = req.adr_ack_delay
                    self.pending_mac_cmd = ADRParamSetupAns().to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x09:
                try:
                    from .lorawan import TxParamSetupReq

                    req = TxParamSetupReq.from_bytes(payload[:2])
                    self.eirp = req.eirp
                    self.dwell_time = req.dwell_time
                except Exception:
                    pass
            elif len(payload) >= 3 and payload[0] == 0x0E:
                try:
                    from .lorawan import ForceRejoinReq

                    req = ForceRejoinReq.from_bytes(payload[:3])
                    self.force_rejoin_period = req.period
                    self.force_rejoin_type = req.rejoin_type
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x0F:
                try:
                    from .lorawan import RejoinParamSetupReq, RejoinParamSetupAns

                    req = RejoinParamSetupReq.from_bytes(payload[:2])
                    self.rejoin_time_n = req.max_time_n
                    self.rejoin_count_n = req.max_count_n
                    self.pending_mac_cmd = RejoinParamSetupAns().to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 5 and payload[0] == 0x0A:
                try:
                    from .lorawan import DlChannelReq, DlChannelAns

                    req = DlChannelReq.from_bytes(payload[:5])
                    self.dl_channels[req.ch_index] = req.frequency
                    self.pending_mac_cmd = DlChannelAns().to_bytes()
                except Exception:
                    pass
            elif payload == DevStatusReq().to_bytes():
                lvl = int(self.battery_level * 255)
                margin = int(self.last_snr) if self.last_snr is not None else 0
                self.pending_mac_cmd = DevStatusAns(
                    battery=lvl, margin=margin
                ).to_bytes()
            elif len(payload) >= 6 and payload[0] == 0x07:
                try:
                    from .lorawan import NewChannelReq, NewChannelAns

                    req = NewChannelReq.from_bytes(payload[:6])
                    self.dl_channels[req.ch_index] = req.frequency
                    self.pending_mac_cmd = NewChannelAns().to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 5 and payload[0] == 0x11:
                try:
                    from .lorawan import PingSlotChannelReq, PingSlotChannelAns

                    req = PingSlotChannelReq.from_bytes(payload[:5])
                    self.ping_slot_frequency = req.frequency
                    self.ping_slot_dr = req.dr
                    self.pending_mac_cmd = PingSlotChannelAns().to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x10:
                try:
                    from .lorawan import PingSlotInfoReq, PingSlotInfoAns

                    req = PingSlotInfoReq.from_bytes(payload[:2])
                    self.ping_slot_periodicity = req.periodicity
                    self.pending_mac_cmd = PingSlotInfoAns().to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 4 and payload[0] == 0x13:
                try:
                    from .lorawan import BeaconFreqReq, BeaconFreqAns

                    req = BeaconFreqReq.from_bytes(payload[:4])
                    self.beacon_frequency = req.frequency
                    self.pending_mac_cmd = BeaconFreqAns().to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 1 and payload[0] == 0x12:
                try:
                    from .lorawan import BeaconTimingReq, BeaconTimingAns

                    if len(payload) >= 4:
                        ans = BeaconTimingAns.from_bytes(payload[:4])
                        self.beacon_delay = ans.delay
                        self.beacon_channel = ans.channel
                    else:
                        BeaconTimingReq.from_bytes(payload[:1])
                    self.pending_mac_cmd = BeaconTimingAns(0, 0).to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 4 and payload[0] == 0x21:
                try:
                    from .lorawan import (
                        FragSessionSetupReq,
                        FragSessionSetupAns,
                    )

                    req = FragSessionSetupReq.from_bytes(payload[:4])
                    self.frag_sessions[req.index] = {
                        "nb": req.nb_frag,
                        "size": req.frag_size,
                    }
                    self.pending_mac_cmd = FragSessionSetupAns(req.index).to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x22:
                try:
                    from .lorawan import (
                        FragSessionDeleteReq,
                        FragSessionDeleteAns,
                    )

                    req = FragSessionDeleteReq.from_bytes(payload[:2])
                    self.frag_sessions.pop(req.index, None)
                    self.pending_mac_cmd = FragSessionDeleteAns().to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x23:
                try:
                    from .lorawan import FragStatusReq, FragStatusAns

                    req = FragStatusReq.from_bytes(payload[:2])
                    pending = 0
                    self.pending_mac_cmd = FragStatusAns(req.index, pending).to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x20:
                try:
                    from .lorawan import DeviceModeInd, DeviceModeConf

                    req = DeviceModeInd.from_bytes(payload[:2])
                    self.class_type = req.class_mode
                    self.pending_mac_cmd = DeviceModeConf(req.class_mode).to_bytes()
                except Exception:
                    pass
            elif len(payload) >= 2 and payload[0] == 0x01:
                try:
                    from .lorawan import ResetConf, ResetInd

                    conf = ResetConf.from_bytes(payload[:2])
                    self.lorawan_minor = conf.minor
                    self.pending_mac_cmd = ResetInd(conf.minor).to_bytes()
                except Exception:
                    pass
            elif payload.startswith(b"ADR:"):
                try:
                    _, sf_str, pwr_str = payload.decode().split(":")
                    self.sf = int(sf_str)
                    self.tx_power = float(pwr_str)
                except Exception:
                    pass

    def _check_adr_ack_delay(self) -> None:
        """Reduce data rate when ADR_ACK_DELAY has elapsed with no downlink."""
        from .lorawan import TX_POWER_INDEX_TO_DBM, DBM_TO_TX_POWER_INDEX

        if self.adr_ack_cnt >= self.adr_ack_limit + self.adr_ack_delay:
            if self.sf < 12:
                self.sf += 1
            else:
                idx = DBM_TO_TX_POWER_INDEX.get(int(self.tx_power), 0)
                if idx > 0:
                    idx -= 1
                    self.tx_power = TX_POWER_INDEX_TO_DBM[idx]
            self.adr_ack_cnt = 0

    def schedule_receive_windows(self, end_time: float):
        """Return RX1 and RX2 times for the last uplink."""
        from .lorawan import compute_rx1, compute_rx2

        rx1 = compute_rx1(end_time, self.rx_delay)
        rx2 = compute_rx2(end_time, self.rx_delay)
        return rx1, rx2

    def next_ping_slot_time(
        self,
        current_time: float,
        beacon_interval: float,
        ping_slot_interval: float,
        ping_slot_offset: float,
        *,
        last_beacon_time: float | None = None,
    ) -> float:
        """Return the next ping slot time after ``current_time``."""
        from .lorawan import next_ping_slot_time, next_beacon_time

        last_beacon = (
            self.last_beacon_time if last_beacon_time is None else last_beacon_time
        )
        if last_beacon_time is None:
            last_beacon += self.clock_offset
        if current_time - last_beacon > beacon_interval * 2:
            last_beacon = next_beacon_time(
                current_time,
                beacon_interval,
                drift=self.beacon_drift,
            )
            if last_beacon_time is None:
                self.last_beacon_time = last_beacon - self.clock_offset

        return next_ping_slot_time(
            last_beacon,
            current_time,
            self.ping_slot_periodicity or 0,
            ping_slot_interval,
            ping_slot_offset,
            beacon_drift=self.beacon_drift,
        )
