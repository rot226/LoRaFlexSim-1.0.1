"""Advanced physical layer models for LoRa simulations."""

from __future__ import annotations

import math
import random


class _CorrelatedFading:
    """Temporal correlation for Rayleigh/Rician fading with multiple taps."""

    def __init__(self, kind: str, k_factor: float, correlation: float, paths: int = 1) -> None:
        self.kind = kind
        self.k = k_factor
        self.corr = correlation
        self.paths = max(1, int(paths))
        self.i = [0.0] * self.paths
        self.q = [0.0] * self.paths

    def sample_db(self) -> float:
        if self.kind not in {"rayleigh", "rician"}:
            return 0.0
        std = math.sqrt(max(1.0 - self.corr ** 2, 0.0))
        if self.kind == "rayleigh":
            mean_i = 0.0
            sigma = 1.0
        else:  # rician
            mean_i = math.sqrt(self.k / (self.k + 1.0))
            sigma = math.sqrt(1.0 / (2.0 * (self.k + 1.0)))

        sum_i = 0.0
        sum_q = 0.0
        for p in range(self.paths):
            self.i[p] = self.corr * self.i[p] + std * random.gauss(mean_i, sigma)
            self.q[p] = self.corr * self.q[p] + std * random.gauss(0.0, sigma)
            sum_i += self.i[p]
            sum_q += self.q[p]
        amp = math.sqrt(sum_i ** 2 + sum_q ** 2) / self.paths
        return 20 * math.log10(max(amp, 1e-12))


class _CorrelatedValue:
    """Correlated random walk used for drifting offsets."""

    def __init__(self, mean: float, std: float, correlation: float) -> None:
        self.mean = mean
        self.std = std
        self.corr = correlation
        self.value = mean

    def sample(self) -> float:
        self.value = self.corr * self.value + (1.0 - self.corr) * self.mean
        if self.std > 0.0:
            self.value += random.gauss(0.0, self.std)
        return self.value


class AdvancedChannel:
    """Optional channel with more detailed propagation models."""

    def __init__(
        self,
        base_station_height: float = 30.0,
        mobile_height: float = 1.5,
        propagation_model: str = "cost231",
        fading: str = "rayleigh",
        rician_k: float = 1.0,
        terrain: str = "urban",
        weather_loss_dB_per_km: float = 0.0,
        weather_loss_std_dB_per_km: float = 0.0,
        weather_correlation: float | None = None,
        fine_fading_std: float = 0.0,
        fading_correlation: float = 0.9,
        variable_noise_std: float = 0.0,
        advanced_capture: bool = False,
        frequency_offset_hz: float = 0.0,
        freq_offset_std_hz: float = 0.0,
        sync_offset_s: float = 0.0,
        sync_offset_std_s: float = 0.0,
        dev_frequency_offset_hz: float = 0.0,
        dev_freq_offset_std_hz: float = 0.0,
        temperature_std_K: float = 0.0,
        pa_non_linearity_dB: float = 0.0,
        pa_non_linearity_std_dB: float = 0.0,
        pa_non_linearity_curve: tuple[float, float, float] | None = None,
        pa_distortion_std_dB: float = 0.0,
        humidity_percent: float = 50.0,
        humidity_std_percent: float = 0.0,
        humidity_noise_coeff_dB: float = 0.0,
        phase_noise_std_dB: float = 0.0,
        clock_jitter_std_s: float = 0.0,
        frontend_filter_order: int = 0,
        frontend_filter_bw: float | None = None,
        obstacle_map: list[list[float]] | None = None,
        map_area_size: float | None = None,
        obstacle_height_map: list[list[float]] | None = None,
        obstacle_map_file: str | None = None,
        obstacle_height_map_file: str | None = None,
        default_obstacle_dB: float = 0.0,
        multipath_paths: int = 1,
        indoor_n_floors: int = 0,
        indoor_floor_loss_dB: float = 15.0,
        **kwargs,
    ) -> None:
        """Initialise the advanced channel with optional propagation models.

        :param base_station_height: Hauteur de la passerelle (m).
        :param mobile_height: Hauteur de l'émetteur mobile (m).
        :param propagation_model: Nom du modèle de perte (``cost231``,
            ``okumura_hata`` ou ``3d``).
        :param fading: Type de fading (``rayleigh`` ou ``rician``).
        :param rician_k: Facteur ``K`` pour le fading rician.
        :param terrain: Type de terrain pour Okumura‑Hata.
        :param weather_loss_dB_per_km: Atténuation météo moyenne en dB/km.
        :param weather_loss_std_dB_per_km: Variation temporelle de cette
            atténuation.
        :param weather_correlation: Corrélation temporelle des variations
            météo (par défaut ``fading_correlation``).
        :param kwargs: Paramètres transmis au constructeur de :class:`Channel`.
        :param fine_fading_std: Écart-type du fading temporel fin.
        :param fading_correlation: Facteur de corrélation temporelle.
        :param variable_noise_std: Variation lente du bruit thermique.
        :param advanced_capture: Active un mode de capture avancée.
        :param frequency_offset_hz: Décalage fréquentiel moyen entre émetteur et
            récepteur (Hz).
        :param freq_offset_std_hz: Variation temporelle (écart-type en Hz) du
            décalage fréquentiel.
        :param sync_offset_s: Décalage temporel moyen (s) pour le calcul des
            collisions partielles.
        :param sync_offset_std_s: Variation temporelle (écart-type en s) du
            décalage temporel.
        :param dev_frequency_offset_hz: Décalage fréquentiel propre à
            l'appareil émetteur (Hz).
        :param dev_freq_offset_std_hz: Variation temporelle du décallage
            fréquentiel propre à l'appareil.
        :param temperature_std_K: Variation de température (K) pour moduler le
            bruit thermique.
        :param pa_non_linearity_dB: Décalage moyen (dB) dû à la non‑linéarité
            de l'amplificateur de puissance.
        :param pa_non_linearity_std_dB: Variation temporelle (dB) de la
            non‑linéarité PA.
        :param pa_non_linearity_curve: Courbe polynomiale ``(a, b, c)`` appliquée
            à la puissance TX pour modéliser une non‑linéarité plus complexe.
        :param pa_distortion_std_dB: Variation aléatoire (dB) due aux
            imperfections de l'amplificateur de puissance.
        :param humidity_percent: Humidité relative moyenne (0‑100 %).
        :param humidity_std_percent: Variation temporelle de l'humidité
            relative.
        :param humidity_noise_coeff_dB: Coefficient appliqué au pourcentage
            d'humidité pour moduler le bruit (dB).
        :param phase_noise_std_dB: Bruit de phase appliqué au SNR (écart-type en
            dB).
        :param clock_jitter_std_s: Gigue d'horloge (s) appliquée au décalage
            temporel à chaque calcul.
        :param frontend_filter_order: Ordre du filtre passe-bande simulé.
        :param frontend_filter_bw: Largeur de bande du filtre (Hz).
        :param multipath_paths: Nombre de trajets multipath à simuler.
        :param indoor_n_floors: Nombre d'étages à traverser pour le modèle
            ``itu_indoor``.
        :param indoor_floor_loss_dB: Perte moyenne par étage pour
            ``itu_indoor``.
        :param obstacle_map: Maillage décrivant les pertes additionnelles (dB)
            sur le trajet. Une valeur négative bloque totalement la liaison.
        :param map_area_size: Taille (mètres) correspondant au maillage pour
            le calcul des obstacles.
        :param obstacle_height_map: Carte des hauteurs (m) obstruant la ligne
            de visée. Si la trajectoire passe sous une hauteur positive, la
            pénalité ``default_obstacle_dB`` ou la valeur de ``obstacle_map`` est
            appliquée.
        :param obstacle_map_file: Fichier JSON ou texte décrivant ``obstacle_map``.
        :param obstacle_height_map_file: Fichier JSON ou texte décrivant
            ``obstacle_height_map``.
        :param default_obstacle_dB: Pénalité par défaut en dB lorsqu'un obstacle
            est rencontré sans valeur explicite dans ``obstacle_map``.
        """

        from .channel import Channel

        self.base = Channel(
            fine_fading_std=fine_fading_std,
            fading_correlation=fading_correlation,
            variable_noise_std=variable_noise_std,
            advanced_capture=advanced_capture,
            humidity_percent=humidity_percent,
            humidity_std_percent=humidity_std_percent,
            humidity_noise_coeff_dB=humidity_noise_coeff_dB,
            frontend_filter_order=frontend_filter_order,
            frontend_filter_bw=frontend_filter_bw,
            **kwargs,
        )
        self.base_station_height = base_station_height
        self.mobile_height = mobile_height
        self.propagation_model = propagation_model
        self.fading = fading
        self.rician_k = rician_k
        self.fading_model = _CorrelatedFading(
            fading, rician_k, fading_correlation, paths=multipath_paths
        )
        self.terrain = terrain.lower()
        self.weather_loss_dB_per_km = weather_loss_dB_per_km
        if weather_correlation is None:
            weather_correlation = fading_correlation
        self._weather_loss = _CorrelatedValue(
            weather_loss_dB_per_km, weather_loss_std_dB_per_km, weather_correlation
        )
        self.frequency_offset_hz = frequency_offset_hz
        self.freq_offset_std_hz = freq_offset_std_hz
        self.sync_offset_s = sync_offset_s
        self.sync_offset_std_s = sync_offset_std_s
        self.clock_jitter_std_s = clock_jitter_std_s
        self._freq_offset = _CorrelatedValue(
            frequency_offset_hz, freq_offset_std_hz, fading_correlation
        )
        self._sync_offset = _CorrelatedValue(
            sync_offset_s, sync_offset_std_s, fading_correlation
        )
        self._tx_power_var = _CorrelatedValue(0.0, self.base.tx_power_std, fading_correlation)
        self._dev_freq_offset = _CorrelatedValue(
            dev_frequency_offset_hz,
            dev_freq_offset_std_hz,
            fading_correlation,
        )
        self.indoor_n_floors = int(indoor_n_floors)
        self.indoor_floor_loss_dB = float(indoor_floor_loss_dB)
        self._temperature = _CorrelatedValue(
            self.base.omnet.temperature_K,
            temperature_std_K,
            fading_correlation,
        )
        self._pa_nl = _CorrelatedValue(
            pa_non_linearity_dB,
            pa_non_linearity_std_dB,
            fading_correlation,
        )
        self._pa_distortion = _CorrelatedValue(0.0, pa_distortion_std_dB, fading_correlation)
        self.pa_non_linearity_curve = pa_non_linearity_curve
        self._humidity = _CorrelatedValue(
            humidity_percent,
            humidity_std_percent,
            fading_correlation,
        )
        self._phase_noise = _CorrelatedValue(0.0, phase_noise_std_dB, fading_correlation)
        if obstacle_map is None and obstacle_map_file:
            from .map_loader import load_map
            obstacle_map = load_map(obstacle_map_file)
        if obstacle_height_map is None and obstacle_height_map_file:
            from .map_loader import load_map
            obstacle_height_map = load_map(obstacle_height_map_file)

        self.obstacle_map = obstacle_map
        self.obstacle_height_map = obstacle_height_map
        self.default_obstacle_dB = float(default_obstacle_dB)
        self.map_area_size = map_area_size
        if obstacle_map or obstacle_height_map:
            self._rows = len(obstacle_map or obstacle_height_map)
            self._cols = len((obstacle_map or obstacle_height_map)[0]) if self._rows else 0
        else:
            self._rows = self._cols = 0

    # ------------------------------------------------------------------
    # Propagation models
    # ------------------------------------------------------------------
    def path_loss(self, distance: float, height_diff: float | None = None) -> float:
        """Return path loss in dB for the selected model."""
        d = distance
        if self.propagation_model == "3d":
            if height_diff is None:
                height_diff = self.base_station_height - self.mobile_height
            d = math.sqrt(distance ** 2 + height_diff ** 2)
            loss = self.base.path_loss(d)
        elif self.propagation_model == "cost231":
            loss = self._cost231_loss(distance)
        elif self.propagation_model == "okumura_hata":
            loss = self._okumura_hata_loss(distance)
        elif self.propagation_model == "itu_indoor":
            loss = self._itu_indoor_loss(distance)
        else:
            loss = self.base.path_loss(distance)

        if self.weather_loss_dB_per_km or self._weather_loss.std > 0.0:
            loss_per_km = self._weather_loss.sample()
            loss += loss_per_km * (max(d, 1.0) / 1000.0)
        return loss

    # ------------------------------------------------------------------
    def _obstacle_loss(
        self,
        tx_pos: tuple[float, float, float] | tuple[float, float],
        rx_pos: tuple[float, float, float] | tuple[float, float],
    ) -> float:
        """Compute additional loss due to obstacles between two points."""
        if (
            not self.obstacle_map
            and not self.obstacle_height_map
            or not self.map_area_size
            or self._rows == 0
        ):
            return 0.0
        visited: set[tuple[int, int]] = set()
        steps = max(self._cols, self._rows)
        loss = 0.0
        for i in range(steps + 1):
            t = i / steps
            x = tx_pos[0] + (rx_pos[0] - tx_pos[0]) * t
            y = tx_pos[1] + (rx_pos[1] - tx_pos[1]) * t
            if len(tx_pos) >= 3 and len(rx_pos) >= 3:
                z = tx_pos[2] + (rx_pos[2] - tx_pos[2]) * t
            else:
                z = 0.0
            cx = int(x / self.map_area_size * self._cols)
            cy = int(y / self.map_area_size * self._rows)
            cx = min(max(cx, 0), self._cols - 1)
            cy = min(max(cy, 0), self._rows - 1)
            cell = (cy, cx)
            if cell in visited:
                continue
            visited.add(cell)
            obstacle_val = None
            if self.obstacle_map:
                obstacle_val = float(self.obstacle_map[cy][cx])
            height = None
            if self.obstacle_height_map:
                height = float(self.obstacle_height_map[cy][cx])
            if height is not None and height > 0.0 and z <= height:
                val = (
                    obstacle_val
                    if obstacle_val is not None
                    else self.default_obstacle_dB
                )
                if val < 0:
                    return float("inf")
                if val > 0:
                    loss += val
                continue
            if obstacle_val is not None:
                if obstacle_val < 0:
                    return float("inf")
                if obstacle_val > 0:
                    loss += obstacle_val
        return loss

    def _cost231_loss(self, distance: float) -> float:
        distance_km = max(distance / 1000.0, 1e-3)
        freq_mhz = self.base.frequency_hz / 1e6
        a_hm = (
            (1.1 * math.log10(freq_mhz) - 0.7) * self.mobile_height
            - (1.56 * math.log10(freq_mhz) - 0.8)
        )
        return (
            46.3
            + 33.9 * math.log10(freq_mhz)
            - 13.82 * math.log10(self.base_station_height)
            - a_hm
            + (44.9 - 6.55 * math.log10(self.base_station_height))
            * math.log10(distance_km)
        )

    def _okumura_hata_loss(self, distance: float) -> float:
        distance_km = max(distance / 1000.0, 1e-3)
        freq_mhz = self.base.frequency_hz / 1e6
        hb = self.base_station_height
        hm = self.mobile_height
        a_hm = (
            (1.1 * math.log10(freq_mhz) - 0.7) * hm
            - (1.56 * math.log10(freq_mhz) - 0.8)
        )
        pl = (
            69.55
            + 26.16 * math.log10(freq_mhz)
            - 13.82 * math.log10(hb)
            - a_hm
            + (44.9 - 6.55 * math.log10(hb)) * math.log10(distance_km)
        )
        if self.terrain == "suburban":
            pl -= 2 * (math.log10(freq_mhz / 28.0)) ** 2 - 5.4
        elif self.terrain == "open":
            pl -= 4.78 * (math.log10(freq_mhz)) ** 2 - 18.33 * math.log10(freq_mhz) + 40.94
        return pl

    def _itu_indoor_loss(self, distance: float) -> float:
        """Simple ITU indoor path loss model."""
        d = max(distance, 1.0)
        freq_mhz = self.base.frequency_hz / 1e6
        n = 30.0
        loss = 20 * math.log10(freq_mhz) + n * math.log10(d) + 28
        loss += self.indoor_floor_loss_dB * max(self.indoor_n_floors - 1, 0)
        return loss

    # ------------------------------------------------------------------
    def compute_rssi(
        self,
        tx_power_dBm: float,
        distance: float,
        sf: int | None = None,
        *,
        tx_pos: tuple[float, float] | tuple[float, float, float] | None = None,
        rx_pos: tuple[float, float] | tuple[float, float, float] | None = None,
        freq_offset_hz: float | None = None,
        sync_offset_s: float | None = None,
    ) -> tuple[float, float]:
        """Return RSSI and SNR for the advanced channel.

        Additional optional frequency and timing offsets can be supplied to
        emulate partial collisions or de-synchronised transmissions. When not
        specified, time‑varying offsets are drawn from correlated distributions
        configured at construction.
        ``tx_pos`` and ``rx_pos`` may include an optional altitude as third
        coordinate to interact with ``obstacle_height_map``.
        """
        if freq_offset_hz is None:
            freq_offset_hz = self._freq_offset.sample()
        # Include time-varying frequency drift
        freq_offset_hz += self.base.omnet.frequency_drift()
        freq_offset_hz += self._dev_freq_offset.sample()
        if sync_offset_s is None:
            sync_offset_s = self._sync_offset.sample()
        # Include short-term clock jitter
        sync_offset_s += self.base.omnet.clock_drift()
        if self.clock_jitter_std_s > 0.0:
            sync_offset_s += random.gauss(0.0, self.clock_jitter_std_s)

        height_diff = None
        if tx_pos is not None and rx_pos is not None and self.propagation_model == "3d":
            if len(tx_pos) >= 3 and len(rx_pos) >= 3:
                height_diff = tx_pos[2] - rx_pos[2]
        loss = self.path_loss(distance, height_diff)
        if tx_pos is not None and rx_pos is not None:
            extra = self._obstacle_loss(tx_pos, rx_pos)
            if extra == float("inf"):
                return -float("inf"), -float("inf")
            loss += extra
        if self.base.shadowing_std > 0:
            loss += random.gauss(0, self.base.shadowing_std)

        tx_power_dBm += self._pa_nl.sample()
        tx_power_dBm += self._pa_distortion.sample()
        if self.pa_non_linearity_curve:
            a, b, c = self.pa_non_linearity_curve
            tx_power_dBm += a * tx_power_dBm ** 2 + b * tx_power_dBm + c
        rssi = (
            tx_power_dBm
            + self.base.tx_antenna_gain_dB
            + self.base.rx_antenna_gain_dB
            - loss
            - self.base.cable_loss_dB
        )
        if self.base.tx_power_std > 0:
            rssi += self._tx_power_var.sample()
        if self.base.fast_fading_std > 0:
            rssi += random.gauss(0, self.base.fast_fading_std)
        if self.base.time_variation_std > 0:
            rssi += random.gauss(0, self.base.time_variation_std)
        rssi += self.base.omnet.fine_fading()

        temperature = self._temperature.sample()
        model = (
            self.base.omnet_phy.model
            if getattr(self.base, "omnet_phy", None)
            else self.base.omnet
        )
        original_temp = model.temperature_K
        model.temperature_K = temperature
        thermal = model.thermal_noise_dBm(self.base.bandwidth)
        model.temperature_K = original_temp
        noise = thermal + self.base.noise_figure_dB + self.base.interference_dB
        noise += self.base.humidity_noise_coeff_dB * (self._humidity.sample() / 100.0)
        if self.base.noise_floor_std > 0:
            noise += random.gauss(0, self.base.noise_floor_std)
        noise += model.noise_variation()
        rssi += self.fading_model.sample_db()

        # Additional penalty if transmissions are not perfectly aligned
        penalty = self._interference_penalty_db(freq_offset_hz, sync_offset_s, sf)
        noise += penalty

        snr = rssi - noise - abs(self._phase_noise.sample())
        if sf is not None:
            snr += 10 * math.log10(2 ** sf)
        return rssi, snr

    # ------------------------------------------------------------------
    def _interference_penalty_db(
        self,
        freq_offset_hz: float,
        sync_offset_s: float,
        sf: int | None,
    ) -> float:
        """Simple penalty model for imperfect alignment."""
        bw = self.base.bandwidth
        freq_factor = abs(freq_offset_hz) / (bw / 2.0)
        if sf is not None:
            symbol_time = (2 ** sf) / bw
        else:
            symbol_time = 1.0 / bw
        time_factor = abs(sync_offset_s) / symbol_time
        if freq_factor >= 1.0 and time_factor >= 1.0:
            return float("inf")
        return 10 * math.log10(1.0 + freq_factor ** 2 + time_factor ** 2)
