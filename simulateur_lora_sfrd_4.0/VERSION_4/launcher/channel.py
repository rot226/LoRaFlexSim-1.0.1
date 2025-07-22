import math
import random
from .omnet_model import OmnetModel


class _CorrelatedValue:
    """Correlated random walk used for optional impairments."""

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

class Channel:
    """Représente le canal de propagation radio pour LoRa."""

    ENV_PRESETS = {
        "urban": (2.7, 6.0),
        "suburban": (2.3, 4.0),
        "rural": (2.0, 2.0),
        # Parameters matching the FLoRa log-normal shadowing model
        "flora": (2.7, 3.57),
        # Additional presets for denser or indoor environments
        "urban_dense": (3.0, 8.0),
        "indoor": (3.5, 7.0),
    }

    # Preset frequency plans for common regions
    REGION_CHANNELS: dict[str, list[float]] = {
        "EU868": [868.1e6, 868.3e6, 868.5e6],
        "US915": [902.3e6 + 200e3 * i for i in range(8)],
        "AU915": [915.2e6 + 200e3 * i for i in range(8)],
        # Additional presets for Asian regions
        "AS923": [923.2e6, 923.4e6, 923.6e6],
        "IN865": [865.1e6, 865.3e6, 865.5e6],
        "KR920": [920.9e6, 921.1e6, 921.3e6],
    }

    def __init__(
        self,
        frequency_hz: float = 868e6,
        path_loss_exp: float = 2.7,
        shadowing_std: float = 6.0,
        fast_fading_std: float = 0.0,
        cable_loss_dB: float = 0.0,
        tx_antenna_gain_dB: float = 0.0,
        rx_antenna_gain_dB: float = 0.0,
        time_variation_std: float = 0.0,
        receiver_noise_floor_dBm: float = -174.0,
        noise_figure_dB: float = 6.0,
        noise_floor_std: float = 0.0,
        # OMNeT++ inspired options
        fine_fading_std: float = 0.0,
        fading_correlation: float = 0.9,
        variable_noise_std: float = 0.0,
        advanced_capture: bool = False,
        phy_model: str = "omnet",
        system_loss_dB: float = 0.0,
        rssi_offset_dB: float = 0.0,
        snr_offset_dB: float = 0.0,
        frequency_offset_hz: float = 0.0,
        sync_offset_s: float = 0.0,
        freq_drift_std_hz: float = 0.0,
        clock_drift_std_s: float = 0.0,
        temperature_K: float = 290.0,
        temperature_std_K: float = 0.0,
        phase_noise_std_dB: float = 0.0,
        pa_non_linearity_dB: float = 0.0,
        pa_non_linearity_std_dB: float = 0.0,
        humidity_percent: float = 50.0,
        humidity_std_percent: float = 0.0,
        humidity_noise_coeff_dB: float = 0.0,
        frontend_filter_order: int = 0,
        frontend_filter_bw: float | None = None,
        *,
        bandwidth: float = 125e3,
        coding_rate: int = 1,
        capture_threshold_dB: float = 6.0,
        tx_power_std: float = 0.0,
        interference_dB: float = 0.0,
        detection_threshold_dBm: float = -float("inf"),
        band_interference: list[tuple[float, float, float]] | None = None,
        environment: str | None = None,
        region: str | None = None,
        channel_index: int = 0,
    ):
        """
        Initialise le canal radio avec paramètres de propagation.

        :param frequency_hz: Fréquence en Hz (par défaut 868 MHz).
        :param path_loss_exp: Exposant de perte de parcours (log-distance).
        :param shadowing_std: Écart-type du shadowing (variations aléatoires en dB), 0 pour ignorer.
        :param fast_fading_std: Variation rapide de l'amplitude (dB) pour simuler le fading multipath.
        :param cable_loss_dB: Pertes fixes dues au câble/connectique (dB).
        :param tx_antenna_gain_dB: Gain de l'antenne émettrice (dB).
        :param rx_antenna_gain_dB: Gain de l'antenne réceptrice (dB).
        :param time_variation_std: Écart-type d'une variation aléatoire
            appliquée au RSSI à chaque appel pour représenter un canal
            temporellement variable.
        :param receiver_noise_floor_dBm: Niveau de bruit thermique de référence (dBm/Hz).
        :param noise_figure_dB: Facteur de bruit ajouté par le récepteur (dB).
        :param noise_floor_std: Écart-type de la variation aléatoire du bruit
            (dB). Utile pour modéliser un canal plus dynamique.
        :param bandwidth: Largeur de bande LoRa (Hz).
        :param coding_rate: Index de code (0=4/5 … 4=4/8).
        :param capture_threshold_dB: Seuil de capture pour le décodage simultané.
        :param tx_power_std: Écart-type de la variation aléatoire de puissance TX.
        :param interference_dB: Bruit supplémentaire moyen dû aux interférences.
        :param detection_threshold_dBm: RSSI minimal détectable (dBm). Les
            signaux plus faibles sont ignorés.
        :param band_interference: Liste optionnelle de tuples ``(freq, bw, dB)``
            décrivant des brouilleurs sélectifs. ``freq`` est la fréquence
            centrale en Hz, ``bw`` la largeur de bande et ``dB`` la puissance
            ajoutée au bruit lorsque le canal chevauche la bande.
        :param fine_fading_std: Écart-type du fading temporel fin.
        :param fading_correlation: Facteur de corrélation temporelle pour le
            fading et le bruit variable.
        :param variable_noise_std: Variation lente du bruit thermique en dB.
        :param advanced_capture: Active un mode de capture inspiré de FLoRa.
        :param phy_model: "omnet" (par défaut) pour utiliser le module OMNeT++.
        :param system_loss_dB: Pertes fixes supplémentaires (par ex. pertes
            système) appliquées à la perte de parcours.
        :param rssi_offset_dB: Décalage appliqué au RSSI calculé (dB).
        :param snr_offset_dB: Décalage appliqué au SNR calculé (dB).
        :param frequency_offset_hz: Désalignement fréquentiel moyen appliqué si
            aucun offset n'est fourni lors du calcul.
        :param sync_offset_s: Front d'attaque temporel moyen appliqué si aucun
            offset n'est fourni.
        :param freq_drift_std_hz: Écart-type de la dérive de fréquence simulée.
        :param clock_drift_std_s: Écart-type de la dérive d'horloge simulée.
        :param temperature_K: Température utilisée pour le calcul du bruit
            thermique.
        :param temperature_std_K: Variation de température pour moduler le
            bruit thermique.
        :param phase_noise_std_dB: Écart-type du bruit de phase appliqué au SNR.
        :param pa_non_linearity_dB: Décalage moyen dû à la non‑linéarité PA.
        :param pa_non_linearity_std_dB: Variation temporelle de cette
            non‑linéarité.
        :param humidity_percent: Humidité relative moyenne (0‑100 %).
        :param humidity_std_percent: Variation temporelle de l'humidité
            relative.
        :param humidity_noise_coeff_dB: Coefficient appliqué au pourcentage
            d'humidité pour moduler le bruit (dB).
        :param frontend_filter_order: Ordre du filtre passe‑bande modélisant
            la sélectivité de la chaîne RF. ``0`` pour désactiver.
        :param frontend_filter_bw: Largeur de bande du filtre (Hz). Par défaut,
            la même valeur que ``bandwidth``.
        :param environment: Chaîne optionnelle pour charger un preset
            ("urban", "suburban" ou "rural").
        :param region: Nom d'un plan de fréquences prédéfini ("EU868", "US915",
            etc.). S'il est fourni, ``frequency_hz`` est ignoré et remplacé par
            la fréquence correspondante du canal ``channel_index``.
        :param channel_index: Index du canal à utiliser dans le plan de la
            région choisie.
        """

        if environment is not None:
            env = environment.lower()
            if env not in self.ENV_PRESETS:
                raise ValueError(f"Unknown environment preset: {environment}")
            path_loss_exp, shadowing_std = self.ENV_PRESETS[env]
            self.environment = env
        else:
            self.environment = None

        if region is not None:
            reg = region.upper()
            if reg not in self.REGION_CHANNELS:
                raise ValueError(f"Unknown region preset: {region}")
            freqs = self.REGION_CHANNELS[reg]
            if channel_index < 0 or channel_index >= len(freqs):
                raise ValueError("channel_index out of range for region preset")
            frequency_hz = freqs[channel_index]
            self.region = reg
            self.channel_index = channel_index
        else:
            self.region = None
            self.channel_index = channel_index

        self.frequency_hz = frequency_hz
        self.path_loss_exp = path_loss_exp
        self.shadowing_std = shadowing_std  # σ en dB (ex: 6.0 pour environnement urbain/suburbain)
        self.fast_fading_std = fast_fading_std
        self.cable_loss_dB = cable_loss_dB
        self.tx_antenna_gain_dB = tx_antenna_gain_dB
        self.rx_antenna_gain_dB = rx_antenna_gain_dB
        self.time_variation_std = time_variation_std
        self.receiver_noise_floor_dBm = receiver_noise_floor_dBm
        self.noise_figure_dB = noise_figure_dB
        self.noise_floor_std = noise_floor_std
        self.tx_power_std = tx_power_std
        self.interference_dB = interference_dB
        self.band_interference = list(band_interference or [])
        self.detection_threshold_dBm = detection_threshold_dBm
        self.frequency_offset_hz = frequency_offset_hz
        self.sync_offset_s = sync_offset_s
        self.freq_drift_std_hz = freq_drift_std_hz
        self.clock_drift_std_s = clock_drift_std_s
        self.temperature_K = temperature_K
        self.temperature_std_K = temperature_std_K
        self.phase_noise_std_dB = phase_noise_std_dB
        self.pa_non_linearity_dB = pa_non_linearity_dB
        self.pa_non_linearity_std_dB = pa_non_linearity_std_dB
        self.humidity_percent = humidity_percent
        self.humidity_std_percent = humidity_std_percent
        self.humidity_noise_coeff_dB = humidity_noise_coeff_dB
        self.frontend_filter_order = int(frontend_filter_order)
        self.frontend_filter_bw = float(frontend_filter_bw) if frontend_filter_bw is not None else bandwidth
        self.omnet = OmnetModel(
            fine_fading_std,
            fading_correlation,
            variable_noise_std,
            freq_drift_std=freq_drift_std_hz,
            clock_drift_std=clock_drift_std_s,
            temperature_K=temperature_K,
        )
        self._temperature = _CorrelatedValue(
            temperature_K, temperature_std_K, fading_correlation
        )
        self._phase_noise = _CorrelatedValue(0.0, phase_noise_std_dB, fading_correlation)
        self._pa_nl = _CorrelatedValue(
            pa_non_linearity_dB, pa_non_linearity_std_dB, fading_correlation
        )
        self._humidity = _CorrelatedValue(
            humidity_percent, humidity_std_percent, fading_correlation
        )
        self.fine_fading_std = fine_fading_std
        self.fading_correlation = fading_correlation
        self.variable_noise_std = variable_noise_std
        self.advanced_capture = advanced_capture
        self.phy_model = phy_model
        self.system_loss_dB = system_loss_dB
        self.rssi_offset_dB = rssi_offset_dB
        self.snr_offset_dB = snr_offset_dB

        # Paramètres LoRa (BW 125 kHz, CR 4/5, préambule 8, CRC activé)
        self.bandwidth = bandwidth
        self.coding_rate = coding_rate
        self.preamble_symbols = 8
        # Low Data Rate Optimization activée au-delà de ce SF
        self.low_data_rate_threshold = 11  # SF >= 11 -> Low Data Rate Optimization activée

        # Sensibilité approximative par SF (dBm) pour BW=125kHz, CR=4/5
        self.sensitivity_dBm = {
            7: -123,
            8: -126,
            9: -129,
            10: -132,
            11: -134.5,
            12: -137
        }
        # Seuil de capture (différence de RSSI en dB pour qu'un signal plus fort capture la réception)
        self.capture_threshold_dB = capture_threshold_dB

        if self.phy_model == "omnet":
            from .omnet_phy import OmnetPHY
            self.omnet_phy = OmnetPHY(
                self,
                temperature_std_K=temperature_std_K,
                pa_non_linearity_dB=pa_non_linearity_dB,
                pa_non_linearity_std_dB=pa_non_linearity_std_dB,
                phase_noise_std_dB=phase_noise_std_dB,
            )
            self.advanced_capture = True
        else:
            self.omnet_phy = None

    def noise_floor_dBm(self) -> float:
        """Retourne le niveau de bruit (dBm) pour la bande passante configurée.

        Le bruit peut varier autour de la valeur moyenne pour simuler un canal
        plus réaliste.
        """
        if self.omnet_phy:
            return self.omnet_phy.noise_floor()
        temp = self._temperature.sample()
        original = self.omnet.temperature_K
        self.omnet.temperature_K = temp
        thermal = self.omnet.variable_thermal_noise_dBm(self.bandwidth)
        self.omnet.temperature_K = original
        noise = thermal + self.noise_figure_dB + self.interference_dB
        noise += self.humidity_noise_coeff_dB * (self._humidity.sample() / 100.0)
        for f, bw, power in self.band_interference:
            half = (bw + self.bandwidth) / 2.0
            if abs(self.frequency_hz - f) <= half:
                noise += power
        if self.noise_floor_std > 0:
            noise += random.gauss(0, self.noise_floor_std)
        return noise

    def path_loss(self, distance: float) -> float:
        """Calcule la perte de parcours (en dB) pour une distance donnée (m)."""
        if self.omnet_phy:
            return self.omnet_phy.path_loss(distance)
        if distance <= 0:
            return 0.0
        freq_mhz = self.frequency_hz / 1e6
        pl_d0 = 32.45 + 20 * math.log10(freq_mhz) - 60.0
        pl = pl_d0 + 10 * self.path_loss_exp * math.log10(max(distance, 1.0) / 1.0)
        return pl + self.system_loss_dB

    def compute_rssi(
        self,
        tx_power_dBm: float,
        distance: float,
        sf: int | None = None,
        *,
        freq_offset_hz: float | None = None,
        sync_offset_s: float | None = None,
    ) -> tuple[float, float]:
        """Calcule le RSSI et le SNR attendus à une certaine distance.

        Un gain additionnel peut être appliqué si ``sf`` est renseigné pour
        représenter l'effet d'étalement de spectre LoRa.
        """
        if self.omnet_phy:
            return self.omnet_phy.compute_rssi(
                tx_power_dBm,
                distance,
                sf,
                freq_offset_hz=freq_offset_hz,
                sync_offset_s=sync_offset_s,
            )
        loss = self.path_loss(distance)
        if self.shadowing_std > 0:
            loss += random.gauss(0, self.shadowing_std)

        tx_power_dBm += self._pa_nl.sample()
        # RSSI = P_tx + gains antennes - pertes - pertes câble
        rssi = (
            tx_power_dBm
            + self.tx_antenna_gain_dB
            + self.rx_antenna_gain_dB
            - loss
            - self.cable_loss_dB
        )
        if self.tx_power_std > 0:
            rssi += random.gauss(0, self.tx_power_std)
        if self.fast_fading_std > 0:
            rssi += random.gauss(0, self.fast_fading_std)
        if self.time_variation_std > 0:
            rssi += random.gauss(0, self.time_variation_std)
        rssi += self.omnet.fine_fading()
        rssi += self.rssi_offset_dB
        if freq_offset_hz is None:
            freq_offset_hz = self.frequency_offset_hz
        freq_offset_hz += self.omnet.frequency_drift()
        if sync_offset_s is None:
            sync_offset_s = self.sync_offset_s
        sync_offset_s += self.omnet.clock_drift()

        rssi -= self._filter_attenuation_db(freq_offset_hz)

        snr = rssi - self.noise_floor_dBm() + self.snr_offset_dB
        penalty = self._alignment_penalty_db(freq_offset_hz, sync_offset_s, sf)
        snr -= penalty
        snr -= abs(self._phase_noise.sample())
        if sf is not None:
            snr += 10 * math.log10(2 ** sf)
        return rssi, snr

    def _filter_attenuation_db(self, freq_offset_hz: float) -> float:
        """Return attenuation due to the front-end filter."""
        if self.frontend_filter_order <= 0 or self.frontend_filter_bw <= 0:
            return 0.0
        fc = self.frontend_filter_bw / 2.0
        if fc <= 0.0:
            return 0.0
        ratio = abs(freq_offset_hz) / fc
        if ratio <= 0.0:
            return 0.0
        return 10 * self.frontend_filter_order * math.log10(1.0 + ratio ** 2)

    def _alignment_penalty_db(
        self, freq_offset_hz: float, sync_offset_s: float, sf: int | None
    ) -> float:
        """Return an SNR penalty due to imperfect alignment."""
        bw = self.bandwidth
        freq_factor = abs(freq_offset_hz) / (bw / 2.0)
        if sf is not None:
            symbol_time = (2 ** sf) / bw
        else:
            symbol_time = 1.0 / bw
        time_factor = abs(sync_offset_s) / symbol_time
        if freq_factor >= 1.0 and time_factor >= 1.0:
            return float("inf")
        return 10 * math.log10(1.0 + freq_factor ** 2 + time_factor ** 2)

    def airtime(self, sf: int, payload_size: int = 20) -> float:
        """Calcule l'airtime complet d'un paquet LoRa en secondes."""
        # Durée d'un symbole
        rs = self.bandwidth / (2 ** sf)
        ts = 1.0 / rs
        de = 1 if sf >= self.low_data_rate_threshold else 0
        cr_denom = self.coding_rate + 4
        numerator = 8 * payload_size - 4 * sf + 28 + 16 - 20 * 0
        denominator = 4 * (sf - 2 * de)
        n_payload = max(math.ceil(numerator / denominator), 0) * cr_denom + 8
        t_preamble = (self.preamble_symbols + 4.25) * ts
        t_payload = n_payload * ts
        return t_preamble + t_payload

    # ------------------------------------------------------------------
    # Helpers for region frequency presets
    # ------------------------------------------------------------------

    @classmethod
    def register_region(cls, name: str, frequencies: list[float]) -> None:
        """Register a new region frequency plan."""
        cls.REGION_CHANNELS[name.upper()] = list(frequencies)

    @classmethod
    def region_channels(cls, region: str, **kwargs) -> list["Channel"]:
        """Return a list of ``Channel`` objects for the given region preset."""
        reg = region.upper()
        if reg not in cls.REGION_CHANNELS:
            raise ValueError(f"Unknown region preset: {region}")
        return [cls(frequency_hz=f, region=reg, channel_index=i, **kwargs)
                for i, f in enumerate(cls.REGION_CHANNELS[reg])]
