import math
import os
from pathlib import Path
import re
import warnings

import numpy as np

from traffic.numpy_compat import create_generator

from .obstacle_loss import ObstacleLoss
from .omnet_model import OmnetModel
from .propagation_cache import PropagationCache


def path_loss_hata_okumura(distance: float, k1: float, k2: float) -> float:
    """Return Hata-Okumura path loss in dB for ``distance`` in meters.

    The formula follows the model used in FLoRa where the loss is
    ``K1 + K2 * log10(d/1000)``. The random shadowing component ``σ`` is
    handled separately by :meth:`Channel.compute_rssi`.
    """

    if distance <= 0:
        raise ValueError("distance must be > 0")
    d_km = distance / 1000.0
    return k1 + k2 * math.log10(d_km)


def path_loss_oulu(
    distance: float,
    b: float,
    n: float,
    d0: float,
    antenna_gain: float,
) -> float:
    """Return Oulu path loss in dB for ``distance`` in meters.

    Implements ``B + 10 * n * log10(d/d0) - antennaGain`` as in FLoRa.
    The Gaussian shadowing term ``σ`` is applied outside this function.
    """

    if distance <= 0:
        raise ValueError("distance must be > 0")
    return b + 10 * n * math.log10(distance / d0) - antenna_gain

class _CorrelatedValue:
    """Correlated random walk used for optional impairments."""

    def __init__(
        self,
        mean: float,
        std: float,
        correlation: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.mean = mean
        self.std = std
        self.corr = correlation
        self.value = mean
        self.rng = rng or create_generator()

    def sample(self) -> float:
        self.value = self.corr * self.value + (1.0 - self.corr) * self.mean
        if self.std > 0.0:
            self.value += self.rng.normal(0.0, self.std)
        return self.value

class Channel:
    """Représente le canal de propagation radio pour LoRa."""

    SPEED_OF_LIGHT = 299_792_458.0
    FLORA_ENERGY_DETECTION_DBM = -90.0
    ENV_PRESETS = {
        "urban": (2.08, 3.57, 127.41, 40.0),
        "suburban": (2.32, 7.08, 128.95, 1000.0),
        "rural": (2.0, 2.0, 113.0, 1.0),
        "rural_long_range": (1.7, 1.5, 105.0, 100.0),
        # Parameters matching the FLoRa log-normal shadowing model
        "flora": (2.08, 3.57, 127.41, 40.0),
        "flora_oulu": (2.32, 7.08, 128.95, 1000.0),
        "flora_hata": (2.08, 3.57, 127.5, 40.0),
        "very_long_range": (2.08, 3.57, 127.41, 40.0),
        # Additional presets for denser or indoor environments
        "urban_dense": (3.0, 8.0, 127.41, 40.0),
        "indoor": (3.5, 7.0, 127.41, 40.0),
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

    FLORA_SENSITIVITY = {
        6: {125000: -121, 250000: -118, 500000: -111},
        7: {125000: -124, 250000: -122, 500000: -116},
        8: {125000: -127, 250000: -125, 500000: -119},
        9: {125000: -130, 250000: -128, 500000: -122},
        10: {125000: -133, 250000: -130, 500000: -125},
        11: {125000: -135, 250000: -132, 500000: -128},
        12: {125000: -137, 250000: -135, 500000: -129},
    }

    @classmethod
    def flora_energy_threshold(cls, bandwidth: float) -> float:
        """Return the most permissive FLoRa detection threshold for ``bandwidth``."""

        bw = int(bandwidth)
        candidates = [
            tables.get(bw)
            for tables in cls.FLORA_SENSITIVITY.values()
            if bw in tables
        ]
        if candidates:
            return min(candidates)
        return cls.FLORA_ENERGY_DETECTION_DBM

    @classmethod
    def flora_detection_threshold(cls, sf: int, bandwidth: float) -> float:
        """Return sensitivity in dBm for ``sf`` and ``bandwidth``.

        The values come from :data:`FLORA_SENSITIVITY` to mirror the
        detection logic used in FLoRa. If the pair ``(sf, bandwidth)`` is not
        present in the table the classical ``-110`` dBm threshold is
        returned.
        """

        return cls.FLORA_SENSITIVITY.get(sf, {}).get(int(bandwidth), -110.0)

    @staticmethod
    def parse_flora_noise_table(path: str | os.PathLike) -> dict[int, dict[int, float]]:
        """Parse LoRaAnalogModel.cc to load exact noise values."""
        text = open(path, "r").read()
        table: dict[int, dict[int, float]] = {}
        current_sf: int | None = None
        for line in text.splitlines():
            m_sf = re.search(r"getLoRaSF\(\) == (\d+)", line)
            if m_sf:
                current_sf = int(m_sf.group(1))
                table[current_sf] = {}
                continue
            if current_sf is not None:
                m_bw = re.search(r"getLoRaBW\(\) == Hz\((\d+)\).*dBmW2mW\((-?\d+)\)", line)
                if m_bw:
                    bw = int(m_bw.group(1))
                    val = int(m_bw.group(2))
                    table[current_sf][bw] = val
        return table

    def __init__(
        self,
        frequency_hz: float = 868e6,
        path_loss_exp: float = 2.08,
        shadowing_std: float | None = None,
        path_loss_d0: float | None = None,
        reference_distance: float = 1.0,
        hata_k1: float = 127.5,
        hata_k2: float = 35.2,
        oulu_b: float = 128.95,
        oulu_n: float = 2.32,
        oulu_d0: float = 1000.0,
        oulu_antenna_gain: float = 2.0,
        fast_fading_std: float = 0.0,
        multipath_taps: int = 1,
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
        flora_capture: bool = False,
        phy_model: str = "omnet",
        flora_loss_model: str = "lognorm",
        flora_per_model: str | None = None,
        qos_path_loss_exp: float | None = None,
        system_loss_dB: float = 0.0,
        rssi_offset_dB: float = 0.0,
        snr_offset_dB: float = 0.0,
        frequency_offset_hz: float = 0.0,
        freq_offset_std_hz: float = 0.0,
        sync_offset_s: float = 0.0,
        sync_offset_std_s: float = 0.0,
        dev_frequency_offset_hz: float = 0.0,
        dev_freq_offset_std_hz: float = 0.0,
        freq_drift_std_hz: float = 0.0,
        clock_drift_std_s: float = 0.0,
        clock_jitter_std_s: float = 0.0,
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
        pa_ramp_up_s: float = 0.0,
        pa_ramp_down_s: float = 0.0,
        impulsive_noise_prob: float = 0.0,
        impulsive_noise_dB: float = 0.0,
        adjacent_interference_dB: float = 0.0,
        use_snir: bool = True,
        snir_fading_std: float | None = None,
        marginal_snir_margin_db: float = 1.5,
        marginal_snir_drop_prob: float = 0.25,
        snir_window: str | float | None = None,
        use_flora_curves: bool = False,
        residual_collision_prob: float | None = None,
        residual_collision_load_scale: float = 4.0,
        baseline_loss_rate: float | None = None,
        baseline_collision_rate: float | None = None,
        snir_off_noise_prob: float = 0.01,
        tx_current_a: float = 0.0,
        rx_current_a: float = 0.0,
        idle_current_a: float = 0.0,
        voltage_v: float = 3.3,
        flora_noise_path: str | os.PathLike | None = None,
        obstacle_map: str | os.PathLike | None = None,
        obstacle_loss: ObstacleLoss | None = None,
        propagation_cache: PropagationCache | None = None,
        *,
        processing_gain: bool = False,
        bandwidth: float = 125e3,
        coding_rate: int = 1,
        capture_threshold_dB: float = 6.0,
        capture_window_symbols: int = 5,
        tx_power_std: float = 0.0,
        interference_dB: float = 0.0,
        detection_threshold_dBm: float = -float("inf"),
        energy_detection_dBm: float = -float("inf"),
        sensitivity_margin_dB: float = 0.0,
        band_interference: list[tuple[float, float, float]] | None = None,
        environment: str | None = None,
        region: str | None = None,
        channel_index: int = 0,
        orthogonal_sf: bool = True,
        alpha_isf: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialise le canal radio avec paramètres de propagation.

        :param frequency_hz: Fréquence en Hz (par défaut 868 MHz).
        :param path_loss_exp: Exposant de perte de parcours (log-distance).
        :param shadowing_std: Écart-type du shadowing (variations aléatoires en dB), 0 pour ignorer.
        :param fast_fading_std: Variation rapide de l'amplitude (dB) pour simuler le fading multipath.
        :param multipath_taps: Nombre de trajets multipath additionnels simulés.
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
        :param capture_window_symbols: Nombre de symboles de préambule requis
            avant qu'un paquet plus fort puisse capturer la réception.
        :param tx_power_std: Écart-type de la variation aléatoire de puissance TX.
        :param interference_dB: Bruit supplémentaire moyen dû aux interférences.
        :param detection_threshold_dBm: RSSI minimal détectable (dBm). Les
            signaux plus faibles sont ignorés.
        :param energy_detection_dBm: Seuil de détection d'énergie (dBm).
            FLoRa applique −90 dBm par défaut afin d'ignorer les canaux trop
            silencieux avant d'engager la chaîne de réception.
        :param sensitivity_margin_dB: Marge ajoutée aux seuils de détection
            FLoRa pour ajuster la sensibilité (dB).
        :param band_interference: Liste optionnelle de tuples ``(freq, bw, dB)``
            décrivant des brouilleurs sélectifs. ``freq`` est la fréquence
            centrale en Hz, ``bw`` la largeur de bande et ``dB`` la puissance
            ajoutée au bruit lorsque le canal chevauche la bande.
        :param fine_fading_std: Écart-type du fading temporel fin.
        :param fading_correlation: Facteur de corrélation temporelle pour le
            fading et le bruit variable.
        :param variable_noise_std: Variation lente du bruit thermique en dB.
        :param advanced_capture: Active un mode de capture inspiré de FLoRa.
        :param flora_capture: Utilise la logique de collision FLoRa dans
            :class:`OmnetPHY`.
        :param phy_model: "omnet" (par défaut) pour utiliser le module OMNeT++.
            Le mode "omnet_full" reprend les équations complètes de
            ``LoRaAnalogModel`` afin de calculer RSSI et SNR avec les mêmes
            variations temporelles et la sélectivité de canal.
        :param flora_loss_model: Variante d'atténuation FLoRa à utiliser
            ("lognorm", "oulu" ou "hata").
        :param flora_per_model: Modèle PER à appliquer dans les modes FLoRa.
            La valeur "logistic" correspond à la sigmoïde OMNeT++ et reste
            imposée par défaut dès que ``phy_model`` commence par "omnet" ou
            "flora", ou que ``use_flora_curves`` est activé. Les autres
            valeurs ("croce", "none" …) sont acceptées mais déclenchent un
            avertissement car elles s'écartent des traces FLoRa historiques.
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
        :param pa_ramp_up_s: Temps de montée du PA simulé (s).
        :param pa_ramp_down_s: Temps de descente du PA (s).
        :param impulsive_noise_prob: Probabilité d'ajouter du bruit impulsif à
            chaque calcul de bruit.
        :param impulsive_noise_dB: Amplitude du bruit impulsif ajouté (dB).
        :param adjacent_interference_dB: Pénalité appliquée aux brouilleurs sur
            un canal adjacent (dB).
        :param use_snir: Si ``True``, applique le calcul SNIR en tenant compte
            des interférences explicites. Sinon, le calcul SNR historique est
            conservé.
        :param snir_window: Fenêtre utilisée pour évaluer l'interférence
            moyenne lorsque ``use_snir`` est activé. La valeur ``None`` ou
            ``"packet"`` conserve le comportement historique (moyenne sur
            l'airtime complet). ``"preamble"`` limite la mesure à la durée du
            préambule, ``"symbol"`` à un seul symbole ou bien une durée en
            secondes peut être fournie.
        :param residual_collision_prob: Probabilité maximale d'une collision
            résiduelle lorsque la charge atteint ``residual_collision_load_scale``.
            Si ``None``, une valeur minimale est injectée uniquement lorsque
            ``use_snir`` est désactivé.
        :param residual_collision_load_scale: Nombre de transmissions
            concurrentes servant de référence pour saturer l'effet de charge.
        :param baseline_loss_rate: Probabilité minimale de perte résiduelle
            appliquée même en l'absence de collisions explicites, et qui peut
            augmenter avec la charge. Si ``None``, une valeur minimale est
            utilisée lorsque ``use_snir`` est désactivé.
        :param baseline_collision_rate: Probabilité de perte additionnelle qui
            augmente avec la charge (jusqu'à ``residual_collision_load_scale``).
            Si ``None``, une valeur minimale est utilisée lorsque
            ``use_snir`` est désactivé.
        :param snir_off_noise_prob: Probabilité minimale de perte aléatoire
            appliquée lorsque ``use_snir`` est désactivé (bruit de fond).
        :param environment: Chaîne optionnelle pour charger un preset
            ("urban", "suburban", "rural", "rural_long_range", "flora",
            "flora_hata" ou "flora_oulu").
        :param region: Nom d'un plan de fréquences prédéfini ("EU868", "US915",
            etc.). S'il est fourni, ``frequency_hz`` est ignoré et remplacé par
            la fréquence correspondante du canal ``channel_index``.
        :param channel_index: Index du canal à utiliser dans le plan de la
            région choisie.
        :param orthogonal_sf: Si ``True``, les transmissions de SF différents
            n'interfèrent pas entre elles.
        :param alpha_isf: Coefficient appliqué aux interférences inter-SF lorsque
            ``orthogonal_sf`` est désactivé (0 signifie SF parfaitement
            orthogonaux).
        :param flora_noise_path: Chemin vers un fichier ``LoRaAnalogModel.cc``
            pour charger la table de bruit FLoRa.
        :param obstacle_map: Chemin vers une carte d'obstacles (GeoJSON ou raster)
            pour appliquer une perte additionnelle due aux bâtiments.
        :param obstacle_loss: Instance préconfigurée de :class:`ObstacleLoss`.
        """

        self.energy_detection_dBm = energy_detection_dBm

        user_shadowing_std = shadowing_std
        env_shadowing_std: float | None = None
        if environment is not None:
            env = environment.lower()
            if env not in self.ENV_PRESETS:
                raise ValueError(f"Unknown environment preset: {environment}")
            (
                path_loss_exp,
                env_shadowing_std,
                path_loss_d0,
                reference_distance,
            ) = self.ENV_PRESETS[env]
            self.environment = env
            if (
                self.energy_detection_dBm == -float("inf")
                and env.startswith("flora")
            ):
                self.energy_detection_dBm = self.flora_energy_threshold(bandwidth)
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

        self.rng = rng or create_generator()
        self.frequency_hz = frequency_hz
        self.wavelength_m = Channel.SPEED_OF_LIGHT / self.frequency_hz
        self.path_loss_exp = path_loss_exp
        self.qos_path_loss_exp = qos_path_loss_exp or path_loss_exp
        if user_shadowing_std is None:
            # If the environment preset provided a value use it, otherwise fall
            # back to the historical default of ``6.0`` dB.  Passing
            # ``shadowing_std`` explicitly must always take precedence over the
            # preset so that callers can disable randomness when comparing with
            # reference traces.
            env_shadowing = env_shadowing_std
            if env_shadowing is None:
                env_shadowing = 6.0
            self.shadowing_std = env_shadowing
        else:
            self.shadowing_std = float(user_shadowing_std)
        if path_loss_d0 is None:
            freq_mhz = self.frequency_hz / 1e6
            path_loss_d0 = 32.45 + 20 * math.log10(freq_mhz) - 60.0
        self.path_loss_d0 = path_loss_d0
        self.reference_distance = reference_distance
        self.hata_k1 = hata_k1
        self.hata_k2 = hata_k2
        self.oulu_b = oulu_b
        self.oulu_n = oulu_n
        self.oulu_d0 = oulu_d0
        self.oulu_antenna_gain = oulu_antenna_gain
        self.fast_fading_std = fast_fading_std
        self.multipath_taps = int(multipath_taps)
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
        self.sensitivity_margin_dB = sensitivity_margin_dB
        self.frequency_offset_hz = frequency_offset_hz
        self.freq_offset_std_hz = freq_offset_std_hz
        self.sync_offset_s = sync_offset_s
        self.sync_offset_std_s = sync_offset_std_s
        self.dev_frequency_offset_hz = dev_frequency_offset_hz
        self.dev_freq_offset_std_hz = dev_freq_offset_std_hz
        self.freq_drift_std_hz = freq_drift_std_hz
        self.clock_drift_std_s = clock_drift_std_s
        self.clock_jitter_std_s = clock_jitter_std_s
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
        self.pa_ramp_up_s = float(pa_ramp_up_s)
        self.pa_ramp_down_s = float(pa_ramp_down_s)
        self.impulsive_noise_prob = float(impulsive_noise_prob)
        self.impulsive_noise_dB = float(impulsive_noise_dB)
        self.adjacent_interference_dB = float(adjacent_interference_dB)
        self._use_flora_curves = False
        self._flora_per_model: str | None = "logistic"
        self.tx_current_a = float(tx_current_a)
        self.rx_current_a = float(rx_current_a)
        self.idle_current_a = float(idle_current_a)
        self.voltage_v = float(voltage_v)
        if snir_fading_std is None:
            self.snir_fading_std = 1.5 if use_snir else 0.0
        else:
            self.snir_fading_std = float(snir_fading_std)
        self.marginal_snir_margin_db = float(marginal_snir_margin_db)
        self.marginal_snir_drop_prob = float(marginal_snir_drop_prob)
        if residual_collision_prob is None:
            residual_collision_prob = 0.03 if use_snir is False else 0.0
        self.residual_collision_prob = float(residual_collision_prob)
        self.residual_collision_load_scale = float(residual_collision_load_scale)
        if baseline_loss_rate is None:
            baseline_loss_rate = 0.005 if use_snir is False else 0.001
        if baseline_collision_rate is None:
            baseline_collision_rate = 0.02 if use_snir is False else 0.005
        self.baseline_loss_rate = float(baseline_loss_rate)
        self.baseline_collision_rate = float(baseline_collision_rate)
        self.snir_off_noise_prob = float(snir_off_noise_prob)
        if flora_noise_path:
            self.flora_noise_table = self.parse_flora_noise_table(flora_noise_path)
        else:
            # Automatically look for FLoRa's noise definition file
            flora_cc = (
                Path(__file__).resolve().parents[2]
                / "flora-master"
                / "src"
                / "LoRaPhy"
                / "LoRaAnalogModel.cc"
            )
            if flora_cc.is_file():
                self.flora_noise_table = self.parse_flora_noise_table(flora_cc)
            else:
                self.flora_noise_table = self.FLORA_SENSITIVITY
        if obstacle_loss is not None:
            self.obstacle_loss = obstacle_loss
        elif obstacle_map is not None:
            self.obstacle_loss = ObstacleLoss.from_file(obstacle_map)
        else:
            self.obstacle_loss = None
        self.propagation_cache = propagation_cache
        self.omnet = OmnetModel(
            fine_fading_std,
            fading_correlation,
            variable_noise_std,
            freq_drift_std=freq_drift_std_hz,
            clock_drift_std=clock_drift_std_s,
            temperature_K=temperature_K,
            rng=self.rng,
        )
        self._temperature = _CorrelatedValue(
            temperature_K, temperature_std_K, fading_correlation, rng=self.rng
        )
        self._phase_noise = _CorrelatedValue(
            0.0, phase_noise_std_dB, fading_correlation, rng=self.rng
        )
        self._pa_nl = _CorrelatedValue(
            pa_non_linearity_dB, pa_non_linearity_std_dB, fading_correlation, rng=self.rng
        )
        self._humidity = _CorrelatedValue(
            humidity_percent, humidity_std_percent, fading_correlation, rng=self.rng
        )
        self._impulse = _CorrelatedValue(0.0, impulsive_noise_dB, fading_correlation, rng=self.rng)
        self.fine_fading_std = fine_fading_std
        self.fading_correlation = fading_correlation
        self.variable_noise_std = variable_noise_std
        self.advanced_capture = advanced_capture
        self.flora_capture = flora_capture
        self.phy_model = phy_model
        self.use_snir = use_snir
        if (
            self.energy_detection_dBm == -float("inf")
            and (
                bool(phy_model) and str(phy_model).startswith("flora")
                or flora_capture
                or use_flora_curves
            )
        ):
            self.energy_detection_dBm = self.flora_energy_threshold(bandwidth)
        self.flora_loss_model = flora_loss_model
        self.system_loss_dB = system_loss_dB
        self.rssi_offset_dB = rssi_offset_dB
        self.snr_offset_dB = snr_offset_dB
        self.processing_gain = bool(processing_gain)
        self.snir_window = snir_window

        # Paramètres LoRa (BW 125 kHz, CR 4/5, préambule 8, CRC activé)
        self.bandwidth = bandwidth
        self.coding_rate = coding_rate
        self.preamble_symbols = 8
        # Low Data Rate Optimization activée au-delà de ce SF
        self.low_data_rate_threshold = 11  # SF >= 11 -> Low Data Rate Optimization activée

        self._update_sensitivity()
        # Seuil de capture (différence de RSSI en dB pour qu'un signal plus fort capture la réception)
        self.capture_threshold_dB = capture_threshold_dB
        self.capture_window_symbols = int(capture_window_symbols)
        self.orthogonal_sf = orthogonal_sf
        self.alpha_isf = float(alpha_isf)
        self.last_rssi_dBm = 0.0
        self.last_noise_dBm = float("nan")
        self.last_filter_att_dB = 0.0
        self.last_interference_mW = 0.0
        self.current_interference_mW = 0.0
        # Track the effective carrier after oscillator drift.
        self.last_freq_hz = self.frequency_hz

        # Activer la logique FLoRa peut ajuster la fenêtre de capture.
        self.use_flora_curves = bool(use_flora_curves)
        self._set_flora_per_model(flora_per_model, explicit=flora_per_model is not None)

        if self.phy_model in ("omnet", "omnet_full"):
            from .omnet_phy import OmnetPHY
            self.omnet_phy = OmnetPHY(
                self,
                freq_offset_std_hz=self.freq_offset_std_hz,
                sync_offset_std_s=self.sync_offset_std_s,
                dev_frequency_offset_hz=dev_frequency_offset_hz,
                dev_freq_offset_std_hz=dev_freq_offset_std_hz,
                temperature_std_K=temperature_std_K,
                pa_non_linearity_dB=pa_non_linearity_dB,
                pa_non_linearity_std_dB=pa_non_linearity_std_dB,
                phase_noise_std_dB=phase_noise_std_dB,
                tx_start_delay_s=0.0,
                rx_start_delay_s=0.0,
                pa_ramp_up_s=pa_ramp_up_s,
                pa_ramp_down_s=pa_ramp_down_s,
                tx_current_a=self.tx_current_a,
                rx_current_a=self.rx_current_a,
                idle_current_a=self.idle_current_a,
                voltage_v=self.voltage_v,
                flora_capture=self.flora_capture,
                capture_window_symbols=self.capture_window_symbols,
            )
            self.flora_phy = None
            self.advanced_capture = True
        elif self.phy_model in ("flora", "flora_full") or self.use_flora_curves:
            from .flora_phy import FloraPHY
            self.flora_phy = FloraPHY(self, loss_model=self.flora_loss_model)
            self.omnet_phy = None
            self.advanced_capture = True
        elif self.phy_model == "flora_cpp":
            try:
                from .flora_cpp import FloraCppPHY
                self.flora_phy = FloraCppPHY()
            except Exception:
                from .flora_phy import FloraPHY
                self.flora_phy = FloraPHY(self, loss_model=self.flora_loss_model)
            self.omnet_phy = None
            self.advanced_capture = True
        else:
            self.omnet_phy = None
            self.flora_phy = None

        self._maybe_enforce_capture_window()

    @property
    def use_flora_curves(self) -> bool:
        return self._use_flora_curves

    @use_flora_curves.setter
    def use_flora_curves(self, value: bool) -> None:
        self._use_flora_curves = bool(value)
        self._maybe_enforce_capture_window()
        if (
            self._use_flora_curves
            and self.energy_detection_dBm == -float("inf")
        ):
            self.energy_detection_dBm = self.flora_energy_threshold(self.bandwidth)
        self._set_flora_per_model(self._flora_per_model, explicit=False)

    @property
    def flora_per_model(self) -> str | None:
        return self._flora_per_model

    @flora_per_model.setter
    def flora_per_model(self, value: str | None) -> None:
        self._set_flora_per_model(value, explicit=True)

    @property
    def capture_window_symbols(self) -> int:
        return self._capture_window_symbols

    @capture_window_symbols.setter
    def capture_window_symbols(self, value: int) -> None:
        self._capture_window_symbols = int(value)
        omnet_phy = getattr(self, "omnet_phy", None)
        if omnet_phy is not None:
            omnet_phy.capture_window_symbols = self._capture_window_symbols

    def _uses_flora_behaviour(self) -> bool:
        phy = getattr(self, "phy_model", "") or ""
        return (
            self._use_flora_curves
            or phy.startswith("flora")
            or getattr(self, "flora_capture", False)
        )

    def _set_flora_per_model(self, value: str | None, *, explicit: bool) -> None:
        model = "logistic" if value is None else str(value)
        phy = str(getattr(self, "phy_model", "") or "")
        enforce = phy.startswith("flora") or phy.startswith("omnet") or self.use_flora_curves
        if enforce and model != "logistic":
            if explicit:
                warnings.warn(
                    (
                        "Le mode PHY '%s' reproduit la sigmoïde FLoRa ; "
                        "flora_per_model='%s' peut s'écarter des traces OMNeT++."
                    )
                    % (phy or "inconnu", model),
                    RuntimeWarning,
                    stacklevel=3,
                )
            else:
                model = "logistic"
        self._flora_per_model = model

    def _maybe_enforce_capture_window(self) -> None:
        if self._uses_flora_behaviour() and self.capture_window_symbols < 6:
            self.capture_window_symbols = 6

    def noise_floor_dBm(self, freq_offset_hz: float = 0.0) -> float:
        """Retourne le niveau de bruit (dBm) pour la bande passante configurée.

        Le bruit peut varier autour de la valeur moyenne pour simuler un canal
        plus réaliste.
        """
        if self.omnet_phy:
            noise = self.omnet_phy.noise_floor()
            self.last_noise_dBm = noise
            return noise
        temp = self._temperature.sample()
        original = self.omnet.temperature_K
        self.omnet.temperature_K = temp
        eff_bw = min(self.bandwidth, self.frontend_filter_bw)
        # Thermal noise according to current temperature with a slowly varying
        # component matching FLoRa's LoRaAnalogModel.
        thermal = self.omnet.thermal_noise_dBm(eff_bw)
        thermal += self.omnet.noise_variation()
        self.omnet.temperature_K = original
        base = thermal + self.noise_figure_dB
        power = 10 ** (base / 10.0)
        if self.interference_dB != 0.0:
            power += 10 ** ((self.interference_dB - self._filter_attenuation_db(freq_offset_hz)) / 10.0)
        hum = self.humidity_noise_coeff_dB * (self._humidity.sample() / 100.0)
        if hum != 0.0:
            power += 10 ** ((base + hum) / 10.0) - 10 ** (base / 10.0)
        for f, bw, p in self.band_interference:
            half = (bw + self.bandwidth) / 2.0
            diff = abs(self.frequency_hz - f)
            if diff <= half:
                att = self._filter_attenuation_db(diff)
                power += 10 ** ((p - att) / 10.0)
            elif self.adjacent_interference_dB > 0 and diff <= half + self.bandwidth:
                val = max(p - self.adjacent_interference_dB, 0.0) - self._filter_attenuation_db(diff)
                power += 10 ** (val / 10.0)
        if self.noise_floor_std > 0:
            power *= 10 ** (self.rng.normal(0.0, self.noise_floor_std) / 10.0)
        if self.impulsive_noise_prob > 0.0:
            val = self._impulse.sample()
            if self.rng.random() < self.impulsive_noise_prob:
                power += 10 ** (val / 10.0)
        noise = 10 * math.log10(power)
        self.last_noise_dBm = noise
        return noise

    def _path_loss_uncached(self, distance: float) -> float:
        """Internal helper implementing the deterministic loss model."""
        if self.flora_loss_model == "hata":
            loss = path_loss_hata_okumura(distance, self.hata_k1, self.hata_k2)
        elif self.flora_loss_model == "oulu":
            loss = path_loss_oulu(
                distance,
                self.oulu_b,
                self.oulu_n,
                self.oulu_d0,
                self.oulu_antenna_gain,
            )
        else:
            d = max(distance, 1.0)
            loss = self.path_loss_d0 + 10 * self.path_loss_exp * math.log10(
                d / self.reference_distance
            )
        return loss + self.system_loss_dB

    def path_loss(self, distance: float) -> float:
        """Calcule la perte de parcours (en dB) pour une distance donnée (m)."""
        if self.omnet_phy:
            return self.omnet_phy.path_loss(distance)
        if distance <= 0:
            return 0.0
        if self.propagation_cache is not None:
            return self.propagation_cache.get(
                distance, lambda: self._path_loss_uncached(distance)
            )
        return self._path_loss_uncached(distance)

    def enable_propagation_cache(
        self,
        *,
        resolution: float = 1.0,
        max_entries: int | None = 10_000,
    ) -> None:
        """Enable an internal cache for deterministic propagation quantities."""

        self.propagation_cache = PropagationCache(
            resolution=resolution, max_entries=max_entries
        )

    def compute_rssi(
        self,
        tx_power_dBm: float,
        distance: float,
        sf: int | None = None,
        *,
        tx_pos: tuple[float, float] | tuple[float, float, float] | None = None,
        rx_pos: tuple[float, float] | tuple[float, float, float] | None = None,
        tx_angle: float | tuple[float, float] | None = None,
        rx_angle: float | tuple[float, float] | None = None,
        freq_offset_hz: float | None = None,
        sync_offset_s: float | None = None,
    ) -> tuple[float, float]:
        """Calcule le RSSI et le SNR attendus à une certaine distance.

        Un gain additionnel peut être appliqué si ``sf`` est renseigné pour
        représenter l'effet d'étalement de spectre LoRa.
        """
        if self.omnet_phy:
            # When using the lightweight OMNeT++-inspired PHY we still want to
            # account for optional obstacle attenuation.  The original
            # implementation returned immediately which skipped any loss
            # contribution from ``self.obstacle_loss``.  As a result the RSSI
            # was identical whether an obstacle map was provided or not.  We
            # compute the base RSSI/SNR using ``OmnetPHY`` and then subtract
            # the additional obstacle loss if positions are supplied.
            rssi, snr = self.omnet_phy.compute_rssi(
                tx_power_dBm,
                distance,
                sf,
                freq_offset_hz=freq_offset_hz,
                sync_offset_s=sync_offset_s,
            )
            if self.obstacle_loss is not None and tx_pos is not None and rx_pos is not None:
                att = self.obstacle_loss.loss(tx_pos, rx_pos)
                rssi -= att
                snr -= att
            return rssi, snr
        loss = self.path_loss(distance)
        if self.obstacle_loss is not None and tx_pos is not None and rx_pos is not None:
            loss += self.obstacle_loss.loss(tx_pos, rx_pos)
        if self.shadowing_std > 0:
            loss += self.rng.normal(0, self.shadowing_std)

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
            rssi += self.rng.normal(0, self.tx_power_std)
        if self.fast_fading_std > 0:
            rssi += self.rng.normal(0, self.fast_fading_std)
        if self.multipath_taps > 1:
            rssi += self._multipath_fading_db()
        if self.time_variation_std > 0:
            rssi += self.rng.normal(0, self.time_variation_std)
        rssi += self.omnet.fine_fading()
        if (
            tx_angle is not None
            and rx_angle is not None
            and tx_pos is not None
            and rx_pos is not None
        ):
            def _vec(pos1, pos2):
                dx = (pos2[0] - pos1[0])
                dy = (pos2[1] - pos1[1])
                dz1 = pos1[2] if len(pos1) >= 3 else 0.0
                dz2 = pos2[2] if len(pos2) >= 3 else 0.0
                dz = dz2 - dz1
                return dx, dy, dz

            dx, dy, dz = _vec(tx_pos, rx_pos)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > 0.0:
                los = (dx / dist, dy / dist, dz / dist)

                def _orient_vec(angle):
                    if isinstance(angle, (tuple, list)):
                        az, el = angle
                    else:
                        az = angle
                        el = 0.0
                    return (
                        math.cos(el) * math.cos(az),
                        math.cos(el) * math.sin(az),
                        math.sin(el),
                    )

                tx_vec = _orient_vec(tx_angle)
                rx_vec = _orient_vec(rx_angle)
                tx_dot = max(min(sum(a * b for a, b in zip(los, tx_vec)), 1.0), -1.0)
                rx_dot = max(min(sum(-a * b for a, b in zip(los, rx_vec)), 1.0), -1.0)
                tx_diff = math.acos(tx_dot)
                rx_diff = math.acos(rx_dot)
                rssi += self._directional_gain(tx_diff)
                rssi += self._directional_gain(rx_diff)
        rssi += self.rssi_offset_dB
        if freq_offset_hz is None:
            freq_offset_hz = self.frequency_offset_hz
        # Apply oscillator drift similar to FLoRa's analog model
        freq_offset_hz += self.omnet.frequency_drift()
        self.last_freq_hz = self.frequency_hz + freq_offset_hz
        if sync_offset_s is None:
            sync_offset_s = self.sync_offset_s
        sync_offset_s += self.omnet.clock_drift()
        if self.clock_jitter_std_s > 0.0:
            sync_offset_s += self.rng.normal(0.0, self.clock_jitter_std_s)

        attenuation = self._filter_attenuation_db(freq_offset_hz)
        rssi -= attenuation

        if sf is not None and (
            self.phy_model in ("flora_full", "flora_cpp")
            or self.use_flora_curves
        ):
            noise = self._flora_noise_dBm(sf)
        elif self.phy_model == "omnet_full" and sf is not None:
            noise = self._omnet_noise_dBm(sf, freq_offset_hz)
        else:
            noise = self.noise_floor_dBm(freq_offset_hz)
        # Store the effective noise floor so that downstream components (e.g.
        # capture logic or server metrics) reuse the same value.  Without this
        # update the attribute stayed at its default ``0.0`` whenever the
        # OMNeT++/FLoRa helpers were used, leading to grossly underestimated
        # SNRs in the validation matrix.
        self.last_noise_dBm = noise
        snr = rssi - noise + self.snr_offset_dB
        penalty = self._alignment_penalty_db(freq_offset_hz, sync_offset_s, sf)
        snr -= penalty
        snr -= abs(self._phase_noise.sample())
        if sf is not None and self.processing_gain:
            snr += 10 * math.log10(2 ** sf)
        self.last_rssi_dBm = rssi
        self.last_filter_att_dB = attenuation
        return rssi, snr

    def snir_window_duration(
        self, sf: int | None, packet_duration: float
    ) -> float | None:
        """Retourne la fenêtre (s) à utiliser pour évaluer l'interférence SNIR."""

        mode = self.snir_window
        if mode is None:
            return None

        if isinstance(mode, (int, float)):
            window = max(float(mode), 0.0)
            if window <= 0.0:
                return None
            return min(window, max(packet_duration, 0.0))

        normalized = str(mode).strip().lower()
        if normalized in {"packet", "full", "full_packet", "full-packet"}:
            return None

        bw = self.bandwidth if self.bandwidth > 0 else 1.0
        symbol_time = (2 ** sf) / bw if sf is not None else 1.0 / bw

        if normalized in {"symbol", "symbols", "per_symbol"}:
            return min(symbol_time, max(packet_duration, 0.0))

        if normalized in {"preamble", "preamble_only", "preamble-only"}:
            duration = max(symbol_time, symbol_time * self.preamble_symbols)
            return min(duration, max(packet_duration, 0.0))

        warnings.warn(
            "snir_window inconnu : %s (valeurs acceptées: packet, preamble, symbol ou durée en secondes)"
            % mode,
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def compute_snir(
        self,
        tx_power_dBm: float,
        distance: float,
        sf: int | None,
        interferers_mW: float | list[float] | tuple[float, ...] | np.ndarray,
        *,
        rssi_dBm: float | None = None,
        snr_dB: float | None = None,
        noise_dBm: float | None = None,
        **kwargs,
    ) -> tuple[float, float, float, float]:
        """Calcule RSSI, SNR, SNIR et bruit pour un lien.

        ``interferers_mW`` peut être un scalaire ou une collection; seules les
        valeurs positives sont prises en compte. Lorsque ``rssi_dBm`` et
        ``snr_dB`` sont fournis, le RSSI/SNR n'est pas recalculé et ``kwargs``
        est ignoré, ce qui permet de réutiliser un calcul préalable.
        """

        interference_mW: float
        if isinstance(interferers_mW, (list, tuple, np.ndarray)):
            interference_mW = float(sum(max(v, 0.0) for v in interferers_mW))
        else:
            interference_mW = max(float(interferers_mW), 0.0)

        if rssi_dBm is None or snr_dB is None or noise_dBm is None:
            rssi_dBm, snr_dB = self.compute_rssi(
                tx_power_dBm, distance, sf, **kwargs,
            )
            noise_dBm = self.last_noise_dBm
        else:
            self.last_rssi_dBm = rssi_dBm
            self.last_noise_dBm = noise_dBm

        signal_mW = 10 ** (rssi_dBm / 10.0)
        if self.snir_fading_std > 0.0:
            fading_signal = 10 ** (self._snir_fading_db() / 10.0)
            fading_interf = 10 ** (self._snir_fading_db() / 10.0)
            signal_mW *= fading_signal
            interference_mW *= fading_interf
        self.last_interference_mW = interference_mW
        noise_mW = 10 ** (noise_dBm / 10.0)
        total_noise = noise_mW + interference_mW
        if total_noise <= 0.0:
            snir_dB = float("inf")
        else:
            snir_dB = 10 * math.log10(signal_mW / total_noise)

        return rssi_dBm, snr_dB, snir_dB, noise_dBm

    @staticmethod
    def estimate_snir_db(
        rssi_dBm: float | None,
        noise_dBm: float | None,
        interference_mW: float | None = None,
    ) -> float | None:
        """Estime le SNIR (dB) à partir du RSSI, du bruit et de l'interférence."""

        if rssi_dBm is None or noise_dBm is None:
            return None
        interference = max(float(interference_mW or 0.0), 0.0)
        noise_mW = 10 ** (noise_dBm / 10.0)
        total_noise = noise_mW + interference
        if total_noise <= 0.0:
            return float("inf")
        signal_mW = 10 ** (rssi_dBm / 10.0)
        return 10 * math.log10(signal_mW / total_noise)

    def estimate_collision_snir_db(
        self,
        rssi_dBm: float | None,
        noise_dBm: float | None,
        interference_mW: float | None = None,
    ) -> float | None:
        """Estime le SNIR (dB) lors d'une collision."""

        if interference_mW is None:
            interference_mW = getattr(self, "last_interference_mW", 0.0)
        if rssi_dBm is None:
            rssi_dBm = getattr(self, "last_rssi_dBm", None)
        if noise_dBm is None:
            noise_dBm = getattr(self, "last_noise_dBm", None)
        snir = self.estimate_snir_db(rssi_dBm, noise_dBm, interference_mW)
        self.last_collision_snir_dB = snir
        return snir

    # ------------------------------------------------------------------
    def qos_path_gain(self, distance: float) -> float:
        """Return :math:`g(d)` for the QoS radio model."""

        if distance <= 0.0:
            raise ValueError("distance must be > 0")
        base = self.wavelength_m / (4.0 * math.pi * distance)
        if base <= 0.0:
            return 0.0
        return base ** self.qos_path_loss_exp

    # ------------------------------------------------------------------
    def connection_probability(
        self,
        distance: float,
        sf: int,
        tx_power_dBm: float,
        *,
        noise_dBm: float | None = None,
        snr_thresholds: dict[int, float] | None = None,
    ) -> float:
        """Return the connection probability ``H(d)`` for ``sf``."""

        gain = self.qos_path_gain(distance)
        if gain == 0.0:
            return 0.0
        if noise_dBm is None:
            if math.isfinite(self.last_noise_dBm):
                noise_dBm = self.last_noise_dBm
            else:
                noise_dBm = self.noise_floor_dBm()
        if snr_thresholds is None:
            snr_thresholds = self.SNR_THRESHOLDS
        if sf not in snr_thresholds:
            raise ValueError(f"unknown SF {sf} for SNR threshold")
        snr_linear = 10 ** (snr_thresholds[sf] / 10.0)
        noise_mw = 10 ** (noise_dBm / 10.0)
        tx_power_mw = 10 ** (tx_power_dBm / 10.0)
        exponent = -noise_mw * snr_linear / (tx_power_mw * gain)
        return math.exp(exponent)

    # ------------------------------------------------------------------
    def capture_probability(self, nu_j: float, delta: float) -> float:
        """Return the capture probability ``Q(d)`` of the QoS model."""

        if delta < 0.0:
            raise ValueError("delta must be >= 0")
        term = math.exp(-2.0 * nu_j)
        return term + (2.0 / (delta + 1.0)) * nu_j * term

    def packet_error_rate(
        self,
        snr: float,
        sf: int,
        payload_bytes: int = 20,
        ber_model: str = "croce",
    ) -> float:
        """Return PER based on the selected BER model."""
        if getattr(self, "flora_phy", None) and self.use_flora_curves:
            return self.flora_phy.packet_error_rate(snr, sf, payload_bytes)

        from .omnet_modulation import calculate_ber, calculate_ber_flora

        snir = 10 ** (snr / 10.0)
        if ber_model == "flora":
            ber = calculate_ber_flora(snir, sf, self.bandwidth)
        elif ber_model == "croce":
            ber = calculate_ber(snir, sf)
        else:
            raise ValueError(f"Unknown BER model: {ber_model}")

        ser = 1.0 - (1.0 - ber) ** sf
        n_bits = payload_bytes * 8
        per_bit = 1.0 - (1.0 - ber) ** n_bits
        n_sym = math.ceil(n_bits / sf)
        per_sym = 1.0 - (1.0 - ser) ** n_sym
        per = max(per_bit, per_sym)
        return min(max(per, 0.0), 1.0)

    def _multipath_fading_db(self) -> float:
        """Return a fading value in dB based on multiple Rayleigh paths."""
        if self.multipath_taps <= 1:
            return 0.0
        i = sum(self.rng.normal(0.0, 1.0) for _ in range(self.multipath_taps))
        q = sum(self.rng.normal(0.0, 1.0) for _ in range(self.multipath_taps))
        amp = math.sqrt(i * i + q * q) / math.sqrt(self.multipath_taps)
        return 20 * math.log10(max(amp, 1e-12))

    def _snir_fading_db(self) -> float:
        """Return a fading term dedicated to SNIR computations."""

        if self.snir_fading_std <= 0.0:
            return 0.0
        return float(self.rng.normal(0.0, self.snir_fading_std))

    def set_rng(self, rng) -> None:
        """Attach a new RNG to the channel and correlated helpers."""

        self.rng = rng
        for attr in ("_temperature", "_phase_noise", "_pa_nl", "_humidity", "_impulse"):
            helper = getattr(self, attr, None)
            if helper is not None:
                helper.rng = rng
        if getattr(self, "omnet", None) is not None:
            self.omnet.rng = rng

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
        # |H(jw)|^2 = 1 / (1 + (w/wc)^(2n)) for an n-th order Butterworth filter
        # Attenuation in dB is -10 log10(|H|^2) = 10 log10(1 + ratio^(2n))
        return 10 * math.log10(1.0 + ratio ** (2 * self.frontend_filter_order))

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
        penalty = 1.5 * (freq_factor ** 2 + time_factor ** 2)
        return 10 * math.log10(1.0 + penalty)

    def _directional_gain(self, angle_rad: float) -> float:
        """Simple cosine-squared antenna pattern."""
        gain = max(math.cos(angle_rad), 0.0) ** 2
        return 10 * math.log10(max(gain, 1e-3))

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

    # ------------------------------------------------------------------
    # Sensitivity computation
    # ------------------------------------------------------------------

    SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}

    def _flora_noise_dBm(self, sf: int) -> float:
        return self.flora_noise_table.get(sf, {}).get(int(self.bandwidth), -126.5)

    def _omnet_noise_dBm(self, sf: int, freq_offset_hz: float = 0.0) -> float:
        """Return noise level similar to LoRaAnalogModel with variations."""
        base = self._flora_noise_dBm(sf)
        thermal = -174 + 10 * math.log10(self.bandwidth) + self.noise_figure_dB
        delta = self.noise_floor_dBm(freq_offset_hz) - thermal
        return base + delta


    def _update_sensitivity(self) -> None:
        bw = min(self.bandwidth, self.frontend_filter_bw)
        noise = -174 + 10 * math.log10(bw) + self.noise_figure_dB
        self.sensitivity_dBm = {
            sf: noise + snr + self.sensitivity_margin_dB
            for sf, snr in self.SNR_THRESHOLDS.items()
        }
