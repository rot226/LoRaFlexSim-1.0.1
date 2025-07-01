import math
import random

class Channel:
    """Représente le canal de propagation radio pour LoRa."""

    def __init__(
        self,
        frequency_hz: float = 868e6,
        path_loss_exp: float = 2.7,
        shadowing_std: float = 6.0,
        fast_fading_std: float = 0.0,
        cable_loss_dB: float = 0.0,
        receiver_noise_floor_dBm: float = -174.0,
        noise_figure_dB: float = 6.0,
        noise_floor_std: float = 0.0,
        *,
        bandwidth: float = 125e3,
        coding_rate: int = 1,
        capture_threshold_dB: float = 6.0,
        tx_power_std: float = 0.0,
        interference_dB: float = 0.0,
    ):
        """
        Initialise le canal radio avec paramètres de propagation.

        :param frequency_hz: Fréquence en Hz (par défaut 868 MHz).
        :param path_loss_exp: Exposant de perte de parcours (log-distance).
        :param shadowing_std: Écart-type du shadowing (variations aléatoires en dB), 0 pour ignorer.
        :param fast_fading_std: Variation rapide de l'amplitude (dB) pour simuler le fading multipath.
        :param cable_loss_dB: Pertes fixes dues au câble/connectique (dB).
        :param receiver_noise_floor_dBm: Niveau de bruit thermique de référence (dBm/Hz).
        :param noise_figure_dB: Facteur de bruit ajouté par le récepteur (dB).
        :param noise_floor_std: Écart-type de la variation aléatoire du bruit
            (dB). Utile pour modéliser un canal plus dynamique.
        :param bandwidth: Largeur de bande LoRa (Hz).
        :param coding_rate: Index de code (0=4/5 … 4=4/8).
        :param capture_threshold_dB: Seuil de capture pour le décodage simultané.
        :param tx_power_std: Écart-type de la variation aléatoire de puissance TX.
        :param interference_dB: Bruit supplémentaire moyen dû aux interférences.
        """

        self.frequency_hz = frequency_hz
        self.path_loss_exp = path_loss_exp
        self.shadowing_std = shadowing_std  # σ en dB (ex: 6.0 pour environnement urbain/suburbain)
        self.fast_fading_std = fast_fading_std
        self.cable_loss_dB = cable_loss_dB
        self.receiver_noise_floor_dBm = receiver_noise_floor_dBm
        self.noise_figure_dB = noise_figure_dB
        self.noise_floor_std = noise_floor_std
        self.tx_power_std = tx_power_std
        self.interference_dB = interference_dB

        # Paramètres LoRa (BW 125 kHz, CR 4/5, préambule 8, CRC activé)
        self.bandwidth = bandwidth
        self.coding_rate = coding_rate
        self.preamble_symbols = 8
        self.low_data_rate_threshold = 11  # SF >= 11 -> Low Data Rate Optimization activé

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

    def noise_floor_dBm(self) -> float:
        """Retourne le niveau de bruit (dBm) pour la bande passante configurée.

        Le bruit peut varier autour de la valeur moyenne pour simuler un canal
        plus réaliste.
        """
        thermal = self.receiver_noise_floor_dBm + 10 * math.log10(self.bandwidth)
        noise = thermal + self.noise_figure_dB + self.interference_dB
        if self.noise_floor_std > 0:
            noise += random.gauss(0, self.noise_floor_std)
        return noise

    def path_loss(self, distance: float) -> float:
        """Calcule la perte de parcours (en dB) pour une distance donnée (m)."""
        if distance <= 0:
            return 0.0
        # Modèle log-distance: PL(d) = PL(d0) + 10*gamma*log10(d/d0), avec d0 = 1 m.
        # Calcul de la perte à 1 m en utilisant le modèle espace libre:
        freq_mhz = self.frequency_hz / 1e6
        # FSPL à d0=1m: 32.45 + 20*log10(freq_MHz) - 60 dB (car 20*log10(0.001 km) = -60)
        pl_d0 = 32.45 + 20 * math.log10(freq_mhz) - 60.0
        # Perte à la distance donnée
        pl = pl_d0 + 10 * self.path_loss_exp * math.log10(max(distance, 1.0) / 1.0)
        return pl

    def compute_rssi(self, tx_power_dBm: float, distance: float) -> tuple[float, float]:
        """Calcule le RSSI et le SNR attendus à une certaine distance."""
        # Calcul de la perte de propagation
        loss = self.path_loss(distance)
        if self.shadowing_std > 0:
            loss += random.gauss(0, self.shadowing_std)
        # RSSI = P_tx - pertes - pertes câble
        rssi = tx_power_dBm - loss - self.cable_loss_dB
        if self.tx_power_std > 0:
            rssi += random.gauss(0, self.tx_power_std)
        if self.fast_fading_std > 0:
            rssi += random.gauss(0, self.fast_fading_std)
        snr = rssi - self.noise_floor_dBm()
        return rssi, snr

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
