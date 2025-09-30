from __future__ import annotations

import configparser
from pathlib import Path

from .simulator import Simulator
from . import server
from .advanced_channel import AdvancedChannel
from .lorawan import TX_POWER_INDEX_TO_DBM
from .channel import Channel
from .gateway import FLORA_NON_ORTH_DELTA

# ---------------------------------------------------------------------------
# Channel degradation profiles.
# Values common to all profiles are defined here while ``path_loss_exp`` and
# ``shadowing_std`` are obtained from :data:`Channel.ENV_PRESETS`.
# ---------------------------------------------------------------------------


def _degrade_params(profile: str, capture_mode: str) -> dict:
    """Return channel degradation parameters for ``profile``.

    ``profile`` selects path loss and shadowing values from
    :data:`Channel.ENV_PRESETS`.  Unknown profiles fall back to ``"flora"``.
    Parameters ``variable_noise_std``, ``fine_fading_std``, ``fading`` and
    ``rician_k`` can be overridden in a ``config.ini`` file under the ``[channel]``
    section.
    """

    ple, shadow, *_ = Channel.ENV_PRESETS.get(
        profile, Channel.ENV_PRESETS["flora"]
    )

    # Default degradation values tuned for a harsh radio environment
    variable_noise_std = 20.0
    fine_fading_std = 20.0
    fading = "rayleigh"
    rician_k = 0.0

    # Override with values from config.ini when available.  Noise and fading
    # standard deviations are intentionally kept high and therefore are not
    # overridden by configuration values.
    cp = configparser.ConfigParser()
    cfg_path = Path(__file__).resolve().parents[2] / "config.ini"
    cp.read(cfg_path)
    if cp.has_section("channel"):
        fading = cp.get("channel", "fading", fallback=fading)
        if fading and fading.lower() == "none":
            fading = None
        rician_k = cp.getfloat("channel", "rician_k", fallback=rician_k)

    if capture_mode == "advanced":
        advanced_capture = True
        flora_capture = False
    else:
        advanced_capture = False
        flora_capture = True

    return {
        "propagation_model": "log_distance",  # or "cost231" with adjusted n
        "fading": fading,  # or None
        "rician_k": rician_k,
        "path_loss_exp": ple,
        "shadowing_std": shadow,
        "variable_noise_std": variable_noise_std,
        "fine_fading_std": fine_fading_std,
        "freq_offset_std_hz": 1500.0,
        "sync_offset_std_s": 0.005,
        "advanced_capture": advanced_capture,
        "flora_capture": flora_capture,
        "flora_loss_model": "lognorm",
        "detection_threshold_dBm": -130.0,
        "capture_threshold_dB": 6.0,
    }

def apply(
    sim: Simulator,
    *,
    degrade_channel: bool = False,
    profile: str = "flora",
    capture_mode: str = "flora",
) -> None:
    """Configure ADR variant ``adr_standard_1`` (LoRaWAN defaults).

    Parameters
    ----------
    sim : Simulator
        Instance to modify in-place.
    degrade_channel : bool, optional
        If ``True``, replace existing :class:`~launcher.channel.Channel` objects
        with :class:`~launcher.advanced_channel.AdvancedChannel` instances using
        more realistic propagation impairments.
    profile : str, optional
        Environment key used to select ``path_loss_exp`` and ``shadowing_std``
        from :data:`Channel.ENV_PRESETS`. Defaults to ``"flora"``.
    capture_mode : str, optional
        Selects which capture model to enable. ``"advanced"`` enables the
        detailed capture effect while ``"flora"`` uses the simplified FLoRa
        capture model. Defaults to ``"flora"``.
    """
    # Marge ADR
    Simulator.MARGIN_DB = 15.0
    server.MARGIN_DB = Simulator.MARGIN_DB
    sim.adr_node = True
    sim.adr_server = True
    # Utilise la moyenne de SNR sur 20 paquets comme dans FLoRa
    sim.adr_method = "avg"
    sim.network_server.adr_enabled = True
    sim.network_server.adr_method = "avg"
    for node in sim.nodes:
        # Démarre au SF12 pour une sensibilité maximale sauf si un SF fixe est
        # déjà défini par le simulateur LoRaFlexSim
        if getattr(sim, "fixed_sf", None) is None:
            node.sf = 12
            node.initial_sf = 12
        else:
            node.initial_sf = node.sf
        # Démarre avec la puissance TX maximale (14 dBm, index 0)
        max_tx_power = TX_POWER_INDEX_TO_DBM[0]
        node.tx_power = max_tx_power
        node.initial_tx_power = max_tx_power
        # Compteurs ADR
        node.adr_ack_cnt = 0
        node.adr_ack_limit = 64
        node.adr_ack_delay = 32

    if not degrade_channel:
        sf = 12  # Default spreading factor used at start
        for ch in sim.multichannel.channels:
            ch.detection_threshold_dBm = Channel.flora_detection_threshold(
                sf, ch.bandwidth
            ) + ch.sensitivity_margin_dB
            # Allow different SFs to interfere like in FLoRa
            ch.orthogonal_sf = False
            ch.non_orth_delta = FLORA_NON_ORTH_DELTA
        # Propagate the non-orthogonal behaviour to existing node channels
        for node in sim.nodes:
            node.channel.orthogonal_sf = False
            node.channel.non_orth_delta = FLORA_NON_ORTH_DELTA
        # For fixed spreading factor scenarios keep a very permissive
        # detection threshold to mirror the original behaviour
        if getattr(sim, "fixed_sf", None) is not None:
            for ch in sim.multichannel.channels:
                ch.detection_threshold_dBm = -float("inf")

    if degrade_channel:
        new_channels = []
        base_params = _degrade_params(profile, capture_mode)
        for ch in sim.multichannel.channels:
            params = dict(base_params)
            # Conserver les paramètres spécifiques au canal original
            params["frequency_hz"] = ch.frequency_hz
            if hasattr(ch, "bandwidth"):
                params["bandwidth"] = ch.bandwidth
            if hasattr(ch, "coding_rate"):
                params["coding_rate"] = ch.coding_rate
            sf = 12
            bw = params.get("bandwidth", 125000)
            params["sensitivity_margin_dB"] = getattr(ch, "sensitivity_margin_dB", 0.0)
            params["detection_threshold_dBm"] = Channel.flora_detection_threshold(
                sf, bw
            ) + params["sensitivity_margin_dB"]
            # Créer un canal avancé avec les paramètres mis à jour
            adv = AdvancedChannel(**params)
            adv.orthogonal_sf = False
            adv.non_orth_delta = FLORA_NON_ORTH_DELTA
            new_channels.append(adv)

        # Remplacer la liste des canaux par les nouveaux canaux dégradés
        sim.multichannel.channels = new_channels
        sim.channel = sim.multichannel.channels[0]
        sim.network_server.channel = sim.channel
        # Mettre à jour la référence de canal de chaque nœud
        for node in sim.nodes:
            node.channel = sim.multichannel.select_mask(getattr(node, "chmask", 0xFFFF))
            node.channel.detection_threshold_dBm = Channel.flora_detection_threshold(
                getattr(node, "sf", 12), node.channel.bandwidth
            ) + node.channel.sensitivity_margin_dB
            node.channel.orthogonal_sf = False
            node.channel.non_orth_delta = FLORA_NON_ORTH_DELTA

    # Ensure server and first channel reflect the non-orthogonal setting
    sim.channel = sim.multichannel.channels[0]
    sim.channel.orthogonal_sf = False
    sim.channel.non_orth_delta = FLORA_NON_ORTH_DELTA
    sim.network_server.channel = sim.channel
