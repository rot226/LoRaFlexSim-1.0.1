from pathlib import Path
import pytest

try:  # pandas is optional for the capture comparison helper
    import pandas  # noqa: F401
    _HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PANDAS = False

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.omnet_phy import OmnetPHY
from loraflexsim.launcher.compare_flora import load_flora_metrics


def test_omnet_phy_flora_capture_matches_sca():
    if not _HAS_PANDAS:
        pytest.skip('pandas import failed')
    ch = Channel(phy_model="omnet", flora_capture=True, shadowing_std=0.0, fast_fading_std=0.0)
    phy: OmnetPHY = ch.omnet_phy
    rssi_list = [-50.0, -55.0]
    start_list = [0.0, 0.0]
    end_list = [0.1, 0.1]
    sf_list = [7, 7]
    freq_list = [868e6, 868e6]
    winners = phy.capture(
        rssi_list,
        start_list=start_list,
        end_list=end_list,
        sf_list=sf_list,
        freq_list=freq_list,
    )
    collisions = len(rssi_list) - sum(1 for w in winners if w)
    sca = Path(__file__).parent / "data" / "flora_capture_expected.sca"
    flora = load_flora_metrics(sca)
    assert collisions == flora["collisions"]


def test_compute_snrs_ignores_other_channels():
    ch = Channel(
        phy_model="omnet",
        flora_capture=True,
        shadowing_std=0.0,
        fast_fading_std=0.0,
        noise_floor_std=0.0,
    )
    phy: OmnetPHY = ch.omnet_phy
    noise = phy.noise_floor()
    rssi_list = [-90.0, -85.0]
    start_list = [0.0, 0.0]
    end_list = [1.0, 1.0]
    freq_list = [868.1e6, 868.3e6]
    bandwidth_list = [ch.bandwidth, ch.bandwidth]
    snrs = phy.compute_snrs(
        rssi_list,
        start_list,
        end_list,
        noise_dBm=noise,
        freq_list=freq_list,
        bandwidth_list=bandwidth_list,
    )
    expected = [rssi - noise for rssi in rssi_list]
    for snr, ref in zip(snrs, expected):
        assert snr == pytest.approx(ref, abs=1e-9)
