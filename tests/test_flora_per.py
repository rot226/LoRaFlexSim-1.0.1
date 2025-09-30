import csv
import math
import random
from pathlib import Path

import pytest

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.flora_phy import FloraPHY
from loraflexsim.loranode import Node
from loraflexsim.phy import LoRaPHY

DATA_DIR = Path(__file__).with_name("data")
FLORA_PER_TRACE = DATA_DIR / "flora_per_sigmoid.csv"


pytestmark = pytest.mark.propagation_campaign


def test_per_matches_croce_curve():
    ch = Channel(phy_model="omnet", use_flora_curves=False, shadowing_std=0.0)
    sf = 7
    # Points extraits de la courbe PER de Croce et al. (2018) pour un
    # paquet de 20 octets.
    reference = [
        (-9.5, 0.9285),
        (-7.5, 0.4363),
        (-5.5, 0.0550),
    ]
    for snr, expected in reference:
        per = ch.packet_error_rate(snr, sf, payload_bytes=20)
        assert math.isclose(per, expected, rel_tol=0.05)


def test_per_uses_flora_ber_model():
    ch = Channel(phy_model="omnet", use_flora_curves=False, shadowing_std=0.0, bandwidth=125e3)
    sf = 7
    snr = 0.0
    snir = 10 ** (snr / 10.0)
    from loraflexsim.launcher.omnet_modulation import calculate_ber_flora

    ber = calculate_ber_flora(snir, sf, ch.bandwidth)
    ser = 1.0 - (1.0 - ber) ** sf
    n_bits = 20 * 8
    per_bit = 1.0 - (1.0 - ber) ** n_bits
    n_sym = math.ceil(n_bits / sf)
    per_sym = 1.0 - (1.0 - ser) ** n_sym
    expected = max(per_bit, per_sym)
    per = ch.packet_error_rate(snr, sf, payload_bytes=20, ber_model="flora")
    assert math.isclose(per, expected, rel_tol=1e-9)


def test_flora_per_none_mode_disables_losses():
    ch = Channel(phy_model="flora_full", use_flora_curves=True, shadowing_std=0.0)
    ch.flora_phy = FloraPHY(ch, loss_model=ch.flora_loss_model)
    with pytest.warns(RuntimeWarning):
        ch.flora_per_model = "none"
    src = Node(1, 0.0, 0.0, 7, 14, channel=ch)
    dst = Node(2, 10.0, 0.0, 7, 14, channel=ch)
    phy = LoRaPHY(src)
    rng = random.Random(0)
    successes = []
    for _ in range(5):
        _, _, _, ok = phy.transmit(dst, 10, rng=rng)
        successes.append(ok)
    assert all(successes)


def test_flora_per_model_defaults_to_logistic_in_flora_mode():
    ch = Channel(phy_model="flora_full", use_flora_curves=False, shadowing_std=0.0)
    assert ch.flora_per_model == "logistic"
    ch = Channel(phy_model="omnet_full", use_flora_curves=False, shadowing_std=0.0)
    assert ch.flora_per_model == "logistic"


def test_flora_per_model_warning_when_overridden():
    with pytest.warns(RuntimeWarning):
        ch = Channel(
            phy_model="flora_full",
            use_flora_curves=False,
            shadowing_std=0.0,
            flora_per_model="croce",
        )
    assert ch.flora_per_model == "croce"
    with pytest.warns(RuntimeWarning):
        ch.flora_per_model = "croce"


def test_monte_carlo_matches_flora_sigmoid():
    if not FLORA_PER_TRACE.is_file():
        pytest.skip("trace FLoRa manquante")
    ch = Channel(phy_model="flora_full", use_flora_curves=False, shadowing_std=0.0)
    phy = ch.flora_phy
    assert phy is not None
    rng = random.Random(1234)
    samples = 5000
    with FLORA_PER_TRACE.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sf = int(row["sf"])
            payload = int(row["payload_bytes"])
            snr = float(row["snr_dB"])
            expected = float(row["success_rate"])
            per = phy.packet_error_rate(
                snr,
                sf,
                payload_bytes=payload,
                per_model=ch.flora_per_model,
            )
            successes = sum(1 for _ in range(samples) if rng.random() >= per)
            rate = successes / samples
            assert rate == pytest.approx(expected, abs=0.03)
