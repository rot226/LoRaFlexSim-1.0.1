import math

from loraflexsim.launcher.omnet_modulation import calculate_ber, calculate_ser


def test_calculate_ser_matches_ber_conversion():
    snir = 5.0
    sf = 7
    ber = calculate_ber(snir, sf)
    expected_ser = 1.0 - (1.0 - ber) ** sf
    assert math.isclose(calculate_ser(snir, sf), expected_ser, rel_tol=1e-9)
