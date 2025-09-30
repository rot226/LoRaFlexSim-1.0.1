import pytest
from loraflexsim.launcher.channel import Channel


def test_energy_accumulation():
    ch = Channel(
        phy_model="omnet",
        tx_current_a=2.0,
        rx_current_a=1.0,
        idle_current_a=0.5,
        voltage_v=3.0,
    )
    phy = ch.omnet_phy
    phy.tx_start_delay_s = 1.0
    phy.tx_start_current_a = 4.0
    phy.rx_start_delay_s = 0.5
    phy.rx_start_current_a = 1.5
    phy.tx_state = "off"
    phy.rx_state = "off"

    # Start transmission
    phy.start_tx()
    phy.update(1.0)
    assert phy.energy_start == pytest.approx(12.0)

    # Transmission energy
    phy.update(2.0)
    assert phy.energy_tx == pytest.approx(12.0)
    phy.stop_tx()

    # Start reception
    phy.start_rx()
    phy.update(0.5)
    assert phy.energy_start == pytest.approx(12.0 + 2.25)

    # Reception energy
    phy.update(1.0)
    assert phy.energy_rx == pytest.approx(3.0)
    phy.stop_rx()

    # Idle energy
    phy.update(1.0)
    assert phy.energy_idle == pytest.approx(1.5)
