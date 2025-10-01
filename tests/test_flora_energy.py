from pathlib import Path

import pytest

from loraflexsim.launcher.energy_profiles import EnergyProfile, FLORA_PROFILE
from loraflexsim.launcher.node import Node
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.compare_flora import load_flora_metrics
from loraflexsim.launcher.simulator import Simulator


def test_cumulative_energy_with_transients_enabled():
    profile = EnergyProfile(
        voltage_v=1.0,
        startup_current_a=2.0,
        startup_time_s=1.0,
        preamble_current_a=3.0,
        preamble_time_s=1.0,
        ramp_up_s=1.0,
        ramp_down_s=1.0,
        tx_current_map_a={14.0: 4.0},
    )
    ch = Channel()
    node = Node(0, 0.0, 0.0, sf=7, tx_power=14.0, channel=ch, energy_profile=profile)
    airtime = 1.0
    tx_energy = profile.get_tx_current(14.0) * profile.voltage_v * airtime
    node.add_energy(tx_energy, "tx", duration_s=airtime)
    expected = (
        tx_energy
        + profile.get_tx_current(14.0) * profile.voltage_v * (profile.ramp_up_s + profile.ramp_down_s)
        + profile.startup_current_a * profile.voltage_v * profile.startup_time_s
        + profile.preamble_current_a * profile.voltage_v * profile.preamble_time_s
    )
    assert node.energy_consumed == pytest.approx(expected)
    assert node.energy_startup == pytest.approx(
        profile.startup_current_a * profile.voltage_v * profile.startup_time_s
    )
    assert node.energy_preamble == pytest.approx(
        profile.preamble_current_a * profile.voltage_v * profile.preamble_time_s
    )
    assert node.energy_tx == pytest.approx(tx_energy)
    assert node.energy_ramp == pytest.approx(
        profile.get_tx_current(14.0)
        * profile.voltage_v
        * (profile.ramp_up_s + profile.ramp_down_s)
    )
    breakdown = node.get_energy_breakdown()
    assert breakdown["tx"] == pytest.approx(node.energy_tx)
    assert breakdown.get("ramp", 0.0) == pytest.approx(node.energy_ramp)
    assert breakdown.get("startup", 0.0) == pytest.approx(node.energy_startup)
    assert breakdown.get("preamble", 0.0) == pytest.approx(node.energy_preamble)
    assert node.energy.total() == pytest.approx(node.energy_consumed)


def test_flora_profile_disables_transients():
    ch = Channel()
    node = Node(0, 0.0, 0.0, sf=7, tx_power=14.0, channel=ch, energy_profile=FLORA_PROFILE)
    airtime = 0.5
    tx_energy = FLORA_PROFILE.energy_for("tx", airtime, power_dBm=14.0)
    node.add_energy(tx_energy, "tx", duration_s=airtime)
    assert node.energy_tx == pytest.approx(tx_energy)
    assert node.energy_consumed == pytest.approx(tx_energy)
    assert node.energy_ramp == 0.0
    assert node.energy_startup == 0.0
    assert node.energy_preamble == 0.0
    breakdown = node.get_energy_breakdown()
    assert breakdown["tx"] == pytest.approx(tx_energy)
    assert "ramp" not in breakdown
    assert "startup" not in breakdown
    assert "preamble" not in breakdown


def test_flora_energy_matches_reference_trace():
    try:
        import pandas  # noqa: F401
    except Exception:
        pytest.skip("pandas indisponible pour la comparaison énergétique")
    reference = Path(__file__).resolve().parent / "data" / "flora_energy_reference.sca"
    flora_metrics = load_flora_metrics(reference)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        mobility=False,
        fixed_sf=7,
        seed=0,
        flora_mode=True,
    )
    sim.run()
    metrics = sim.get_metrics()
    assert metrics["energy_J"] == pytest.approx(
        flora_metrics["energy_J"], rel=0.03
    )
