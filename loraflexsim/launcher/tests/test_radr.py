from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher import radr, ADR_MODULES


def make_simulator():
    return Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        mobility=False,
        adr_node=False,
        adr_server=False,
        duty_cycle=None,
        seed=1,
    )


def test_radr_is_registered():
    assert "RADR" in ADR_MODULES


def test_radr_reward_and_action():
    sim = make_simulator()
    radr.apply(sim)
    assert sim.network_server.adr_method == "radr"
    assert hasattr(sim.network_server, "adr_model")

    sf = 12
    snr = -2.0  # above REQUIRED_SNR + margin -> should decrease SF
    action = sim.network_server.adr_action(snr, sf)
    assert action == "sf_down"

    sim.network_server.adr_reward(snr, sf, action)
    state = (round(snr), sf)
    expected = snr - Simulator.REQUIRED_SNR[sf]
    assert sim.network_server.adr_model.q_table[(state, action)] == expected


def test_radr_model_convergence():
    model = radr.RADRModel()
    state = (0, 12)
    for _ in range(5):
        model.update(state, "sf_down", 1.0)
        model.update(state, "sf_up", -1.0)
    assert model.best_action(state) == "sf_down"
