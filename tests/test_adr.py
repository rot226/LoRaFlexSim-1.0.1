from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.adr_standard_1 import apply as apply_adr
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.lorawan import TX_POWER_INDEX_TO_DBM


def _run(distance: float, initial_sf: int = 12, packets: int = 30):
    ch = Channel(shadowing_std=0.0, fast_fading_std=0.0, noise_floor_std=0.0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=packets,
        mobility=False,
        adr_server=True,
        adr_method="avg",
        channels=[ch],
        seed=1,
    )
    apply_adr(sim)
    node = sim.nodes[0]
    gw = sim.gateways[0]
    node.x = 0.0
    node.y = 0.0
    gw.x = distance
    gw.y = 0.0
    node.sf = initial_sf
    node.initial_sf = initial_sf
    sim.run()
    return node


def test_adr_decreases_sf_with_good_link():
    node = _run(distance=1.0)
    assert node.sf == 7
    assert node.tx_power == TX_POWER_INDEX_TO_DBM[4]


def test_adr_increases_sf_with_poor_link():
    node = _run(distance=10000.0, initial_sf=8)
    assert node.sf == 7
    assert node.tx_power == TX_POWER_INDEX_TO_DBM[3]
