import random
from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.lorawan import TX_POWER_INDEX_TO_DBM


def test_adr_ack_delay_adjustment():
    random.seed(0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        duty_cycle=None,
        mobility=False,
        adr_node=True,
        adr_server=False,
        fixed_sf=7,
        fixed_tx_power=2.0,
        seed=42,
    )
    node = sim.nodes[0]
    for _ in range(1000):
        node.prepare_uplink(b"test")
    assert node.sf == 12
    assert node.tx_power == TX_POWER_INDEX_TO_DBM[1]
    assert node.adr_ack_cnt == 1000 % (node.adr_ack_limit + node.adr_ack_delay)
