from loraflexsim.launcher.node import Node
from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.qos import QoSManager


def _make_simulator():
    return Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        first_packet_interval=0.1,
        mobility=False,
        warm_up_intervals=0,
        packets_to_send=0,
        seed=42,
    )


def test_qos_reallocation_triggers_automatically():
    simulator = _make_simulator()
    manager = QoSManager()
    manager.configure_clusters(
        1,
        proportions=[1.0],
        arrival_rates=[1.0],
        pdr_targets=[0.9],
    )
    manager.pdr_drift_threshold = 0.2
    manager.reconfig_interval_s = 0.0
    manager.apply(simulator, "APRA-like")

    initial_time = manager._last_reconfig_time or 0.0

    simulator.run(max_steps=50, max_time=5.0)
    drift_time = manager._last_reconfig_time or 0.0
    assert drift_time > initial_time

    manager.reconfig_interval_s = 0.5
    manager.apply(simulator, "APRA-like")
    periodic_base = manager._last_reconfig_time or 0.0
    simulator.run(max_time=simulator.current_time + 1.0)
    periodic_time = manager._last_reconfig_time or 0.0
    assert periodic_time > periodic_base

    new_id = max(node.id for node in simulator.nodes) + 1
    channel = simulator.multichannel.select_mask(0xFFFF)
    new_node = Node(new_id, 0.0, 0.0, 7, 14.0, channel=channel, activated=False)
    new_node.simulator = simulator
    new_node.assigned_channel_index = simulator.channel_index(channel)
    new_node.rng = simulator.rng_manager.get_stream("traffic", new_id)
    simulator.nodes.append(new_node)
    simulator.node_map[new_id] = new_node
    simulator.num_nodes += 1
    simulator.network_server.nodes = simulator.nodes

    simulator.network_server._activate(new_node)
    assert new_id in manager._last_node_ids
    assert simulator.qos_node_clusters.get(new_id) is not None
