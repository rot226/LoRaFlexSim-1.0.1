from loraflexsim.launcher.qos import QoSManager


class _DummySimulator:
    def __init__(self, node_count: int) -> None:
        self.nodes = [object() for _ in range(node_count)]


def test_mixra_h_periodic_refresh_interval_scales_for_large_networks():
    manager = QoSManager()
    manager.active_algorithm = "MixRA-H"
    manager.reconfig_interval_s = 60.0
    manager.mixra_h_refresh_interval_s = None
    manager.mixra_h_large_network_size_threshold = 10
    manager.mixra_h_large_network_refresh_interval_s = 180.0

    small = manager.periodic_refresh_interval_s(_DummySimulator(8))
    large = manager.periodic_refresh_interval_s(_DummySimulator(10))

    assert small == 60.0
    assert large == 180.0
