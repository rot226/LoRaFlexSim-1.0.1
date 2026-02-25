from mobilesfrdth.simulator.channel import ChannelConfig, pathloss_log_distance_db, received_power_dbm
from mobilesfrdth.simulator.engine import EventDrivenEngine, Node
from mobilesfrdth.simulator.interference import InterferenceConfig, transmission_success
from mobilesfrdth.simulator.mab.ucb import UCB1
from mobilesfrdth.simulator.mab.ucb_forget import UCBForget


def test_engine_event_driven_periodic_uplink():
    engine = EventDrivenEngine(seed=1)
    nodes = [Node(node_id=1, period_s=5.0), Node(node_id=2, period_s=10.0)]
    result = engine.run(nodes=nodes, until_s=25.0)
    assert result.uplink_count >= 6


def test_channel_pathloss_increases_with_distance():
    cfg = ChannelConfig()
    near = pathloss_log_distance_db(10.0, cfg)
    far = pathloss_log_distance_db(100.0, cfg)
    assert far > near
    pr = received_power_dbm(14.0, 100.0, cfg)
    assert pr < 14.0


def test_interference_snir_on_off():
    cfg_off = InterferenceConfig(snir_enabled=False)
    ok_off, _ = transmission_success(-110.0, signal_sf=7, interferers=[(-95.0, 7)], cfg=cfg_off)
    assert ok_off

    cfg_on = InterferenceConfig(snir_enabled=True)
    ok_on, _ = transmission_success(-110.0, signal_sf=7, interferers=[(-95.0, 7)], cfg=cfg_on)
    assert not ok_on


def test_ucb_variants():
    ucb = UCB1(n_arms=2)
    arm = ucb.select_arm()
    ucb.update(arm, 1.0)
    assert sum(ucb.counts) == 1

    adaptive = UCBForget.from_yaml_config({"n_arms": 3, "mode": "sliding_window", "window_size": 4})
    a = adaptive.select_arm()
    adaptive.update(a, 0.5)
    assert adaptive.total_pulls == 1
