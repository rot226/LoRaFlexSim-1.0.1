import types
import pytest

from loraflexsim.launcher.simulator import Simulator
import loraflexsim.run as run
import pytest
import types

# Test that None first_packet_interval defaults to packet_interval
def test_simulator_default_first_interval():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        mobility=False,
        packets_to_send=1,
        packet_interval=7.5,
        first_packet_interval=None,
    )
    assert sim.packet_interval == 7.5
    assert sim.first_packet_interval == 7.5


# Test that CLI parameter --first-interval overrides default

def test_cli_first_interval_overrides(monkeypatch):
    received = {}

    def fake_simulate(*args, first_interval=None, **kwargs):
        received['first_interval'] = first_interval
        return 0, 0, 0, 0, 0, 0

    monkeypatch.setattr(run, 'simulate', fake_simulate)
    run.main([
        '--nodes', '1',
        '--gateways', '1',
        '--interval', '5',
        '--steps', '10',
        '--first-interval', '3'
    ])
    assert received['first_interval'] == 3.0


# Test dashboard callback syncing behaviour

def test_dashboard_first_interval_sync(monkeypatch):
    try:
        pn = pytest.importorskip('panel')
        dashboard = pytest.importorskip('loraflexsim.launcher.dashboard')
    except Exception:
        pytest.skip('panel import failed', allow_module_level=True)

    dashboard.first_packet_user_edited = False
    dashboard._syncing_first_packet = False
    dashboard.interval_input.value = 10
    dashboard.first_packet_input.value = 10

    event = types.SimpleNamespace(new=20)
    dashboard.on_interval_update(event)
    assert dashboard.first_packet_input.value == 20

    # Simulate user edit breaking the link
    event_fp = types.SimpleNamespace(new=25)
    dashboard.on_first_packet_change(event_fp)
    dashboard.interval_input.value = 30
    dashboard.on_interval_update(types.SimpleNamespace(new=30))
    assert dashboard.first_packet_input.value == 25


def test_first_interval_matches_poisson():
    seed = 123
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=5.0,
        first_packet_interval=10.0,
        packets_to_send=2,
        mobility=False,
        pure_poisson_mode=True,
        seed=seed,
    )
    from traffic.rng_manager import RngManager
    from traffic.exponential import sample_interval

    rng = RngManager((seed or 0) ^ 3091881735).get_stream("traffic", 0)
    expected_first = sample_interval(5.0, rng)
    expected_second = expected_first + sample_interval(5.0, rng)

    sim.run()
    times = [e["poisson_time"] for e in sim.nodes[0].interval_log]
    assert times[0] == pytest.approx(expected_first)
    assert times[1] == pytest.approx(expected_second)
