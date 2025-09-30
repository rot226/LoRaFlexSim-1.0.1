import pytest

from loraflexsim.launcher.simulator import Simulator
try:
    pn = pytest.importorskip("panel")
    import loraflexsim.launcher.dashboard as dashboard  # noqa: E402
except Exception:
    pytest.skip("panel import failed", allow_module_level=True)


def test_fast_forward_on_finished_simulation():
    sim = Simulator(num_nodes=1, num_gateways=1, packets_to_send=1, mobility=False)
    sim.run()
    assert not sim.event_queue

    dashboard.sim = sim
    dashboard.fast_forward_button.disabled = False
    dashboard.fast_forward_progress.value = 0
    dashboard.fast_forward_progress.visible = False

    dashboard.fast_forward()

    assert dashboard.fast_forward_button.disabled
    assert dashboard.fast_forward_progress.value == 0
    assert dashboard.fast_forward_progress.visible is False
    assert dashboard.sim.running is False
