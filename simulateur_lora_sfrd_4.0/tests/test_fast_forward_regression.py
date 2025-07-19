import importlib.util
import pytest

# Skip the test if Panel or required dependencies aren't available
required = ["panel", "numpy", "pandas", "plotly"]
if any(importlib.util.find_spec(pkg) is None for pkg in required):
    pytest.skip("panel not available in test environment", allow_module_level=True)

import threading

import VERSION_4.launcher.dashboard as dash


def test_fast_forward_triggers_on_stop(monkeypatch):
    """on_stop(None) should run even when session_alive() is False."""

    class DummySim:
        def __init__(self):
            self.running = True
            self.event_queue = [1]
            self.packets_to_send = 1
            self.num_nodes = 1
            self.packets_sent = 0

        def step(self):
            self.packets_sent += 1
            self.event_queue.pop(0)

        def get_metrics(self):
            return {
                "PDR": 1.0,
                "collisions": 0,
                "energy_J": 0.0,
                "avg_delay_s": 0.0,
                "throughput_bps": 0.0,
                "retransmissions": 0,
                "sf_distribution": {7: 1},
            }

        def get_events_dataframe(self):
            return None

    dash.sim = DummySim()
    dash.export_button.disabled = True

    called = []

    def fake_on_stop(ev):
        called.append(True)
        dash.export_button.disabled = False

    monkeypatch.setattr(dash, "on_stop", fake_on_stop)
    monkeypatch.setattr(dash, "session_alive", lambda: False)

    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(threading, "Thread", DummyThread)

    dash.fast_forward()

    assert called
    assert dash.export_button.disabled is False


def test_fast_forward_button_disabled_when_no_packets(monkeypatch):
    """setup_simulation should disable fast forward when packets_to_send==0."""

    class DummySim:
        def __init__(self, **kw):
            self.packets_to_send = kw.get("packets_to_send", 0)
            self.running = True
            self.num_nodes = 1
            self.nodes = []
            self.gateways = []

    monkeypatch.setattr(dash, "Simulator", DummySim)
    monkeypatch.setattr(dash, "update_map", lambda: None)
    monkeypatch.setattr(dash, "update_timeline", lambda: None)
    monkeypatch.setattr(dash.pn.state, "add_periodic_callback", lambda *a, **k: None)
    monkeypatch.setattr(dash, "selected_adr_module", None)
    dash.manual_pos_toggle.value = False
    dash.packets_input.value = 0
    dash.real_time_duration_input.value = 10

    dash.setup_simulation(seed_offset=0)

    assert dash.fast_forward_button.disabled is True


def test_fast_forward_multiple_runs(monkeypatch):
    """fast_forward should work across consecutive runs."""

    class DummySim:
        def __init__(self):
            self.running = True
            self.event_queue = [1]
            self.packets_to_send = 1
            self.num_nodes = 1
            self.packets_sent = 0

        def step(self):
            self.packets_sent += 1
            self.event_queue.pop(0)

        def get_metrics(self):
            return {
                "PDR": 1.0,
                "collisions": 0,
                "energy_J": 0.0,
                "avg_delay_s": 0.0,
                "throughput_bps": 0.0,
                "retransmissions": 0,
                "sf_distribution": {7: 1},
            }

        def get_events_dataframe(self):
            return None

    def fake_setup(seed_offset=0):
        dash.sim = DummySim()
        dash.fast_forward_button.disabled = dash.sim.packets_to_send <= 0

    monkeypatch.setattr(dash, "setup_simulation", fake_setup)
    monkeypatch.setattr(dash, "session_alive", lambda: False)

    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(threading, "Thread", DummyThread)

    dash.sim = DummySim()
    dash.total_runs = 2
    dash.current_run = 1
    dash.runs_events.clear()
    dash.runs_metrics.clear()

    dash.fast_forward()

    assert dash.current_run == 2
    assert dash.fast_forward_progress.value == 0
    assert dash.fast_forward_progress.visible is False
