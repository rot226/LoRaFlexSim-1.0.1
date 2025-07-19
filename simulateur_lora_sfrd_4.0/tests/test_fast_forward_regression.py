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
