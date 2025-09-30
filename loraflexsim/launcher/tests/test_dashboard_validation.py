import pytest

panel = pytest.importorskip("panel")
dashboard = pytest.importorskip("loraflexsim.launcher.dashboard")


def reset_inputs():
    dashboard.sim = None
    dashboard.interval_input.value = 1.0
    dashboard.num_nodes_input.value = 1
    dashboard.area_input.value = 100.0
    dashboard.packets_input.value = 1
    dashboard.real_time_duration_input.value = 0.0
    dashboard.export_message.object = ""


def test_invalid_interval_prevents_start():
    reset_inputs()
    dashboard.interval_input.value = 0
    dashboard.on_start(None)
    assert dashboard.sim is None
    assert "⚠️" in dashboard.export_message.object


def test_invalid_nodes_prevents_start():
    reset_inputs()
    dashboard.num_nodes_input.value = 0
    dashboard.on_start(None)
    assert dashboard.sim is None
    assert "⚠️" in dashboard.export_message.object


def test_invalid_area_prevents_start():
    reset_inputs()
    dashboard.area_input.value = -1
    dashboard.on_start(None)
    assert dashboard.sim is None
    assert "⚠️" in dashboard.export_message.object
