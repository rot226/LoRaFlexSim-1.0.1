import pytest

try:
    dashboard = pytest.importorskip('loraflexsim.launcher.dashboard')
except Exception:
    pytest.skip('dashboard import failed', allow_module_level=True)


def test_pause_then_finish_resets_buttons():
    # use minimal packets to finish quickly
    dashboard.packets_input.value = 1
    dashboard.num_runs_input.value = 1

    dashboard.setup_simulation()
    assert dashboard.sim is not None
    assert dashboard.sim.running

    # Pause the simulation
    dashboard.on_pause()
    assert dashboard.paused
    assert dashboard.pause_button.name.startswith('▶')

    # Stop while paused
    dashboard.on_stop(None)

    assert not dashboard.paused
    assert dashboard.pause_button.name == '⏸ Pause'
    assert dashboard.fast_forward_button.disabled
    assert not dashboard.start_button.disabled

    # cleanup
    dashboard._cleanup_callbacks()
