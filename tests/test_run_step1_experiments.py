from __future__ import annotations

import argparse
import json

import pytest

from scripts import run_step1_experiments


def test_ensure_collisions_snir_requires_field():
    with pytest.raises(ValueError, match="collisions_snir"):
        run_step1_experiments._ensure_collisions_snir({})


def test_ensure_collisions_snir_accepts_field():
    run_step1_experiments._ensure_collisions_snir({"collisions_snir": 0})


def _dummy_simulator(*, use_snir: bool) -> object:
    class DummyChannel:
        def __init__(self, snir_state: bool) -> None:
            self.use_snir = snir_state
            self.flora_capture = False
            self.advanced_capture = True
            self.capture_threshold_dB = 6.0
            self.capture_window_symbols = 6
            self.snir_window = "packet"
            self.noise_floor_dB = -174.0
            self.interference_dB = 0.0
            self.orthogonal_sf = False
            self.alpha_isf = 0.0
            self.snir_model = True
            self.marginal_snir_margin_db = 1.5
            self.marginal_snir_drop_prob = 0.25
            self.snir_fading_std = 1.5
            self.noise_floor_std = 0.5

    class DummyNode:
        learning_method = "ucb1"

    class DummySimulator:
        def __init__(self, snir_state: bool) -> None:
            self.multichannel = type("Multi", (), {"channels": [DummyChannel(snir_state)]})()
            self.qos_active = True
            self.qos_algorithm = "mixra_h"
            self.adr_node = False
            self.adr_server = False
            self.nodes = [DummyNode()]

    return DummySimulator(use_snir)


def test_snir_switch_report_marks_only_decode_gate_changed() -> None:
    simulator = _dummy_simulator(use_snir=False)

    report = run_step1_experiments._snir_switch_report(simulator, requested_use_snir=False)

    assert report["decode_gate"]["changed"] is True
    assert report["collision_model"]["changed"] is False
    assert report["capture_effect"]["changed"] is False
    assert report["snir_thresholds"]["changed"] is False
    assert report["interference_treatment"]["changed"] is False


def test_write_run_config_exports_expected_sections(tmp_path) -> None:
    simulator = _dummy_simulator(use_snir=True)
    args = argparse.Namespace(
        seed=7,
        algorithm="adr",
        mixra_solver="auto",
        nodes=123,
        packet_interval=42.0,
        duration=180.0,
        pure_poisson=False,
        fading_model="rayleigh",
        use_snir=True,
        channel_config="config.ini",
        skip_lorawan_validation=False,
    )

    path = run_step1_experiments._write_run_config(
        tmp_path,
        args=args,
        simulator=simulator,
        effective_use_snir=True,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 7
    assert payload["algorithm"] == "adr"
    assert payload["simulation"]["num_nodes"] == 123
    assert payload["snir"]["switches"]["decode_gate"]["changed"] is True
    assert payload["snir"]["switches"]["collision_model"]["changed"] is False
    assert "radio" in payload and isinstance(payload["radio"], dict)
    assert "qos" in payload and isinstance(payload["qos"], dict)


def test_main_use_snir_without_fading_std_uses_default_and_does_not_crash(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummySimulator:
        def __init__(self) -> None:
            self.use_snir = True
            self.multichannel = type("Multi", (), {"channels": []})()
            self.channel = None
            self.current_time = 0.0
            self.snir_fading_std = 3.0
            self.noise_floor_std = 0.0
            self.capture_delta_db = 6.0
            self.marginal_snir_margin_db = 0.0
            self.marginal_snir_drop_prob = 0.0

        def run(self, *, max_time: float) -> None:
            self.current_time = max_time

        def get_metrics(self) -> dict[str, object]:
            return {"collisions_snir": 0, "PDR": 1.0, "DER": 1.0}

    def fake_instantiate(*args, **kwargs):
        captured["fading_std_db"] = kwargs.get("fading_std_db")
        return DummySimulator()

    monkeypatch.setattr(run_step1_experiments, "_instantiate_simulator", fake_instantiate)
    monkeypatch.setattr(run_step1_experiments, "_configure_clusters", lambda *a, **k: None)
    monkeypatch.setattr(run_step1_experiments, "_apply_algorithm", lambda *a, **k: None)
    monkeypatch.setattr(run_step1_experiments, "_compute_additional_metrics", lambda *a, **k: a[1])
    monkeypatch.setattr(run_step1_experiments, "_flatten_metrics", lambda payload: payload)
    monkeypatch.setattr(run_step1_experiments, "_write_csv", lambda *a, **k: None)
    monkeypatch.setattr(run_step1_experiments, "_write_run_config", lambda output_dir, **k: output_dir / "run_config.json")

    result = run_step1_experiments.main([
        "--algorithm",
        "adr",
        "--nodes",
        "10",
        "--duration",
        "1",
        "--packet-interval",
        "60",
        "--seed",
        "1",
        "--output-dir",
        str(tmp_path),
        "--use-snir",
        "--quiet",
    ])

    assert captured["fading_std_db"] == run_step1_experiments.DEFAULT_SNIR_FADING_STD_DB
    assert result["csv_path"].name.endswith("_snir-on.csv")
