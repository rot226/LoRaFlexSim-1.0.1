from __future__ import annotations

import json

from sfrd.cli import run_campaign


def test_run_campaign_is_seed_deterministic(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    result_a = run_campaign(
        network_size=12,
        algorithm="adr",
        snir_mode="snir_on",
        seed=42,
        warmup_s=240.0,
        output_dir=out_a,
    )
    result_b = run_campaign(
        network_size=12,
        algorithm="adr",
        snir_mode="snir_on",
        seed=42,
        warmup_s=240.0,
        output_dir=out_b,
    )

    payload_a = json.loads(result_a["summary_path"].read_text(encoding="utf-8"))
    payload_b = json.loads(result_b["summary_path"].read_text(encoding="utf-8"))

    assert payload_a["runtime"]["gateways"] == 1
    assert payload_a["runtime"]["internal_entrypoint"] == "loraflexsim.launcher.Simulator.run"
    metrics_a = dict(payload_a["metrics"])
    metrics_b = dict(payload_b["metrics"])
    metrics_a.pop("runtime_profile_s", None)
    metrics_b.pop("runtime_profile_s", None)
    metrics_a.pop("qos_refresh_benchmark", None)
    metrics_b.pop("qos_refresh_benchmark", None)
    assert metrics_a == metrics_b
