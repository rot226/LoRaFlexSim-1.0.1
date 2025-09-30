from pathlib import Path

from loraflexsim.launcher.config_loader import load_config


def test_load_yaml_json_syntax(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(
        "{\n"
        "  \"gateways\": [\n"
        "    {\"x\": 0.0, \"y\": 0.0},\n"
        "    {\"x\": 6000.0, \"y\": 0.0}\n"
        "  ],\n"
        "  \"nodes\": [\n"
        "    {\"x\": 100.0, \"y\": 200.0, \"sf\": 7, \"tx_power\": 14.0},\n"
        "    {\"x\": 2500.0, \"y\": 300.0, \"sf\": 9, \"tx_power\": 20.0}\n"
        "  ]\n"
        "}\n"
    )

    nodes, gateways, next_interval, first_interval = load_config(path)
    assert next_interval is None
    assert first_interval is None
    assert len(gateways) == 2
    assert gateways[1]["x"] == 6000.0
    assert gateways[0]["tx_power"] is None
    assert len(nodes) == 2
    assert nodes[0]["sf"] == 7
    assert nodes[1]["tx_power"] == 20.0

