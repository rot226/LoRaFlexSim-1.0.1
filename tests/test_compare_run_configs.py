import subprocess
import sys


def test_compare_run_configs_detects_delta(tmp_path):
    old_cfg = tmp_path / "old.json"
    new_cfg = tmp_path / "new.json"
    old_cfg.write_text('{"radio": {"snir_mode": false}, "traffic": {"packets_per_node": 10}}', encoding="utf-8")
    new_cfg.write_text('{"radio": {"snir_mode": true}, "traffic": {"packets_per_node": 10}}', encoding="utf-8")

    cmd = [sys.executable, "scripts/compare_run_configs.py", str(old_cfg), str(new_cfg)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 1
    assert "radio.snir_mode" in result.stdout


def test_compare_run_configs_reports_no_difference(tmp_path):
    cfg_a = tmp_path / "a.json"
    cfg_b = tmp_path / "b.json"
    payload = '{"seed": 42, "topology": {"num_gateways": 1}}'
    cfg_a.write_text(payload, encoding="utf-8")
    cfg_b.write_text(payload, encoding="utf-8")

    cmd = [sys.executable, "scripts/compare_run_configs.py", str(cfg_a), str(cfg_b)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "Aucune différence" in result.stdout
