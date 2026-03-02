from pathlib import Path

import pytest

from mobilesfrdth.scenarios import generate_jobs, parse_grid_spec


def test_parse_grid_spec_normalizes_aliases_and_required_keys() -> None:
    grid = parse_grid_spec("N=50;model=rwp,smooth;mode=off,snir_on;algo=adr,ucb_forget;reps=2;seed_base=100")

    assert grid["model"] == ["RWP", "SMOOTH"]
    assert grid["mode"] == ["SNIR_OFF", "SNIR_ON"]
    assert grid["algo"] == ["ADR", "UCB_FORGET"]


def test_parse_grid_spec_rejects_unknown_or_empty_items() -> None:
    with pytest.raises(ValueError, match="Clé inconnue"):
        parse_grid_spec("N=50;mode=SNIR_ON;algo=ADR;reps=1;seed_base=0;foo=bar")

    with pytest.raises(ValueError, match="valeur vide"):
        parse_grid_spec("N=50;mode=SNIR_ON;algo=ADR;reps=1;seed_base=0;model=")


def test_parse_grid_spec_requires_mandatory_keys() -> None:
    with pytest.raises(ValueError, match="Clés obligatoires manquantes"):
        parse_grid_spec("N=50;mode=SNIR_ON;algo=ADR;reps=1")


def test_generate_jobs_expands_cartesian_and_derives_deterministic_run_id() -> None:
    grid = parse_grid_spec("N=50,100;model=rwp;mode=off;algo=adr;reps=2;seed_base=7")

    jobs = generate_jobs(config_path=Path("experiments/default.yaml"), output_root=Path("runs"), grid=grid)

    assert len(jobs) == 4
    assert jobs[0]["params"]["run_id"] == "n50_model-rwp_mode-snir_off_algo-adr_rep-001_seed-7"
    assert jobs[1]["params"]["run_id"] == "n50_model-rwp_mode-snir_off_algo-adr_rep-002_seed-8"
    assert jobs[2]["params"]["run_id"] == "n100_model-rwp_mode-snir_off_algo-adr_rep-001_seed-7"
    assert jobs[3]["params"]["run_id"] == "n100_model-rwp_mode-snir_off_algo-adr_rep-002_seed-8"
