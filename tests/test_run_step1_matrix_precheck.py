import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for entry in (ROOT_DIR, SCRIPTS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.run_step1_matrix import _validate_precheck_business_rules



def _write_csv(path: Path, header: list[str], row: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(header) + "\n" + ",".join(row) + "\n", encoding="utf8")


def test_precheck_business_rules_accepts_positive_pdr(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "snir_on" / "seed_1" / "adr_N80_T300_snir-on.csv",
        ["algorithm", "num_nodes", "random_seed", "PDR", "DER", "with_snir", "snir_state"],
        ["adr", "80", "1", "0.1", "0.1", "True", "snir_on"],
    )
    _validate_precheck_business_rules(tmp_path)


def test_precheck_business_rules_rejects_missing_columns(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "snir_off" / "seed_1" / "adr_N80_T300_snir-off.csv",
        ["algorithm", "num_nodes", "random_seed", "PDR"],
        ["adr", "80", "1", "0.2"],
    )
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        _validate_precheck_business_rules(tmp_path)


def test_precheck_business_rules_rejects_all_zero_pdr(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "snir_on" / "seed_2" / "apra_N80_T300_snir-on.csv",
        ["algorithm", "num_nodes", "random_seed", "PDR", "DER", "with_snir", "snir_state"],
        ["apra", "80", "2", "0.0", "1.0", "True", "snir_on"],
    )
    with pytest.raises(ValueError, match="aucun PDR strictement positif"):
        _validate_precheck_business_rules(tmp_path)
