from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "data"

VALID_SF = set(range(7, 13))
DER_COLUMN_PATTERN = re.compile(r"(^|_)der($|_)")


def _collect_csv_paths() -> list[Path]:
    candidates = [
        ROOT_DIR / "IWCMC",
        ROOT_DIR / "results" / "iwcmc",
        ROOT_DIR / "experiments" / "ucb1",
        FIXTURE_DIR,
    ]
    paths: set[Path] = set()
    for base in candidates:
        if base.exists():
            paths.update(base.rglob("*.csv"))
    return sorted(paths)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _der_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if DER_COLUMN_PATTERN.search(col.lower())]


def test_csv_no_nan() -> None:
    csv_paths = _collect_csv_paths()
    if not csv_paths:
        pytest.skip("Aucun CSV trouvé pour valider les NaN.")

    for path in csv_paths:
        df = _load_csv(path)
        assert not df.isna().any().any(), f"NaN détectés dans {path}"


def test_der_within_bounds() -> None:
    csv_paths = _collect_csv_paths()
    if not csv_paths:
        pytest.skip("Aucun CSV trouvé pour valider les DER.")

    for path in csv_paths:
        df = _load_csv(path)
        columns = _der_columns(list(df.columns))
        if not columns:
            continue
        for column in columns:
            series = pd.to_numeric(df[column], errors="coerce")
            assert series.notna().all(), f"DER non numérique dans {path} ({column})"
            assert series.between(0, 1).all(), f"DER hors bornes dans {path} ({column})"


def test_sf_values() -> None:
    csv_paths = _collect_csv_paths()
    if not csv_paths:
        pytest.skip("Aucun CSV trouvé pour valider les SF.")

    for path in csv_paths:
        df = _load_csv(path)
        if "sf" not in df.columns:
            continue
        series = pd.to_numeric(df["sf"], errors="coerce")
        assert series.notna().all(), f"SF non numérique dans {path}"
        assert series.between(min(VALID_SF), max(VALID_SF)).all(), f"SF invalide dans {path}"


def _find_column(df: pd.DataFrame, names: set[str]) -> str | None:
    for column in df.columns:
        if column.lower() in names:
            return column
    return None


def test_ucb1_convergence_definition() -> None:
    csv_paths = _collect_csv_paths()
    if not csv_paths:
        pytest.skip("Aucun CSV trouvé pour valider la convergence UCB1.")

    for path in csv_paths:
        df = _load_csv(path)
        algorithm_col = _find_column(df, {"algorithm", "algo"})
        replication_col = _find_column(df, {"replication_index"})
        der_col = _find_column(df, {"der"})
        if not algorithm_col or not replication_col or not der_col:
            continue

        ucb1_df = df[df[algorithm_col].astype(str).str.lower() == "ucb1"]
        if ucb1_df.empty:
            continue

        ordered = ucb1_df.sort_values(replication_col)
        der_series = pd.to_numeric(ordered[der_col], errors="coerce")
        assert der_series.notna().all(), f"DER non numérique pour UCB1 dans {path}"
        cumulative = der_series.expanding().mean()
        assert cumulative.between(0, 1).all(), f"Convergence UCB1 hors bornes dans {path}"

        convergence_col = _find_column(df, {"der_cumulative", "der_convergence", "convergence_der"})
        if convergence_col:
            expected = cumulative.reset_index(drop=True)
            actual = pd.to_numeric(ordered[convergence_col], errors="coerce").reset_index(drop=True)
            assert actual.notna().all(), f"Convergence UCB1 non numérique dans {path}"
            assert actual.tolist() == pytest.approx(expected.tolist()), (
                f"Convergence UCB1 invalide dans {path}"
            )
