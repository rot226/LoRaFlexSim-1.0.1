from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_step1_matrix as step1_matrix
import run_step1_experiments as step1_experiments


MIN_SNIR_DELTA_DB = 10.0
MIN_DER_PDR_DELTA = 0.1
MIN_COLLISION_DELTA = 12.0
MIN_COLLISION_MIN_DELTA = 2.0
MIN_COLLISION_CURVE_DELTA = 5.0
MIN_COLLISION_POINT_DELTA = 2.0
MIN_PDR_CURVE_DELTA = 0.06
MIN_DER_CURVE_DELTA = 0.06
MIN_SNIR_CURVE_DELTA_DB = 5.0
MIN_COMBINED_CURVE_DISTANCE = 1.2


def _mean_snir(histogram_source: str) -> float:
    histogram = json.loads(histogram_source)
    total_samples = sum(histogram.values()) or 1
    return sum(float(bin_key) * count for bin_key, count in histogram.items()) / total_samples


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_snir_toggle_generates_distinct_csv(tmp_path: Path) -> None:
    results_dir = tmp_path / "step1"
    args = [
        "--algos",
        "adr",
        "--with-snir",
        "true",
        "false",
        "--seeds",
        "1",
        "2",
        "--nodes",
        "300",
        "--packet-intervals",
        "0.1",
        "--duration",
        "60",
        "--results-dir",
        str(results_dir),
    ]

    step1_matrix.main(args)

    csv_paths = sorted(results_dir.glob("**/*.csv"))
    assert csv_paths, "Aucun CSV généré par run_step1_matrix"

    snir_states: set[str] = set()
    mean_snir_by_state: dict[bool, list[float]] = {True: [], False: []}
    der_by_state: dict[bool, list[float]] = {True: [], False: []}
    pdr_by_state: dict[bool, list[float]] = {True: [], False: []}
    collisions_by_state: dict[bool, list[float]] = {True: [], False: []}

    for path in csv_paths:
        assert path.name.endswith(("_snir-on.csv", "_snir-off.csv")), path.name

        with path.open(newline="", encoding="utf8") as handle:
            row = next(csv.DictReader(handle))

        use_snir = row["use_snir"] == "True"
        snir_state = row["snir_state"]
        snir_states.add(snir_state)
        expected_state = "snir_on" if use_snir else "snir_off"
        assert snir_state == expected_state
        assert row["snir_state_effective"] == expected_state

        mean_snir_by_state[use_snir].append(_mean_snir(row["snir_histogram_json"]))

        der_by_state[use_snir].append(float(row["DER"]))
        pdr_by_state[use_snir].append(float(row["PDR"]))
        collisions_by_state[use_snir].append(float(row["collisions"]))

    assert snir_states == {"snir_on", "snir_off"}

    mean_on = sum(mean_snir_by_state[True]) / len(mean_snir_by_state[True])
    mean_off = sum(mean_snir_by_state[False]) / len(mean_snir_by_state[False])
    assert abs(mean_on - mean_off) >= MIN_SNIR_DELTA_DB

    avg_der_on = sum(der_by_state[True]) / len(der_by_state[True])
    avg_der_off = sum(der_by_state[False]) / len(der_by_state[False])
    avg_pdr_on = sum(pdr_by_state[True]) / len(pdr_by_state[True])
    avg_pdr_off = sum(pdr_by_state[False]) / len(pdr_by_state[False])
    avg_collisions_on = sum(collisions_by_state[True]) / len(collisions_by_state[True])
    avg_collisions_off = sum(collisions_by_state[False]) / len(collisions_by_state[False])
    min_collision_gap = min(
        abs(on - off)
        for on in collisions_by_state[True]
        for off in collisions_by_state[False]
    )

    assert abs(avg_der_on - avg_der_off) >= MIN_DER_PDR_DELTA
    assert abs(avg_pdr_on - avg_pdr_off) >= MIN_DER_PDR_DELTA
    assert abs(avg_collisions_on - avg_collisions_off) >= MIN_COLLISION_DELTA
    assert min_collision_gap >= MIN_COLLISION_MIN_DELTA, (
        f"Δ collisions minimal trop faible entre états SNIR (Δ min={min_collision_gap:.3f})"
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_snir_toggle_curves_are_not_identical(tmp_path: Path) -> None:
    results_dir = tmp_path / "step1_curves"
    args = [
        "--algos",
        "adr",
        "--with-snir",
        "true",
        "false",
        "--seeds",
        "1",
        "--nodes",
        "200",
        "400",
        "600",
        "--packet-intervals",
        "0.2",
        "--duration",
        "50",
        "--results-dir",
        str(results_dir),
    ]

    step1_matrix.main(args)

    csv_paths = sorted(results_dir.glob("**/*.csv"))
    assert csv_paths, "Aucun CSV généré pour comparer les courbes SNIR"

    by_state: dict[bool, dict[int, dict[str, float]]] = {True: {}, False: {}}
    for path in csv_paths:
        with path.open(newline="", encoding="utf8") as handle:
            row = next(csv.DictReader(handle))

        use_snir = row["use_snir"] == "True"
        nodes = int(float(row["num_nodes"]))
        by_state[use_snir][nodes] = {
            "pdr": float(row["PDR"]),
            "der": float(row["DER"]),
            "collisions": float(row["collisions"]),
        }

    assert by_state[True] and by_state[False], "Les deux états SNIR doivent fournir des points de courbe"

    common_nodes = sorted(set(by_state[True]) & set(by_state[False]))
    assert len(common_nodes) >= 3, "Au moins trois points sont nécessaires pour comparer les courbes SNIR"

    def _avg_gap(metric: str) -> float:
        gaps = [
            abs(by_state[True][nodes][metric] - by_state[False][nodes][metric])
            for nodes in common_nodes
        ]
        return sum(gaps) / len(gaps)

    def _min_gap(metric: str) -> float:
        gaps = [
            abs(by_state[True][nodes][metric] - by_state[False][nodes][metric])
            for nodes in common_nodes
        ]
        return min(gaps)

    pdr_gap = _avg_gap("pdr")
    der_gap = _avg_gap("der")
    collisions_gap = _avg_gap("collisions")
    collisions_min_gap = _min_gap("collisions")

    assert pdr_gap >= MIN_PDR_CURVE_DELTA, (
        f"Courbes PDR quasi identiques entre états SNIR (Δ moyen={pdr_gap:.3f})"
    )
    assert der_gap >= MIN_DER_CURVE_DELTA, (
        f"Courbes DER quasi identiques entre états SNIR (Δ moyen={der_gap:.3f})"
    )
    assert collisions_gap >= MIN_COLLISION_CURVE_DELTA, (
        f"Courbes de collisions quasi identiques entre états SNIR (Δ moyen={collisions_gap:.3f})"
    )
    assert collisions_min_gap >= MIN_COLLISION_POINT_DELTA, (
        f"Δ collisions trop faible pour au moins un point SNIR (Δ min={collisions_min_gap:.3f})"
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_snir_toggle_snir_curves_are_not_identical(tmp_path: Path) -> None:
    results_dir = tmp_path / "step1_snir_curves"
    args = [
        "--algos",
        "adr",
        "--with-snir",
        "true",
        "false",
        "--seeds",
        "1",
        "--nodes",
        "200",
        "400",
        "600",
        "--packet-intervals",
        "0.2",
        "--duration",
        "50",
        "--results-dir",
        str(results_dir),
    ]

    step1_matrix.main(args)

    csv_paths = sorted(results_dir.glob("**/*.csv"))
    assert csv_paths, "Aucun CSV généré pour comparer les courbes SNIR on/off"

    mean_snir_by_state: dict[bool, dict[int, float]] = {True: {}, False: {}}
    for path in csv_paths:
        with path.open(newline="", encoding="utf8") as handle:
            row = next(csv.DictReader(handle))

        use_snir = row["use_snir"] == "True"
        nodes = int(float(row["num_nodes"]))
        mean_snir_by_state[use_snir][nodes] = _mean_snir(row["snir_histogram_json"])

    common_nodes = sorted(set(mean_snir_by_state[True]) & set(mean_snir_by_state[False]))
    assert len(common_nodes) >= 3, "Au moins trois points sont nécessaires pour comparer les courbes SNIR"

    gaps = [
        abs(mean_snir_by_state[True][nodes] - mean_snir_by_state[False][nodes])
        for nodes in common_nodes
    ]
    avg_gap = sum(gaps) / len(gaps)
    assert avg_gap >= MIN_SNIR_CURVE_DELTA_DB, (
        f"Courbes SNIR on/off quasi identiques (Δ moyen={avg_gap:.2f} dB)"
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_snir_toggle_combined_curves_diverge(tmp_path: Path) -> None:
    results_dir = tmp_path / "step1_combined_curves"
    args = [
        "--algos",
        "adr",
        "--with-snir",
        "true",
        "false",
        "--seeds",
        "1",
        "--nodes",
        "150",
        "300",
        "450",
        "--packet-intervals",
        "0.2",
        "--duration",
        "50",
        "--results-dir",
        str(results_dir),
    ]

    step1_matrix.main(args)

    csv_paths = sorted(results_dir.glob("**/*.csv"))
    assert csv_paths, "Aucun CSV généré pour comparer les courbes combinées SNIR"

    by_state: dict[bool, dict[int, dict[str, float]]] = {True: {}, False: {}}
    for path in csv_paths:
        with path.open(newline="", encoding="utf8") as handle:
            row = next(csv.DictReader(handle))

        use_snir = row["use_snir"] == "True"
        nodes = int(float(row["num_nodes"]))
        by_state[use_snir][nodes] = {
            "pdr": float(row["PDR"]),
            "der": float(row["DER"]),
            "collisions": float(row["collisions"]),
            "mean_snir": _mean_snir(row["snir_histogram_json"]),
        }

    common_nodes = sorted(set(by_state[True]) & set(by_state[False]))
    assert len(common_nodes) >= 3, "Au moins trois points sont nécessaires pour comparer les courbes SNIR"

    distances = []
    for nodes in common_nodes:
        on_metrics = by_state[True][nodes]
        off_metrics = by_state[False][nodes]
        distance = (
            abs(on_metrics["pdr"] - off_metrics["pdr"])
            + abs(on_metrics["der"] - off_metrics["der"])
            + abs(on_metrics["collisions"] - off_metrics["collisions"]) / 10.0
            + abs(on_metrics["mean_snir"] - off_metrics["mean_snir"]) / 10.0
        )
        distances.append(distance)

    avg_distance = sum(distances) / len(distances)
    assert avg_distance >= MIN_COMBINED_CURVE_DISTANCE, (
        "Courbes SNIR on/off trop similaires (distance combinée moyenne="
        f"{avg_distance:.2f})"
    )


def test_sync_snir_state_rejects_divergence() -> None:
    class StickyChannel:
        def __init__(self, value: bool) -> None:
            self._use_snir = value

        @property
        def use_snir(self) -> bool:
            return self._use_snir

        @use_snir.setter
        def use_snir(self, value: bool) -> None:
            return None

    class DummySimulator:
        def __init__(self) -> None:
            self.channel = StickyChannel(False)
            self.multichannel = type("Multi", (), {"channels": [StickyChannel(False)]})()

    simulator = DummySimulator()

    with pytest.raises(ValueError, match="effectif"):
        step1_experiments._sync_snir_state(simulator, True)


def test_multichannel_snir_consistency_rejects_requested_mismatch() -> None:
    class DummyChannel:
        def __init__(self, value: bool) -> None:
            self.use_snir = value

    class DummySimulator:
        def __init__(self) -> None:
            self.multichannel = type(
                "Multi", (), {"channels": [DummyChannel(False), DummyChannel(False)]}
            )()

    simulator = DummySimulator()

    with pytest.raises(ValueError, match="effectif"):
        step1_experiments._ensure_multichannel_snir_consistency(simulator, True)
