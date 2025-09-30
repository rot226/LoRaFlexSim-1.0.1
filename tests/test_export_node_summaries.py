from __future__ import annotations

import csv
import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mne3sd" / "export_node_summaries.py"
SPEC = importlib.util.spec_from_file_location("export_node_summaries", MODULE_PATH)
summaries = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader  # defensive: ensure module can be imported
SPEC.loader.exec_module(summaries)


def _write_csv(path, header, rows):
    path.write_text("\n".join([",".join(header), *[",".join(map(str, row)) for row in rows]]) + "\n")


def test_generate_tables_with_mixed_inputs(tmp_path):
    file_a = tmp_path / "class_density_metrics_nodes_20.csv"
    file_b = tmp_path / "class_density_metrics_nodes_50.csv"

    _write_csv(
        file_a,
        ["class", "replicate", "pdr", "collision_rate", "energy_per_node_J"],
        [
            ["A", 1, 0.90, 0.10, 0.05],
            ["A", 2, 0.80, 0.20, 0.06],
            ["B", 1, 0.70, 0.25, 0.07],
            ["B", 2, 0.65, 0.30, 0.08],
        ],
    )
    _write_csv(
        file_b,
        ["class", "nodes", "replicate", "pdr", "collision_rate", "energy_per_node_J"],
        [
            ["A", 50, 1, 0.72, 0.28, 0.09],
            ["B", 50, 1, 0.68, 0.32, 0.10],
        ],
    )

    output_csv = tmp_path / "summary.csv"
    output_tex = tmp_path / "summary.tex"

    args = Namespace(
        inputs=[str(tmp_path / "class_density_metrics_nodes_*.csv")],
        group_columns=["class"],
        metrics=("pdr", "collision_rate", "energy_per_node_J"),
        output_csv=output_csv,
        output_tex=output_tex,
        tex_caption="",
        tex_label="",
        precision=3,
    )

    rows = summaries.generate_tables(args)

    assert output_csv.exists()
    assert output_tex.exists()
    assert len(rows) == 4

    with output_csv.open() as handle:
        reader = csv.DictReader(handle)
        table = list(reader)

    first = table[0]
    assert first["nodes"] == "20"
    assert first["class"] == "A"
    assert float(first["pdr_mean"]) == pytest.approx(0.85)
    assert float(first["pdr_std"]) == pytest.approx(0.05)

    last = table[-1]
    assert last["nodes"] == "50"
    assert last["class"] == "B"
    assert float(last["collision_rate_mean"]) == pytest.approx(0.32)
    assert float(last["energy_per_node_J_mean"]) == pytest.approx(0.10)

    latex = output_tex.read_text()
    assert "\\begin{table}" in latex
    assert "Article" not in latex  # caption left empty
