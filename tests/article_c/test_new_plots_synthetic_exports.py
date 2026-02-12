from __future__ import annotations

import csv
import importlib
from pathlib import Path

from PIL import Image
import pytest

from article_c.common.plot_style import MIN_EXPORT_DPI


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_step1_aggregated_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for snir_mode in ("snir_on", "snir_off"):
        for network_size in (80, 160):
            for algo in ("adr", "mixra_h", "mixra_opt", "apra", "aimi"):
                rows.append(
                    {
                        "network_size": network_size,
                        "algo": algo,
                        "snir_mode": snir_mode,
                        "cluster": "all",
                        "mixra_opt_fallback": "false",
                        "pdr_mean": 0.9,
                        "throughput_bps_mean": 1200.0,
                        "sent_mean": 120.0,
                        "received_mean": 90.0,
                        "mean_toa_s_mean": 0.12,
                    }
                )
    _write_csv(path, rows)


def _build_step1_raw_packets_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for snir_mode in ("snir_on", "snir_off"):
        for network_size in (80, 160):
            for algo in ("adr", "mixra_h", "mixra_opt"):
                for packet_id in (1, 2):
                    rows.extend(
                        [
                            {
                                "packet_id": packet_id,
                                "sf_selected": 7,
                                "rssi_dbm": -110.0,
                                "network_size": network_size,
                                "algo": algo,
                                "snir_mode": snir_mode,
                                "replication": 1,
                            },
                            {
                                "packet_id": packet_id,
                                "sf_selected": 7,
                                "rssi_dbm": -111.0,
                                "network_size": network_size,
                                "algo": algo,
                                "snir_mode": snir_mode,
                                "replication": 1,
                            },
                        ]
                    )
    _write_csv(path, rows)


def _build_step2_aggregated_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for snir_mode in ("snir_on", "snir_off"):
        for network_size in (80, 160):
            for algo in ("adr", "mixra_h", "mixra_opt", "ucb1_sf"):
                rows.append(
                    {
                        "network_size": network_size,
                        "algo": algo,
                        "snir_mode": snir_mode,
                        "cluster": "all",
                        "pdr_global_mean": 0.82,
                        "sent_mean": 100.0,
                        "received_mean": 70.0,
                        "mean_toa_s_mean": 0.11,
                    }
                )
    _write_csv(path, rows)


def _build_step2_learning_curve_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for network_size in (80, 160):
        for algo in ("adr", "mixra_h", "ucb1_sf"):
            for round_id in (1, 2, 3):
                for replication_id in (0, 1):
                    rows.append(
                        {
                            "network_size": network_size,
                            "algo": algo,
                            "round": round_id,
                            "avg_reward": 0.4 + 0.05 * round_id + 0.01 * replication_id,
                        }
                    )
    _write_csv(path, rows)


def _build_step2_sf_selection_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for network_size in (80, 160):
        for round_id in (1, 2, 3):
            rows.extend(
                [
                    {
                        "network_size": network_size,
                        "round": round_id,
                        "sf": 7,
                        "selection_prob": 0.65,
                    },
                    {
                        "network_size": network_size,
                        "round": round_id,
                        "sf": 8,
                        "selection_prob": 0.35,
                    },
                ]
            )
    _write_csv(path, rows)


PLOT_CASES = [
    (
        "article_c.step1.plots.plot_S_new1_pdr_cluster_paper",
        "plot_S_new1_pdr_cluster_paper",
        "article_c/step1/results/aggregated_results.csv",
        _build_step1_aggregated_csv,
    ),
    (
        "article_c.step1.plots.plot_S_new2_throughput_cluster_global",
        "plot_S_new2_throughput_cluster_global",
        "article_c/step1/results/aggregated_results.csv",
        _build_step1_aggregated_csv,
    ),
    (
        "article_c.step1.plots.plot_S_new3_energy_per_delivered_packet",
        "plot_S_new3_energy_per_delivered_packet",
        "article_c/step1/results/aggregated_results.csv",
        _build_step1_aggregated_csv,
    ),
    (
        "article_c.step1.plots.plot_S_new4_interference_realism",
        "plot_S_new4_interference_realism",
        "article_c/step1/results/raw_packets.csv",
        _build_step1_raw_packets_csv,
    ),
    (
        "article_c.step2.plots.plot_R_new1_pdr_global",
        "plot_R_new1_pdr_global",
        "article_c/step2/results/aggregated_results.csv",
        _build_step2_aggregated_csv,
    ),
    (
        "article_c.step2.plots.plot_R_new2_energy_per_packet",
        "plot_R_new2_energy_per_packet",
        "article_c/step2/results/aggregated_results.csv",
        _build_step2_aggregated_csv,
    ),
    (
        "article_c.step2.plots.plot_R_new3_learning_curve",
        "plot_R_new3_learning_curve",
        "article_c/step2/results/learning_curve.csv",
        _build_step2_learning_curve_csv,
    ),
    (
        "article_c.step2.plots.plot_R_new4_sf_policy",
        "plot_R_new4_sf_policy",
        "article_c/step2/results/sf_selection_by_round.csv",
        _build_step2_sf_selection_csv,
    ),
]


@pytest.mark.parametrize(
    ("module_path", "output_stem", "relative_csv", "builder"),
    PLOT_CASES,
)
def test_new_plots_export_png_with_legend_without_suptitle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    output_stem: str,
    relative_csv: str,
    builder,
) -> None:
    module = importlib.import_module(module_path)

    csv_path = tmp_path / relative_csv
    builder(csv_path)

    fake_file = tmp_path / Path(*module_path.split("."))
    fake_file = fake_file.with_suffix(".py")
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("# synthetic test module location\n", encoding="utf-8")

    legend_checked = {"value": False}

    def _assert_legend_present(fig, figure_name: str) -> None:
        del figure_name
        has_legend = bool(fig.legends) or any(ax.get_legend() is not None for ax in fig.axes)
        assert has_legend, "Légende absente dans la figure exportée."
        suptitle = getattr(fig, "_suptitle", None)
        assert suptitle is None or not suptitle.get_text().strip(), "Un titre global est présent."
        legend_checked["value"] = True

    def _save_figure(fig, output_dir: Path, stem: str, use_tight: bool = False, **_: object) -> None:
        del use_tight
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.png", dpi=MIN_EXPORT_DPI)

    monkeypatch.setattr(module, "__file__", str(fake_file))
    monkeypatch.setattr(module, "assert_legend_present", _assert_legend_present)
    monkeypatch.setattr(module, "save_figure", _save_figure)

    module.main()

    step_root = Path(relative_csv).parts[1]
    output_png = tmp_path / "article_c" / step_root / "plots" / "output" / f"{output_stem}.png"
    assert output_png.exists(), "PNG non généré."
    assert legend_checked["value"], "La validation de légende n'a pas été exécutée."

    with Image.open(output_png) as image:
        width, height = image.size
        assert width >= 400 and height >= 250, "Dimensions PNG trop petites."
        dpi = image.info.get("dpi")

    assert dpi is not None, "DPI absent dans les métadonnées PNG."
    min_dpi = min(float(dpi[0]), float(dpi[1]))
    assert min_dpi >= MIN_EXPORT_DPI - 1, "DPI inférieur au minimum attendu."
