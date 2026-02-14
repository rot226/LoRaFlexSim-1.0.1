from __future__ import annotations

from pathlib import Path

import article_c.compare_with_snir as compare_with_snir


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join([header, *rows]) + "\n"
    path.write_text(content, encoding="utf-8")


def test_compare_with_snir_by_size_multi_sizes(tmp_path: Path) -> None:
    step1_csv = tmp_path / "step1" / "results" / "aggregates" / "aggregated_results.csv"
    step2_csv = tmp_path / "step2" / "results" / "aggregates" / "aggregated_results.csv"

    for size, pdr_on, pdr_off in ((50, 0.92, 0.84), (100, 0.89, 0.79)):
        step1_rep = (
            tmp_path
            / "step1"
            / "results"
            / "by_size"
            / f"size_{size}"
            / "rep_1"
            / "aggregated_results.csv"
        )
        _write_csv(
            step1_rep,
            "network_size,algo,snir_mode,cluster,pdr_global_mean",
            [
                f"{size},adr,snir_on,all,{pdr_on}",
                f"{size},adr,snir_off,all,{pdr_off}",
            ],
        )

        step2_rep = (
            tmp_path
            / "step2"
            / "results"
            / "by_size"
            / f"size_{size}"
            / "rep_1"
            / "aggregated_results.csv"
        )
        _write_csv(
            step2_rep,
            "network_size,algo,snir_mode,cluster,throughput_success",
            [
                f"{size},adr,snir_on,all,{1500 - size}",
                f"{size},adr,snir_off,all,{1300 - size}",
            ],
        )

    output_dir = tmp_path / "plots"
    compare_with_snir.main(
        [
            "--step1-csv",
            str(step1_csv),
            "--step2-csv",
            str(step2_csv),
            "--source",
            "by_size",
            "--output-dir",
            str(output_dir),
            "--no-suptitle",
        ],
        close_figures=True,
    )

    assert (output_dir / "compare_pdr_snir.png").is_file()
    assert (output_dir / "compare_der_snir.png").is_file()
    assert (output_dir / "compare_throughput_snir.png").is_file()


def test_compare_with_snir_invalid_source_message() -> None:
    try:
        compare_with_snir.main(argv=[], source="invalid_source")
    except ValueError as exc:
        assert "source invalide pour ce module" in str(exc)
    else:
        raise AssertionError("La validation de source doit lever une ValueError.")
