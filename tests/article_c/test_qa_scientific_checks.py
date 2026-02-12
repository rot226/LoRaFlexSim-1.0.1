from __future__ import annotations

from pathlib import Path

from article_c.qa_scientific_checks import run_scientific_checks


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_scientific_checks_pass(tmp_path: Path) -> None:
    step1_csv = tmp_path / "step1.csv"
    step2_csv = tmp_path / "step2.csv"
    report_txt = tmp_path / "report.txt"
    report_csv = tmp_path / "report.csv"

    _write_csv(
        step1_csv,
        "network_size,algo,snir_mode,pdr_mean,energy_per_delivered_packet_mean\n"
        "50,adr,snir_on,0.90,0.80\n"
        "100,adr,snir_on,0.70,1.10\n"
        "150,adr,snir_on,0.55,1.35\n",
    )
    _write_csv(
        step2_csv,
        "network_size,algo,snir_mode,success_rate_mean,energy_per_success_mean\n"
        "50,adr,snir_on,0.88,0.70\n"
        "100,adr,snir_on,0.66,0.95\n"
        "150,adr,snir_on,0.50,1.40\n",
    )

    code, reports = run_scientific_checks(
        step1_csv=step1_csv,
        step2_csv=step2_csv,
        report_txt=report_txt,
        report_csv=report_csv,
    )

    assert code == 0
    assert any(item.verdict == "PASS" for item in reports)
    assert report_txt.exists()
    assert report_csv.exists()


def test_run_scientific_checks_fail_on_nan(tmp_path: Path) -> None:
    step1_csv = tmp_path / "step1.csv"
    step2_csv = tmp_path / "step2.csv"

    _write_csv(
        step1_csv,
        "network_size,algo,snir_mode,pdr_mean,energy_per_delivered_packet_mean\n"
        "50,adr,snir_on,0.90,0.80\n"
        "100,adr,snir_on,nan,1.10\n",
    )
    _write_csv(
        step2_csv,
        "network_size,algo,snir_mode,success_rate_mean,energy_per_success_mean\n"
        "50,adr,snir_on,0.88,0.70\n"
        "100,adr,snir_on,0.66,0.95\n",
    )

    code, reports = run_scientific_checks(
        step1_csv=step1_csv,
        step2_csv=step2_csv,
        report_txt=tmp_path / "report.txt",
        report_csv=tmp_path / "report.csv",
    )

    assert code == 1
    assert any(item.check_id == "nan_inf_absence" and item.verdict == "FAIL" for item in reports)
