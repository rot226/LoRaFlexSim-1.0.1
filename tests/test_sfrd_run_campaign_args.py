from __future__ import annotations

import sfrd.cli.run_campaign as run_campaign_module


def _base_args() -> list[str]:
    return [
        "--network-sizes",
        "80",
        "--replications",
        "1",
        "--seeds-base",
        "1",
        "--snir",
        "OFF,ON",
        "--algos",
        "ADR",
        "MixRA-H",
        "UCB",
        "--warmup-s",
        "0",
    ]


def test_parse_args_skip_algos_removes_mixra_h(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_campaign", *_base_args(), "--skip-algos", "MixRA-H"])

    args = run_campaign_module._parse_args()

    assert args.algos == ["ADR", "UCB"]


def test_parse_args_accepts_algos_without_mixra_h(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_campaign",
            "--network-sizes",
            "80",
            "--replications",
            "1",
            "--seeds-base",
            "1",
            "--snir",
            "OFF",
            "--algos",
            "ADR",
            "UCB",
            "--warmup-s",
            "0",
        ],
    )

    args = run_campaign_module._parse_args()

    assert args.algos == ["ADR", "UCB"]
