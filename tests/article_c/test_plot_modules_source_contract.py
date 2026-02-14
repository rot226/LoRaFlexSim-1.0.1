from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import types

import article_c.make_all_plots as make_all_plots
from article_c.make_all_plots import PLOT_MODULES


def test_plot_modules_expose_main_source_parameter() -> None:
    missing: list[str] = []
    for module_paths in PLOT_MODULES.values():
        for module_path in module_paths:
            module = importlib.import_module(module_path)
            signature = inspect.signature(module.main)
            parameters = signature.parameters
            has_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            has_source = "source" in parameters
            if not has_source and not has_kwargs:
                missing.append(module_path)

    assert not missing, (
        "Modules sans `main(..., source=...)` contractuel: "
        + ", ".join(sorted(missing))
    )


def test_plot_modules_support_contractual_sources_and_last_effective_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    aggregate_csv = fixtures_root / "aggregates" / "aggregated_results.csv"
    by_size_csv = fixtures_root / "by_size" / "size_80" / "rep_0" / "aggregated_results.csv"
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    by_size_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_payload = (
        "network_size,algo,snir_mode,cluster,pdr_mean,success_rate_mean,reward_mean\n"
        "80,adr,snir_on,all,0.90,0.90,0.10\n"
    )
    aggregate_csv.write_text(csv_payload, encoding="utf-8")
    by_size_csv.write_text(csv_payload, encoding="utf-8")

    fake_modules: dict[str, object] = {}

    for module_paths in PLOT_MODULES.values():
        for module_path in module_paths:
            module = types.ModuleType(module_path)
            module.LAST_EFFECTIVE_SOURCE = ""

            def _main(*, source: str, _module=module, **_kwargs: object) -> None:
                if source == "aggregates":
                    assert aggregate_csv.exists()
                elif source == "by_size":
                    assert by_size_csv.exists()
                else:  # pragma: no cover - sécurité contractuelle
                    raise AssertionError(f"source inattendue: {source}")
                _module.LAST_EFFECTIVE_SOURCE = source

            module.main = _main
            fake_modules[module_path] = module

    monkeypatch.setattr(
        make_all_plots.importlib,
        "import_module",
        lambda module_path: fake_modules[module_path],
    )

    for source in ("aggregates", "by_size"):
        for module_paths in PLOT_MODULES.values():
            for module_path in module_paths:
                module = make_all_plots._run_plot_module(module_path, source=source)
                assert getattr(module, "LAST_EFFECTIVE_SOURCE") == source
