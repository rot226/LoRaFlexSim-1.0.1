from __future__ import annotations

import importlib
import inspect

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
