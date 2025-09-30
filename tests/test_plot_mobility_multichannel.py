from pathlib import Path
import importlib
import sys
import types


def test_plot(tmp_path, monkeypatch):
    # Use the real numpy module so that pandas can be imported.
    stubs_dir = Path(__file__).resolve().parent / 'stubs'
    path_no_stubs = [p for p in sys.path if p != str(stubs_dir)]
    monkeypatch.setattr(sys, 'path', path_no_stubs)
    monkeypatch.delitem(sys.modules, 'numpy', raising=False)
    monkeypatch.delitem(sys.modules, 'numpy.random', raising=False)
    np = importlib.import_module('numpy')
    sys.modules['numpy'] = np
    sys.modules['numpy.random'] = np.random

    import pandas as pd

    # Minimal matplotlib stand-ins.
    class _Figure:
        def savefig(self, filename, dpi=None):
            pass

        def tight_layout(self, *args, **kwargs):
            pass

    class _Tick:
        def __init__(self, text):
            self._text = text

        def get_text(self):
            return self._text

    class _Legend:
        def __init__(self, title=""):
            self._title = title

        def get_title(self):
            return _Tick(self._title)

    class _Axes:
        def __init__(self):
            self._labels = []
            self._legend = None

        def bar(self, *args, **kwargs):
            return []

        def set_xlabel(self, *args, **kwargs):
            pass

        def set_xticks(self, ticks):
            pass

        def set_xticklabels(self, labels, **kwargs):
            self._labels = list(labels)

        def set_ylabel(self, *args, **kwargs):
            pass

        def axhline(self, *args, **kwargs):
            pass

        def set_ylim(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass

        def bar_label(self, *args, **kwargs):
            pass

        def legend(self, *args, **kwargs):
            self._legend = _Legend(kwargs.get("title", ""))
            return self._legend

        def get_legend(self):
            return self._legend

        def get_xticklabels(self):
            return [_Tick(label) for label in self._labels]

    def _subplots(*args, **kwargs):
        return _Figure(), _Axes()

    def _close(fig=None):
        pass

    plt_module = types.ModuleType('pyplot')
    plt_module.subplots = _subplots
    plt_module.close = _close

    figure_module = types.ModuleType('figure')
    figure_module.Figure = _Figure

    matplotlib_module = types.ModuleType('matplotlib')
    matplotlib_module.pyplot = plt_module
    matplotlib_module.figure = figure_module

    monkeypatch.setitem(sys.modules, 'matplotlib', matplotlib_module)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', plt_module)
    monkeypatch.setitem(sys.modules, 'matplotlib.figure', figure_module)

    plot_module = importlib.import_module('scripts.plot_mobility_multichannel')

    def fake_savefig(self, filename, dpi=None, **kwargs):
        Path(filename).touch()

    monkeypatch.setattr(figure_module.Figure, 'savefig', fake_savefig)
    monkeypatch.setattr(plt_module, 'close', lambda fig=None: None)

    captured = {}
    orig_subplots = plt_module.subplots

    def fake_subplots(*args, **kwargs):
        fig, ax = orig_subplots(*args, **kwargs)
        captured['ax'] = ax
        return fig, ax

    monkeypatch.setattr(plt_module, 'subplots', fake_subplots)

    csv_path = Path('tests/data/mobility_multichannel_summary.csv')
    df = pd.read_csv(csv_path)
    tmp_csv = tmp_path / "mobility_multichannel_summary.csv"
    df.to_csv(tmp_csv, index=False)

    plot_module.plot(str(tmp_csv), tmp_path)

    for ext in ("png", "jpg", "svg", "eps"):
        assert (tmp_path / f"pdr_vs_scenario.{ext}").is_file()
    # The x-axis should list every tested node/channel combination once.
    labels = [tick.get_text() for tick in captured['ax'].get_xticklabels()]
    unique_labels = list(dict.fromkeys(labels))
    def _label(row):
        return (
            f"N={int(row['nodes'])}, C={int(row['channels'])}, static"
            if not row["mobility"]
            else f"N={int(row['nodes'])}, C={int(row['channels'])}, speed={row['speed']:.0f} m/s"
        )

    expected = [
        _label(row)
        for row in df.drop_duplicates(subset=["nodes", "channels", "mobility", "speed"])
            .to_dict("records")
    ]
    assert unique_labels == expected

    # Filtering by allowed pairs should reduce the scenarios.
    allowed = {(50, 1)}
    plot_module.plot(str(tmp_csv), tmp_path, allowed=allowed)
    labels = [tick.get_text() for tick in captured['ax'].get_xticklabels()]
    unique_labels = list(dict.fromkeys(labels))
    df_allowed = df[
        df[["nodes", "channels"]].apply(tuple, axis=1).isin(allowed)
    ].drop_duplicates(subset=["nodes", "channels", "mobility", "speed"])
    expected_allowed = [_label(row) for row in df_allowed.to_dict("records")]
    assert unique_labels == expected_allowed

    # Filtering by scenario names should select those scenarios.
    scenarios = ['n50_c1_static', 'n50_c1_mobile']
    plot_module.plot(str(tmp_csv), tmp_path, scenarios=set(scenarios))
    labels = [tick.get_text() for tick in captured['ax'].get_xticklabels()]
    unique_labels = list(dict.fromkeys(labels))
    df_scen = df[df['scenario'].isin(scenarios)].drop_duplicates(
        subset=["nodes", "channels", "mobility", "speed"]
    )
    expected_scenarios = [_label(row) for row in df_scen.to_dict("records")]
    assert unique_labels == expected_scenarios

    legend = captured['ax'].get_legend()
    if legend is not None:
        title = legend.get_title().get_text()
        # Legend should explain the abbreviations for nodes and channels.
        assert 'N:' in title and 'C:' in title
