from loraflexsim.scenarios.qos_cluster_presets import (
    ScenarioPreset,
    describe_presets,
    get_preset,
    list_presets,
)


def test_get_preset_case_insensitive():
    quick = get_preset("quick")
    assert isinstance(quick, ScenarioPreset)
    assert get_preset("QUICK") is quick


def test_list_presets_contains_expected_names():
    names = {preset.name for preset in list_presets()}
    assert {"quick", "baseline", "full"}.issubset(names)


def test_describe_presets_mentions_runs():
    description = describe_presets(5)
    assert "Préréglages disponibles" in description
    assert "quick" in description
    assert "A×C" in description
