"""Tests de cohérence entre les profils FLORA et les paramètres FLoRa.

Ce test lit le fichier XML ``flora-master/simulations/energyConsumptionParameters.xml``
utilisé par le projet FLoRa (OMNeT++).  La localisation est documentée ici afin de
faciliter la mise à jour des valeurs si le projet amont change les paramètres.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from loraflexsim.launcher.energy_profiles import (
    DEFAULT_TX_CURRENT_MAP_A,
    FLORA_PROFILE,
)

# Le fichier XML exprime les courants en milliampères (mA).
XML_PARAMETERS_PATH = (
    Path(__file__).resolve().parent.parent
    / "flora-master"
    / "simulations"
    / "energyConsumptionParameters.xml"
)


@pytest.mark.parametrize(
    "xml_tag",
    [
        ("idleSupplyCurrent", "sleep_current_a"),
        ("receiverReceivingSupplyCurrent", "rx_current_a"),
    ],
)
def test_base_currents_match_flora_profile(xml_tag: tuple[str, str]) -> None:
    tag_name, profile_attr = xml_tag
    root = ET.parse(XML_PARAMETERS_PATH).getroot()
    element = root.find(tag_name)
    assert element is not None, f"Balise {tag_name} absente du fichier XML"
    value_a = float(element.attrib["value"]) / 1000.0
    assert getattr(FLORA_PROFILE, profile_attr) == pytest.approx(value_a)


def test_tx_currents_match_default_map() -> None:
    root = ET.parse(XML_PARAMETERS_PATH).getroot()
    tx_nodes = root.find("txSupplyCurrents")
    assert tx_nodes is not None, "Balise txSupplyCurrents absente du fichier XML"
    xml_map = {
        float(entry.attrib["txPower"]): float(entry.attrib["supplyCurrent"]) / 1000.0
        for entry in tx_nodes.findall("txSupplyCurrent")
    }
    assert set(xml_map) == set(DEFAULT_TX_CURRENT_MAP_A)
    assert FLORA_PROFILE.tx_current_map_a is not None
    assert set(xml_map) == set(FLORA_PROFILE.tx_current_map_a)
    for power_dbm, current_a in xml_map.items():
        assert DEFAULT_TX_CURRENT_MAP_A[power_dbm] == pytest.approx(current_a)
        assert FLORA_PROFILE.tx_current_map_a[power_dbm] == pytest.approx(current_a)


def test_flora_rx_window_duration_matches_reference() -> None:
    """La fenêtre d'écoute FLoRa doit durer exactement une seconde."""
    assert FLORA_PROFILE.rx_window_duration == pytest.approx(1.0)


def test_flora_profile_disables_transients() -> None:
    """Le profil FLORA ne doit pas comptabiliser les phases transitoires."""
    assert FLORA_PROFILE.include_transients is False


