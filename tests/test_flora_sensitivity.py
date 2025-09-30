import re
from pathlib import Path

from loraflexsim.launcher.channel import Channel


def parse_flora_sensitivity():
    path = Path('flora-master/src/LoRaPhy/LoRaAnalogModel.cc')
    text = path.read_text()
    table = {}
    current_sf = None
    for line in text.splitlines():
        m_sf = re.search(r'getLoRaSF\(\) == (\d+)', line)
        if m_sf:
            current_sf = int(m_sf.group(1))
            table[current_sf] = {}
            continue
        if current_sf is not None:
            m_bw = re.search(r'getLoRaBW\(\) == Hz\((\d+)\).*dBmW2mW\((-?\d+)\)', line)
            if m_bw:
                bw = int(m_bw.group(1))
                val = int(m_bw.group(2))
                table[current_sf][bw] = val
    return table


def test_flora_sensitivity_table_matches():
    expected = parse_flora_sensitivity()
    assert Channel.FLORA_SENSITIVITY == expected


def test_flora_noise_path_loading():
    path = Path('flora-master/src/LoRaPhy/LoRaAnalogModel.cc')
    ch = Channel(flora_noise_path=path)
    assert ch.flora_noise_table == parse_flora_sensitivity()
