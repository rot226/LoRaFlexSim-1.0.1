#!/usr/bin/env python3
"""Compare l'impact d'un filtre RF et d'une synchronisation imparfaite."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402


def main() -> None:
    base = Channel(shadowing_std=0)
    filt = Channel(
        shadowing_std=0,
        frontend_filter_order=2,
        frontend_filter_bw=100e3,
        frequency_offset_hz=40e3,
    )
    r1, s1 = base.compute_rssi(14.0, 100.0)
    r2, s2 = filt.compute_rssi(14.0, 100.0)
    print("Canal idéal : RSSI={:.2f} dBm SNR={:.2f} dB".format(r1, s1))
    print(
        "Avec filtre et désalignement : RSSI={:.2f} dBm SNR={:.2f} dB".format(
            r2, s2
        )
    )


if __name__ == "__main__":
    main()
