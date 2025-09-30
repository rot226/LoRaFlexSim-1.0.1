import pytest

from loraflexsim.launcher.channel import path_loss_hata_okumura, path_loss_oulu

def test_hata_okumura_rejects_non_positive_distance():
    with pytest.raises(ValueError):
        path_loss_hata_okumura(0.0, 127.5, 35.2)
    with pytest.raises(ValueError):
        path_loss_hata_okumura(-10.0, 127.5, 35.2)

def test_oulu_rejects_non_positive_distance():
    with pytest.raises(ValueError):
        path_loss_oulu(0.0, 128.95, 2.32, 1000.0, 2.0)
    with pytest.raises(ValueError):
        path_loss_oulu(-5.0, 128.95, 2.32, 1000.0, 2.0)
