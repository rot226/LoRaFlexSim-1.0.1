import tempfile
import configparser
import loraflexsim.run as run


def test_mu_send_from_config(monkeypatch):
    cfg = tempfile.NamedTemporaryFile('w+', delete=False)
    cp = configparser.ConfigParser()
    cp['simulation'] = {'mu_send': '12.5'}
    cp.write(cfg)
    cfg.flush()

    received = {}

    def fake_simulate(nodes, gateways, mode, interval, steps, *a, first_interval=None, **k):
        received['interval'] = interval
        received['first_interval'] = first_interval
        return 0, 0, 0, 0, 0, 0

    monkeypatch.setattr(run, 'simulate', fake_simulate)
    run.main(['--config', cfg.name, '--nodes', '1', '--gateways', '1', '--steps', '10'])

    assert received['interval'] == 12.5
    assert received['first_interval'] == 12.5
