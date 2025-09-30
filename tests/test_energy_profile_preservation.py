from loraflexsim.launcher.energy_profiles import EnergyProfile
from loraflexsim.launcher.node import Node
from loraflexsim.launcher.gateway import Gateway


class FalseyProfile(EnergyProfile):
    def __bool__(self):
        return False


def test_falsey_profile_preserved():
    profile = FalseyProfile()

    node = Node(0, 0.0, 0.0, 7, 14, energy_profile=profile)
    assert node.profile is profile

    gw = Gateway(0, 0.0, 0.0, energy_profile=profile)
    assert gw.profile is profile
