from types import SimpleNamespace

from ash.modules import module_singlepoint


class FakeFragment:
    formula = "H2"
    label = "hydrogen"
    charge = 0
    mult = 1

    def set_energy(self, energy):
        self.energy = energy


class FakeTheory:
    theorytype = "QM"

    def cleanup(self):
        pass


class FakeReaction:
    def __init__(self):
        self.fragments = [FakeFragment(), FakeFragment()]
        self.stoichiometry = [-1, 1]
        self.unit = "Eh"
        self.energies = []
        self.orbital_dictionary = {}
        self.properties = {}

    def reset_energies(self):
        self.energies = []

    def calculate_reaction_energy(self):
        self.reaction_energy = 0.0


def test_singlepoint_reaction_reports_each_fragment_time(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module_singlepoint,
        "Singlepoint",
        lambda **kwargs: SimpleNamespace(energy=-1.0),
    )
    timings = []
    monkeypatch.setattr(
        module_singlepoint,
        "print_time_rel",
        lambda timestamp, modulename, **kwargs: timings.append(modulename),
    )

    reaction = FakeReaction()
    module_singlepoint.Singlepoint_reaction(theory=FakeTheory(), reaction=reaction)

    assert timings == [
        "Singlepoint_reaction fragment 1",
        "Singlepoint_reaction fragment 2",
        "Singlepoint_reaction",
    ]


def test_singlepoint_fragments_reports_each_fragment_time(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module_singlepoint,
        "Singlepoint",
        lambda **kwargs: SimpleNamespace(energy=-1.0),
    )
    timings = []
    monkeypatch.setattr(
        module_singlepoint,
        "print_time_rel",
        lambda timestamp, modulename, **kwargs: timings.append(modulename),
    )

    module_singlepoint.Singlepoint_fragments(
        theory=FakeTheory(), fragments=[FakeFragment(), FakeFragment()]
    )

    assert timings == [
        "Singlepoint_fragments fragment 1",
        "Singlepoint_fragments fragment 2",
        "Singlepoint_fragments",
    ]
