from types import SimpleNamespace

import numpy as np

from ash.interfaces import interface_geometric_new


def test_evaluation_label_distinguishes_hessian_displacements():
    engine = interface_geometric_new.ASHengineclass.__new__(interface_geometric_new.ASHengineclass)
    engine.EG_count = 21
    engine.calculation_directory = "geometric_OPTtraj.tmp/hessian/displace/001p"

    assert engine._evaluation_label() == "evaluation 22 (Hessian displacement)"


def test_evaluation_label_for_regular_geometry_calculation():
    engine = interface_geometric_new.ASHengineclass.__new__(interface_geometric_new.ASHengineclass)
    engine.EG_count = 3
    engine.calculation_directory = "geometric_OPTtraj.tmp"

    assert engine._evaluation_label() == "evaluation 4"


def test_regular_calculation_prints_evaluation_number(monkeypatch, capsys):
    class FakeFragment:
        elems = ["H"]

        def replace_coords(self, elems, coords, conn=False):
            pass

        def write_xyzfile(self, xyzfilename):
            pass

    class FakeTheory:
        def run(self, **kwargs):
            return -1.0, np.zeros((1, 3))

    monkeypatch.setattr(interface_geometric_new, "pygrep2", lambda *args, **kwargs: [])
    engine = interface_geometric_new.ASHengineclass.__new__(interface_geometric_new.ASHengineclass)
    engine.full_current_coords = []
    engine.fragment = FakeFragment()
    engine.printlevel = 1
    engine.calculation_directory = "geometric_OPTtraj.tmp/hessian/displace/001p"
    engine.EG_count = 0
    engine.iteration_count = 0
    engine.print_atoms_list = [0]
    engine.theory = FakeTheory()
    engine.M = SimpleNamespace(elem=["H"])
    engine.charge = 0
    engine.mult = 2

    engine.regular_calc(np.zeros((1, 3)))

    output = capsys.readouterr().out
    assert "Current geometry (Å) for evaluation 1 (Hessian displacement)" in output
    assert "Current geometry (Å) in step" not in output
    assert engine.EG_count == 1
