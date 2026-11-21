import numpy as np

from ash import Fragment, Singlepoint
from ash.modules.module_theory import MicroIterativeclass


class StubTheory:
    """Minimal ASH-like theory that follows the theory.run return convention:
    energy (float) without Grad, (energy, gradient) with Grad.
    """
    def __init__(self, numatoms=2):
        self.theorytype = "QM"
        self.numcores = 1
        self.energy = -1.5
        self.numatoms = numatoms

    def run(self, current_coords=None, elems=None, qm_elems=None, Grad=False, PC=False,
            numcores=None, charge=None, mult=None, label=None, **kwargs):
        if Grad:
            return self.energy, np.zeros((self.numatoms, 3))
        return self.energy


def make_fragment():
    return Fragment(elems=["H", "F"], coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.95]],
                    charge=0, mult=1, printlevel=0)


def test_theory_run_returns_bare_energy():
    theory = StubTheory()
    energy = theory.run(current_coords=np.zeros((2, 3)), elems=["H", "F"], charge=0, mult=1)
    assert isinstance(energy, float)
    energy, gradient = theory.run(current_coords=np.zeros((2, 3)), elems=["H", "F"],
                                  Grad=True, charge=0, mult=1)
    assert isinstance(energy, float)
    assert gradient.shape == (2, 3)


def test_microiterative_run_returns_same_shape_as_other_theories():
    # MicroIterativeclass.run must return the energy itself without Grad, not a 1-tuple,
    # so that it can be used wherever any other ASH theory can be used.
    theory = MicroIterativeclass(theory=StubTheory())
    energy = theory.run(current_coords=np.zeros((2, 3)), elems=["H", "F"], charge=0, mult=1)
    assert isinstance(energy, float)
    assert float(energy) == -1.5

    theory_grad = MicroIterativeclass(theory=StubTheory())
    energy, gradient = theory_grad.run(current_coords=np.zeros((2, 3)), elems=["H", "F"],
                                       Grad=True, charge=0, mult=1)
    assert isinstance(energy, float)
    assert gradient.shape == (2, 3)


def test_singlepoint_returns_ash_results():
    fragment = make_fragment()
    result = Singlepoint(fragment=fragment, theory=StubTheory(), charge=0, mult=1,
                         printlevel=0, result_write_to_disk=False)
    # Singlepoint returns an ASH_Results object, not a bare energy
    assert not isinstance(result, float)
    assert result.energy == -1.5
    assert result.gradient is None
    # The energy is also stored in the fragment
    assert fragment.energy == -1.5


def test_singlepoint_gradient_returns_ash_results_with_gradient():
    fragment = make_fragment()
    result = Singlepoint(fragment=fragment, theory=StubTheory(), Grad=True, charge=0, mult=1,
                         printlevel=0, result_write_to_disk=False)
    assert result.energy == -1.5
    assert result.gradient.shape == (2, 3)
