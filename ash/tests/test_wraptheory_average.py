"""WrapTheory with theory_operators="average" must average the pointcharge gradient too.

The pointcharge gradient is accumulated as a sum over the theories, and the
average/ensemble branch divided the energy and the gradient by the number of theories but
not the pointcharge gradient. The force on the pointcharges then came out
len(theories) times too large while the energy and gradient were averages.

Stub theories are used so the test needs no QM program.
"""
import numpy as np
import pytest

from ash.modules.module_hybridtheory import WrapTheory

COORDS = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]])
ELEMS = ["O", "H", "H"]
MM_COORDS = np.array([[0.0, 0.0, 3.0], [0.0, 0.8, 3.6]])
MM_CHARGES = [-0.8, 0.4]


class StubTheory:
    """Returns fixed energy/gradient/pointcharge-gradient, like a QM theory would."""
    theorytype = "QM"
    numcores = 1

    def __init__(self, label, energy, scale):
        self.theorynamelabel = label
        self.label = label
        self.energy = energy
        self.scale = scale

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            mm_elems=None, elems=None, Grad=False, PC=False, numcores=None, restart=False,
            label=None, charge=None, mult=None):
        gradient = self.scale * np.ones((len(qm_elems), 3))
        if PC:
            pc_gradient = self.scale * np.ones((len(MMcharges), 3))
            return self.energy, gradient, pc_gradient
        return self.energy, gradient

    def cleanup(self):
        pass


def run_wrap(operators, theories, PC=True):
    wrap = WrapTheory(theories=theories, theory_operators=operators, printlevel=0)
    return wrap.run(current_coords=COORDS, current_MM_coords=MM_COORDS, MMcharges=MM_CHARGES,
                    qm_elems=ELEMS, Grad=True, PC=PC, charge=0, mult=1)


def test_average_averages_pc_gradient():
    t1 = StubTheory("t1", -1.0, 1.0)
    t2 = StubTheory("t2", -3.0, 3.0)
    energy, gradient, pc_gradient = run_wrap("average", [t1, t2])
    # energy and gradient are averages
    assert energy == pytest.approx(-2.0)
    assert np.allclose(gradient, 2.0 * np.ones((3, 3)))
    # the pointcharge gradient has to be the average too, not the sum (4.0)
    assert np.allclose(pc_gradient, 2.0 * np.ones((2, 3))), \
        f"pointcharge gradient not averaged: {pc_gradient}"


def test_average_of_three_theories():
    theories = [StubTheory(f"t{i}", -float(i), float(i)) for i in (1, 2, 3)]
    energy, gradient, pc_gradient = run_wrap("average", theories)
    assert energy == pytest.approx(-2.0)
    assert np.allclose(gradient, 2.0 * np.ones((3, 3)))
    assert np.allclose(pc_gradient, 2.0 * np.ones((2, 3)))


def test_pc_gradient_consistent_with_gradient():
    """The stubs return the same value for gradient and pc_gradient, so the combined
    results must agree as well whatever the number of theories."""
    for n in (1, 2, 4):
        theories = [StubTheory(f"t{i}", -1.0, 2.0) for i in range(n)]
        _, gradient, pc_gradient = run_wrap("average", theories)
        assert np.allclose(gradient[0], pc_gradient[0]), f"n={n}: {gradient[0]} vs {pc_gradient[0]}"


def test_sum_operators_still_sum():
    """With explicit +/- operators the pointcharge gradient must stay a signed sum."""
    t1 = StubTheory("t1", -1.0, 1.0)
    t2 = StubTheory("t2", -3.0, 3.0)
    energy, gradient, pc_gradient = run_wrap(["+", "-"], [t1, t2])
    assert energy == pytest.approx(2.0)
    assert np.allclose(gradient, -2.0 * np.ones((3, 3)))
    assert np.allclose(pc_gradient, -2.0 * np.ones((2, 3)))
