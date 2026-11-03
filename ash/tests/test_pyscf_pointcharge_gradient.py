"""
Point-charge (MM) gradient of PySCFTheory against finite differences of the energy.

Covers closed-shell (RHF) and open-shell (UHF) QM regions: for unrestricted references the
electronic part of the point-charge gradient needs the total (alpha + beta) density.
"""
import numpy as np
import pytest

pytest.importorskip("pyscf")

from ash.interfaces.interface_pyscf import PySCFTheory
from ash.constants import ang2bohr

# QM region (Angstrom)
QM = {
    "RHF": (["O", "H", "H"],
            np.array([[0.000, 0.000, 0.000],
                      [0.757, 0.586, 0.000],
                      [-0.757, 0.586, 0.000]]), 0, 1),
    "UHF": (["O", "H"],
            np.array([[0.000, 0.000, 0.000],
                      [0.000, 0.000, 0.970]]), 0, 2),
}
# a TIP3P-like water as point charges, 3 Angstrom away from the QM region
MM_COORDS = np.array([[2.800, 0.300, 0.400],
                      [3.300, 1.050, 0.700],
                      [3.300, -0.450, 0.700]])
MM_CHARGES = np.array([-0.834, 0.417, 0.417])


def run(scf_type, mm_coords, grad):
    elems, coords, charge, mult = QM[scf_type]
    theory = PySCFTheory(scf_type=scf_type, basis="def2-SVP", printlevel=0,
                         conv_tol=1e-12, scf_maxiter=200, do_pop_analysis=False)
    return theory.run(current_coords=coords, current_MM_coords=mm_coords, MMcharges=MM_CHARGES,
                      qm_elems=elems, Grad=grad, PC=True, charge=charge, mult=mult)


@pytest.mark.parametrize("scf_type", ["RHF", "UHF"])
def test_pointcharge_gradient_matches_finite_differences(scf_type):
    _, _, pcgrad = run(scf_type, MM_COORDS, True)          # Eh/Bohr
    h = 1e-3                                                # Angstrom
    fd = np.zeros_like(MM_COORDS)
    for i in range(len(MM_COORDS)):
        for k in range(3):
            plus, minus = MM_COORDS.copy(), MM_COORDS.copy()
            plus[i, k] += h
            minus[i, k] -= h
            fd[i, k] = (run(scf_type, plus, False) - run(scf_type, minus, False)) / (2 * h)
    fd /= ang2bohr                                          # Eh/Angstrom -> Eh/Bohr
    assert np.max(np.abs(np.asarray(pcgrad) - fd)) < 1e-6
