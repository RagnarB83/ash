"""Run PM7 energy/gradient, COSMO, optimization, frequencies, and QMMM examples."""

import os

import numpy as np

from ash import Fragment, MOPACTheory, Singlepoint
from ash.interfaces.interface_geometric_new import geomeTRICOptimizer
from ash.modules.module_freq import NumFreq


def mopac_theory(filename, keywords=None):
    return MOPACTheory(
        mopacdir=os.environ.get("MOPACDIR"),
        method="PM7",
        filename=filename,
        precise=True,
        scfcrt=1e-10,
        keywords=keywords,
    )


def make_water(coords):
    return Fragment(elems=["H", "O", "H"], coords=coords, charge=0, mult=1)


initial_coords = [
    [-0.02110, -0.00200, 0.0],
    [0.83450, 0.45190, 0.0],
    [1.47690, -0.27300, 0.0],
]
native_relaxed_coords = [
    [-0.016619016, 0.009603882, 0.0],
    [0.833396536, 0.445864077, 0.0],
    [1.476815457, -0.260555447, 0.0],
]

# Official MOPAC example_scf.mop geometry.
water = make_water(initial_coords)
pm7 = mopac_theory("mopac_pm7")
scf_result = Singlepoint(theory=pm7, fragment=water, Grad=True)
print("PM7 SCF energy (Eh):", scf_result.energy)
print("PM7 gradient (Eh/Bohr):\n", scf_result.gradient)
print("PM7 atomic charges:", pm7.charges)
print("PM7 dipole (Debye):", pm7.properties.get("dipole"))

# Official MOPAC example_cosmo.mop adds EPS=78.4 to the same geometry.
cosmo = mopac_theory("mopac_cosmo", keywords="EPS=78.4")
cosmo_result = Singlepoint(theory=cosmo, fragment=water, Grad=True)
print("PM7 COSMO energy (Eh):", cosmo_result.energy)

# Generic ASH optimization uses MOPAC single-point gradients.
optimized_water = make_water(initial_coords)
opt_theory = mopac_theory("mopac_opt")
opt_result = geomeTRICOptimizer(
    theory=opt_theory,
    fragment=optimized_water,
    coordsystem="cart",
    force_coordsystem=True,
    maxiter=100,
)
print("Optimized energy (Eh):", opt_result.energy)
print("Optimized coordinates (Angstrom):\n", optimized_water.coords)

# Numerical frequencies at the geometry supplied in MOPAC's example_vibe.mop.
vibrational_water = make_water(native_relaxed_coords)
frequency_result = NumFreq(
    fragment=vibrational_water,
    theory=mopac_theory("mopac_numfreq"),
    npoint=2,
    displacement=0.005,
    IR=False,
)
print("Vibrational frequencies (cm^-1):", frequency_result.frequencies[-3:])

# Electrostatic point-charge embedding; gradients are returned for both QM and MM atoms.
qm_coords = np.array([[0.0, 0.0, 0.0], [0.0, -0.757, 0.587], [0.0, 0.757, 0.587]])
pc_coords = np.array([[2.5, 0.0, 1.5], [-2.5, 0.0, -1.5]])
pc_charges = np.array([0.3, -0.2])
qmmm_theory = mopac_theory("mopac_qmmm")
qmmm_energy, qm_gradient, pc_gradient = qmmm_theory.run(
    current_coords=qm_coords,
    current_MM_coords=pc_coords,
    MMcharges=pc_charges,
    qm_elems=["O", "H", "H"],
    charge=0,
    mult=1,
    Grad=True,
    PC=True,
)
print("QMMM energy (Eh):", qmmm_energy)
print("QMMM QM gradient (Eh/Bohr):\n", qm_gradient)
print("QMMM point-charge gradient (Eh/Bohr):\n", pc_gradient)
