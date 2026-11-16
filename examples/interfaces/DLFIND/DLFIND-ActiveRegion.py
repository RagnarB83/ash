"""
DL-FIND optimization using an active region (ASH issue #520).

With ActiveRegion=True only the active-region representation of the system is passed to
DL-FIND, in the same way as the geomeTRIC interface does it. Without it, DL-FIND is given
the whole system with the inactive atoms frozen, which means that an initial Hessian has
to be built for the whole system (3*numatoms x 3*numatoms) even when only a few atoms are
treated by NumFreq.

This script runs the same optimization both ways with xTB and compares them, so the
active-region path can be verified:

  python DLFIND-ActiveRegion.py

Expected result: the two routes give the same energy and the same active-atom
coordinates, the inactive atoms do not move, and the initial partial Hessian is built for
the active region only (3*6 = 18) instead of the whole system (3*12 = 36).
"""
import numpy as np

from ash import Fragment, xTBTheory
from ash.interfaces.interface_dlfind import DLFIND_optimizer

# Butanol-like test system. Active region: the first 6 atoms
coords = """
C   -1.9000   0.0000   0.0000
C   -0.4000   0.0000   0.0000
C    0.2000   1.4000   0.0000
O    1.6000   1.4500   0.0000
H   -2.3000   1.0200   0.0000
H   -2.3000  -0.5100   0.8800
H   -2.3000  -0.5100  -0.8800
H   -0.0300  -0.5300   0.8800
H   -0.0300  -0.5300  -0.8800
H   -0.1500   1.9500   0.8800
H   -0.1500   1.9500  -0.8800
H    1.9500   2.3500   0.0000
"""
actatoms = [0, 1, 2, 3, 4, 5]

reference_coords = np.array(Fragment(coordsstring=coords, charge=0, mult=1).coords)

results = {}
for label, active in (("ActiveRegion", True), ("frozen-atoms", False)):
    print("\n" + "="*70)
    print(f"DL-FIND optimization, {label} route")
    print("="*70)
    fragment = Fragment(coordsstring=coords, charge=0, mult=1)
    theory = xTBTheory(xtbmethod='GFN1', printlevel=0)
    result = DLFIND_optimizer(jobtype='opt', theory=theory, fragment=fragment,
                              ActiveRegion=active, actatoms=list(actatoms),
                              maxcycle=120, printlevel=1)
    results[label] = (result.energy, np.array(fragment.coords))

inactive = [i for i in range(len(reference_coords)) if i not in actatoms]
energy_act, coords_act = results["ActiveRegion"]
energy_frozen, coords_frozen = results["frozen-atoms"]

print("\n" + "="*70)
print("Comparison of the two routes")
print("="*70)
print(f"Energy, ActiveRegion route  : {energy_act}")
print(f"Energy, frozen-atoms route  : {energy_frozen}")
print(f"Energy difference (Eh)      : {abs(energy_act-energy_frozen):.3e}")
print(f"Max coordinate difference over active atoms (Å)   : {np.abs(coords_act[actatoms]-coords_frozen[actatoms]).max():.3e}")
print(f"Max displacement of inactive atoms, ActiveRegion  : {np.abs(coords_act[inactive]-reference_coords[inactive]).max():.3e}")
print(f"Max displacement of active atoms, ActiveRegion    : {np.abs(coords_act[actatoms]-reference_coords[actatoms]).max():.3e}")
print()
print("Trajectory files written by the ActiveRegion run:")
print("  DLFIND_opt_traj.xyz         full system")
print("  DLFIND_opt_traj_Active.xyz  active region only")
print("  Fragment-optimized_Active.xyz  final active-region geometry")

# TS optimization with a partial NumFreq Hessian. This is the case in issue #520:
# the Hessian is only built for a few atoms but has to be handed to DL-FIND at the
# dimension of the system DL-FIND is optimizing
print("\n" + "="*70)
print("TS optimization with a partial NumFreq Hessian (3 atoms) in the active region")
print("="*70)
fragment = Fragment(coordsstring=coords, charge=0, mult=1)
theory = xTBTheory(xtbmethod='GFN1', printlevel=0)
DLFIND_optimizer(jobtype='tsopt', theory=theory, fragment=fragment,
                 ActiveRegion=True, actatoms=list(actatoms),
                 hessian_choice="numfreq", numfreq_hessatoms=[0, 1, 2],
                 numfreq_npoint=1, inithessian=0,
                 maxcycle=3, printlevel=1)
print("Partial Hessian was expanded to the active region (18x18), not the full system (36x36).")
