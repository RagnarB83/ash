"""Minimal GKS/X2C spin-orbit regression example with PySCF.

This follows the setup described in ASH issue #506. GKS with x2c=True
selects PySCF's generalized X2C reference, which includes spin-orbit coupling.
GPU4PySCF does not currently convert GKS objects, so this example uses CPU.
"""

from ash import Fragment, PySCFTheory, Singlepoint


fragment = Fragment(
    coords=[[0.0, 0.0, 0.0]],
    elems=["H"],
    charge=0,
    mult=2,
    printlevel=0,
)

theory = PySCFTheory(
    scf_type="GKS",
    basis="def2-SVP",
    functional="BLYP",
    x2c=True,
    platform="CPU",
    numcores=1,
    printlevel=0,
    gridlevel=0,
    do_pop_analysis=False,
    filename="pyscf_gks_x2c_blyp",
)

result = Singlepoint(fragment=fragment, theory=theory)
expected_energy = -0.49641185918856406
assert abs(result.energy - expected_energy) < 1e-5, (
    f"GKS/X2C regression changed: expected {expected_energy:.10f} Eh, "
    f"got {result.energy:.10f} Eh"
)
print("GKS/X2C BLYP regression passed:", result.energy, "Eh")
