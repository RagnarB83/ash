"""Free-projection AFQMC with a single-determinant RHF trial state."""

from ash import DiceTheory, Fragment, PySCFTheory, Singlepoint

fragment = Fragment(
    coordsstring="H 0.000000 0.000000 0.000000\nH 0.000000 0.000000 0.740000",
    charge=0,
    mult=1,
)
pyscf = PySCFTheory(basis="STO-3G", scf_type="RHF", numcores=1)
dice = DiceTheory(
    pyscftheoryobject=pyscf,
    AFQMC=True,
    AFQMC_free_projection=True,
    numcores=4,
    FP_AFQMC_dt=0.05,
    FP_AFQMC_nsteps=60,
    FP_AFQMC_eneSteps=(20, 40, 50, 60),
    FP_AFQMC_errorTargets=(0.8e-3, 0.8e-3, 0.8e-3, 0.8e-3),
)
result = Singlepoint(fragment=fragment, theory=dice)
print(f"Free-projection AFQMC energy: {result.energy:.10f} Eh")
print(f"Free-projection AFQMC error: {dice.error} Eh")
