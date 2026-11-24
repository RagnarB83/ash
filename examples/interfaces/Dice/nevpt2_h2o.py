"""SHCI/SC-NEVPT2 example using Dice and PySCF."""

from ash import DiceTheory, Fragment, PySCFTheory, Singlepoint

fragment = Fragment(
    coordsstring=(
        "O 0.000000 0.000000 0.000000\n"
        "H 0.758600 0.000000 0.504300\n"
        "H -0.758600 0.000000 0.504300"
    ),
    charge=0,
    mult=1,
)
pyscf = PySCFTheory(basis="6-31G", scf_type="RHF", numcores=1)
dice = DiceTheory(
    pyscftheoryobject=pyscf,
    NEVPT2=True,
    SHCI_active_space=[8, 6],
    initial_orbitals="RHF",
    SHCI_PTiter=0,
    NEVPT2_numdets=100,
    numcores=4,
)
result = Singlepoint(fragment=fragment, theory=dice)
print(f"Dice SC-NEVPT2 energy: {result.energy:.10f} Eh")
print(f"Dice SC-NEVPT2 stochastic error: {dice.error} Eh")
