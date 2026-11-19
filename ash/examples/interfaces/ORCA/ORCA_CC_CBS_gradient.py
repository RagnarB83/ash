"""Canonical two-basis CCSD(T)/CBS energy and analytic gradient example."""

from ash import Fragment, ORCA_CC_CBS_Theory, Singlepoint


fragment = Fragment(
    diatomic="H2",
    diatomic_bondlength=0.74,
    charge=0,
    mult=1,
    printlevel=0,
)

theory = ORCA_CC_CBS_Theory(
    elements=fragment.elems,
    basisfamily="cc",
    cardinals=[2, 3],
    numcores=1,
)

result = Singlepoint(
    fragment=fragment,
    theory=theory,
    Grad=True,
    printlevel=0,
    result_write_to_disk=False,
)
energy = result.energy
gradient = result.gradient

assert gradient.shape == (fragment.numatoms, 3)
assert abs(gradient[:, 2].sum()) < 1e-8
print("CCSD(T)/CBS energy (Eh):", energy)
print("CCSD(T)/CBS gradient (Eh/bohr):\n", gradient)
