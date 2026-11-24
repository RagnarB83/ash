"""Minimal ORCA CCSD(T)/CBS example with a locally dense basis sequence.

At each nominal cardinal, atom 0 uses a basis one cardinal larger than the
basis used for the rest of the molecule. Atom indices are zero-based and refer
to the element/coordinate ordering passed to the CBS theory at run time.
"""

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
    locally_dense_atoms=[0],
    locally_dense_cardinal_offset=1,
)

result = Singlepoint(
    fragment=fragment,
    theory=theory,
    Grad=False,
    printlevel=0,
    result_write_to_disk=False,
)

print("Locally dense CCSD(T)/CBS energy (Eh):", result.energy)
