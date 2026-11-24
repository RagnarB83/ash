"""Water HF energy and gradient through the Dalton Project Python API.

Requires the Dalton Project package from
``git+https://gitlab.com/daltonproject/daltonproject.git`` and a Dalton
executable on ``PATH``.
"""

import ash


water = ash.Fragment(
    elems=["O", "H", "H"],
    coords=[
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.958400],
        [0.000000, 0.926627, -0.239987],
    ],
    charge=0,
    mult=1,
)

theory = ash.DaltonProjectTheory(
    basis="STO-3G",
    qc_method="HF",
    program="dalton",
    label="water_hf",
    numcores=1,
)
result = ash.Singlepoint(fragment=water, theory=theory, Grad=True)
print("Energy (Eh):", result.energy)
print("Gradient (Eh/Bohr):")
print(result.gradient)
