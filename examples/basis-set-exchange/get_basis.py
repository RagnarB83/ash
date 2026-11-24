"""Export selected BSE basis sets in ORCA and NWChem formats."""

import ash


elements = ["H", "O"]
basis_name = "def2-SVP"

ash.write_basis_from_bse(
    basis_name,
    filename=f"{basis_name}_H_O.orca",
    elements=elements,
    fmt="orca",
)
ash.write_basis_from_bse(
    basis_name,
    filename=f"{basis_name}_H_O.nwchem",
    elements=elements,
    fmt="nwchem",
)
