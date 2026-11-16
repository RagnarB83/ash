# Basis Set Exchange

ASH can retrieve and export basis data through the optional
[`basis_set_exchange`](https://github.com/MolSSI-BSE/basis_set_exchange)
Python package. The package must be installed in the Python environment used
to run ASH. BSE data retrieval is local; the example does not download files
from a server at runtime.

Run `get_basis.py` to write `def2-SVP` for hydrogen and oxygen as ORCA and
NWChem input text. The NWChem output can be supplied to `PySCFTheory` with its
`basis_file` argument. The ORCA output can be included in an ORCA input basis
block. Pass another format supported by BSE using the `fmt` argument; use
`ash.get_basis_from_bse(...)` when the text should be handled in memory.
