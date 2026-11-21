"""Helpers for retrieving basis sets from the Basis Set Exchange (BSE).

The ``basis_set_exchange`` package is optional. Basis data are resolved from
the local BSE data package; this module does not download files at runtime.
"""

from pathlib import Path


def get_basis_from_bse(basis_name, elements=None, fmt="nwchem", header=True):
    """Return a basis set in a format supported by Basis Set Exchange.

    Args:
        basis_name (str): Basis set name, for example ``"def2-SVP"``.
        elements (iterable, optional): Element symbols or atomic numbers to
            include. If omitted, BSE returns data for all elements in the
            basis set.
        fmt (str): BSE output format. Examples include ``"nwchem"`` (usable
            as a PySCF ``basis_file``) and ``"orca"``.
        header (bool): Include BSE's provenance header in the returned text.

    Returns:
        str: The formatted basis set text.

    Raises:
        ImportError: If the optional ``basis_set_exchange`` package is absent.
        ValueError: If the basis name or output format is invalid.
    """
    if not isinstance(basis_name, str) or not basis_name.strip():
        raise ValueError("basis_name must be a non-empty string")
    if not isinstance(fmt, str) or not fmt.strip():
        raise ValueError("fmt must be a non-empty BSE output format")

    try:
        import basis_set_exchange as bse
    except ImportError as exc:
        raise ImportError(
            "BSE basis retrieval requires the optional 'basis_set_exchange' "
            "package. Install it in the ASH Python environment."
        ) from exc

    available_formats = bse.get_formats()
    if fmt not in available_formats:
        formats = ", ".join(sorted(available_formats))
        raise ValueError(f"Unknown BSE format {fmt!r}. Available formats: {formats}")

    try:
        return bse.get_basis(
            basis_name.strip(), elements=elements, fmt=fmt, header=header
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Could not retrieve basis set {basis_name!r} in BSE format {fmt!r}: {exc}"
        ) from exc


def write_basis_from_bse(basis_name, filename, elements=None, fmt="nwchem", header=True):
    """Write BSE basis text to ``filename`` and return its path.

    Args:
        basis_name (str): Basis set name in BSE.
        filename (str or pathlib.Path): Destination file.
        elements (iterable, optional): Element symbols or atomic numbers to
            include.
        fmt (str): BSE output format, defaulting to NWChem format.
        header (bool): Include BSE's provenance header.

    Returns:
        pathlib.Path: The destination path.
    """
    path = Path(filename)
    basis_text = get_basis_from_bse(
        basis_name, elements=elements, fmt=fmt, header=header
    )
    path.write_text(basis_text, encoding="utf-8")
    return path
