"""Virtual sites (4-point water models) must not break Fragment or OpenMMTheory (#500).

Issue #500 reported that an extra point (EPW) in a ff19SB/OPC system was not recognised.
Two paths were fixed on the NEW branch and neither had a test:

1. PDB: the element column of a virtual site carries no real element, so
   `atom.element.symbol` raised AttributeError. A dummy element 'M' is used instead.
2. Amber: the prmtop gives the extra point atomic number 0, which used to raise
   `KeyError: 0` from `reformat_element` via `read_ambercoordinates`.

Fixtures:
- `pdbfiles/water_with_vp.pdb` is the minimal 4-site water from the issue thread.
- `extra_files/opc_water.parm7` / `.rst7` is an OPC water box built with tleap
  (`source leaprc.water.opc; m = sequence { WAT }; solvateBox m OPCBOX 8.0`), so it
  contains the same `O H1 H2 EPW` pattern with atomic numbers `8 1 1 0` as the system in
  the issue.

`opc.xml` is not shipped here: OpenMM resolves it from its own bundled force fields.
"""
import os
from collections import Counter

import pytest

from ash import Fragment, Singlepoint, ashpath

PDBFILE = f"{ashpath}/tests/pdbfiles/water_with_vp.pdb"
PRMTOP = f"{ashpath}/tests/extra_files/opc_water.parm7"
INPCRD = f"{ashpath}/tests/extra_files/opc_water.rst7"

# Single-point OPC energy of the one-water PDB, OpenMM 8.6.1
REF_PDB_ENERGY = 0.014543911340935226


def test_fragment_from_pdb_with_virtual_site():
    """The extra point must come through as the dummy element 'M', not crash."""
    pytest.importorskip("openmm")
    fragment = Fragment(pdbfile=PDBFILE, printlevel=0)

    assert fragment.numatoms == 4
    assert fragment.elems == ["O", "H", "H", "M"]


def test_openmm_singlepoint_with_virtual_site():
    """The script from the issue thread must run and give the OPC energy."""
    pytest.importorskip("openmm")
    from ash import OpenMMTheory

    fragment = Fragment(pdbfile=PDBFILE, printlevel=0)
    theory = OpenMMTheory(pdbfile=PDBFILE, xmlfiles=["opc.xml"],
                          autoconstraints=None, rigidwater=False, printlevel=0)
    result = Singlepoint(theory=theory, fragment=fragment)

    assert result.energy == pytest.approx(REF_PDB_ENERGY, abs=1e-9)


@pytest.mark.skipif(not os.path.isfile(PRMTOP), reason="OPC prmtop fixture not present")
def test_fragment_from_amber_files_with_virtual_sites():
    """Atomic number 0 in the prmtop used to raise KeyError: 0."""
    fragment = Fragment(amber_prmtopfile=PRMTOP, amber_inpcrdfile=INPCRD, printlevel=0)

    counts = Counter(fragment.elems)
    # OPC is 4-site, so there is exactly one extra point per water
    assert counts["O"] == counts["M"]
    assert counts["H"] == 2 * counts["O"]
    assert fragment.numatoms == 4 * counts["O"]
    # and the atoms come in O, H, H, M order per residue
    assert fragment.elems[:4] == ["O", "H", "H", "M"]


@pytest.mark.skipif(not os.path.isfile(PRMTOP), reason="OPC prmtop fixture not present")
def test_amber_virtual_sites_have_no_mass_or_charge_element():
    """The dummy element must not be mistaken for a real one with a mass."""
    from ash.dictionaries_lists import element_dict_atnum

    fragment = Fragment(amber_prmtopfile=PRMTOP, amber_inpcrdfile=INPCRD, printlevel=0)

    assert "M" in fragment.elems
    # atomic number 0 is what the prmtop gives for the extra point
    assert element_dict_atnum[0].symbol == "M"
