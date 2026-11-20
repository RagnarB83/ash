"""dipole_correction is ignored when chargeboundary_method is 'rcd', so warn about it.

The RCD charge-boundary scheme replaces the charge-shift scheme together with its dipole
correction: on the rcd path dipole_correction is never consulted, and setting it has no
effect on the energy or the gradient. Without a warning this silently looks like an
applied correction.
"""
import os
import shutil

import pytest

openmm = pytest.importorskip("openmm")

from ash import Fragment, OpenMMTheory, QMMMTheory, xTBTheory, ashpath
from ash.modules.module_oniom import ONIOMTheory

XYZFILE = os.path.join(ashpath, "tests", "xyzfiles", "educt.xyz")
PDBFILE = os.path.join(ashpath, "tests", "pdbfiles", "educt.pdb")
XMLFILE = os.path.join(ashpath, "tests", "extra_files", "gaff_educt.xml")
REGION1 = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13]

needs_xtb = pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb not in PATH")
needs_files = pytest.mark.skipif(not all(os.path.isfile(f) for f in (XYZFILE, PDBFILE, XMLFILE)),
                                 reason="test input files not present")

WARNING_TEXT = "The RCD scheme does not use the dipole correction"


@pytest.fixture
def fragment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Fragment(xyzfile=XYZFILE, charge=0, mult=1, printlevel=0)


def build_mm():
    return OpenMMTheory(pdbfile=PDBFILE, xmlfiles=[XMLFILE], autoconstraints=None,
                        rigidwater=False, printlevel=0)


@needs_xtb
@needs_files
@pytest.mark.parametrize("dipole_correction,expect_warning", [(True, True), (False, False)])
def test_qmmm_warns_only_when_dipole_correction_is_on(fragment, capsys, dipole_correction,
                                                      expect_warning):
    QMMMTheory(qm_theory=xTBTheory(xtbmethod="GFN1", printlevel=0), mm_theory=build_mm(),
               qmatoms=REGION1, fragment=fragment, embedding="elstat",
               chargeboundary_method="rcd", dipole_correction=dipole_correction,
               printlevel=1)
    out = capsys.readouterr().out
    assert (WARNING_TEXT in out) is expect_warning, out[-600:]


@needs_xtb
@needs_files
def test_qmmm_no_warning_for_the_shift_scheme(fragment, capsys):
    """With the charge-shift scheme the dipole correction is used, so no warning."""
    QMMMTheory(qm_theory=xTBTheory(xtbmethod="GFN1", printlevel=0), mm_theory=build_mm(),
               qmatoms=REGION1, fragment=fragment, embedding="elstat",
               chargeboundary_method="shift", dipole_correction=True, printlevel=1)
    assert WARNING_TEXT not in capsys.readouterr().out


@needs_xtb
@needs_files
def test_oniom_warns_for_rcd_with_dipole_correction(fragment, capsys):
    ONIOMTheory(theories_N=[xTBTheory(xtbmethod="GFN2", printlevel=0),
                            xTBTheory(xtbmethod="GFN1", printlevel=0)],
                regions_N=[REGION1, fragment.allatoms], fragment=fragment,
                regions_chargemult=[[0, 1], [0, 1]],
                fullregion_charge=0, fullregion_mult=1,
                embedding="elstat", chargeboundary_method="rcd", dipole_correction=True,
                printlevel=1)
    assert WARNING_TEXT in capsys.readouterr().out
