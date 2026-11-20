"""Tests for tbliteTheory, the interface to the tblite library.

tbliteTheory is exported from ash but had no tests. The reference energies below were
produced with tblite 0.8.0 and agree with the xtb executable (6.8.0) to 3e-11 Eh, and the
gradients to 7e-7 au/Bohr, so they also serve as a cross-check between the two backends.
"""
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("tblite")

from ash import Fragment, Singlepoint, xTBTheory
from ash.interfaces.interface_xtb import tbliteTheory

COORDS = """
O       -1.377626260      0.000000000     -1.740199718
H       -1.377626260      0.759337000     -1.144156718
H       -1.377626260     -0.759337000     -1.144156718
"""
# tblite 0.8.0, water at the geometry above
REFERENCE_ENERGIES = {"GFN1-xTB": -5.7685026933, "GFN2-xTB": -5.0702533455}
# methodname in tblite -> xtbmethod for the xtb executable
BACKEND_PAIRS = {"GFN1-xTB": "GFN1", "GFN2-xTB": "GFN2"}


@pytest.fixture
def fragment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Fragment(coordsstring=COORDS, charge=0, mult=1, printlevel=0)


@pytest.mark.parametrize("method", sorted(REFERENCE_ENERGIES))
def test_energy_matches_reference(fragment, method):
    result = Singlepoint(theory=tbliteTheory(method=method, printlevel=0), fragment=fragment)
    assert result.energy == pytest.approx(REFERENCE_ENERGIES[method], abs=1e-8)


@pytest.mark.parametrize("method", sorted(REFERENCE_ENERGIES))
def test_gradient_shape_and_translational_invariance(fragment, method):
    result = Singlepoint(theory=tbliteTheory(method=method, printlevel=0), fragment=fragment,
                         Grad=True)
    gradient = np.array(result.gradient)
    assert gradient.shape == (fragment.numatoms, 3)
    # The sum of the forces on an isolated molecule has to vanish
    assert np.abs(gradient.sum(axis=0)).max() < 1e-8


@pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb executable not in PATH")
@pytest.mark.parametrize("method", sorted(BACKEND_PAIRS))
def test_agrees_with_the_xtb_executable(fragment, method):
    """The tblite library and the xtb executable must give the same energy and gradient."""
    library = Singlepoint(theory=tbliteTheory(method=method, printlevel=0),
                          fragment=fragment, Grad=True)
    executable = Singlepoint(theory=xTBTheory(runmode='inputfile',
                                              xtbmethod=BACKEND_PAIRS[method], printlevel=0),
                             fragment=fragment, Grad=True)
    assert library.energy == pytest.approx(executable.energy, abs=1e-8)
    assert np.allclose(np.array(library.gradient), np.array(executable.gradient), atol=1e-5)


def test_unknown_method_is_rejected(fragment):
    """A method name tblite does not know must not pass silently."""
    with pytest.raises((SystemExit, Exception)):
        Singlepoint(theory=tbliteTheory(method="NOT-A-METHOD", printlevel=0), fragment=fragment)
