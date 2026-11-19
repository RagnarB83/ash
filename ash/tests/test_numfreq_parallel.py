"""NumFreq runmode='parallel' must agree with runmode='serial', properties included.

Two things are covered here.

1. The Hessian from runmode='parallel' must match runmode='serial'. This holds for a
   plain QM theory and for QMMMTheory, which issue #331 listed as not working because of
   an OpenMMTheory pickling problem.

2. Worker_par collected the dipole moment and the polarizability tensor by assigning a
   fresh dict for each one, so the second assignment replaced the first. A theory that
   provides both lost its dipole moment, Job_parallel returned an empty
   displacement_dipole_dictionary, and NumFreq(runmode='parallel') then produced no IR
   intensities at all while runmode='serial' produced them.

Stub theories are used for the property tests so no QM program is needed. They are
defined at module level so that they survive being sent to the worker processes.
"""
import shutil

import numpy as np
import pytest

from ash import Fragment, NumFreq
from ash.functions.functions_parallel import Job_parallel

ELEMS = ["O", "H", "H"]
COORDS = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]

# Diagonal spring constants and point charges for the stub theories
SPRING = np.array([0.9, 1.1, 1.3])
CHARGES = np.array([-0.8, 0.4, 0.4])


class StubBothProperties:
    """Stub QM theory that provides both a dipole moment and a polarizability tensor."""
    theorytype = "QM"
    numcores = 1
    theorynamelabel = "StubBoth"
    label = "StubBoth"
    filename = "stub"
    printlevel = 1

    def __init__(self, with_polarizability=True):
        self.with_polarizability = with_polarizability
        self._coords = None

    def run(self, current_coords=None, elems=None, Grad=False, label=None,
            charge=None, mult=None, numcores=None, **kwargs):
        coords = np.array(current_coords)
        self._coords = coords
        energy = float(0.5 * np.sum(SPRING * coords**2))
        if Grad:
            return energy, coords * SPRING
        return energy

    def get_dipole_moment(self):
        # Linear in the coordinates, so the dipole derivatives are non-zero
        return CHARGES.dot(self._coords)

    def get_polarizability_tensor(self):
        if not self.with_polarizability:
            raise RuntimeError("stub provides no polarizability")
        return np.eye(3) * 2.0

    def cleanup(self):
        pass


def water_fragment():
    return Fragment(elems=ELEMS, coords=COORDS, charge=0, mult=1, printlevel=0)


##############################################
# Property collection in the parallel worker
##############################################

def test_parallel_worker_keeps_both_properties():
    """Both properties must survive. Before the fix only the polarizability did."""
    fragments = [Fragment(elems=ELEMS, coords=COORDS, charge=0, mult=1,
                          label=f"disp{i}", printlevel=0) for i in range(2)]
    result = Job_parallel(fragments=fragments, theories=[StubBothProperties()],
                          numcores=2, allow_theory_parallelization=True, Grad=True,
                          printlevel=0, copytheory=True)

    assert len(result.displacement_dipole_dictionary) == len(fragments)
    assert len(result.displacement_polarizability_dictionary) == len(fragments)
    for properties in result.properties.values():
        assert "dipole_moment" in properties
        assert "polarizability" in properties


def test_parallel_worker_dipole_only_theory():
    """A theory with no polarizability must still report its dipole moment."""
    fragments = [Fragment(elems=ELEMS, coords=COORDS, charge=0, mult=1,
                          label="disp0", printlevel=0)]
    result = Job_parallel(fragments=fragments,
                          theories=[StubBothProperties(with_polarizability=False)],
                          numcores=1, allow_theory_parallelization=True, Grad=True,
                          printlevel=0, copytheory=True)

    assert len(result.displacement_dipole_dictionary) == 1
    assert len(result.displacement_polarizability_dictionary) == 0


def test_numfreq_parallel_gives_IR_intensities():
    """NumFreq(runmode='parallel') returned no IR intensities when both were provided."""
    result = NumFreq(theory=StubBothProperties(), fragment=water_fragment(),
                     runmode="parallel", numcores=2, printlevel=0, IR=True)

    assert result.IR_intensities is not None
    assert len(result.IR_intensities) == 3 * len(ELEMS)


def test_numfreq_IR_intensities_serial_vs_parallel():
    """The IR intensities must not depend on the runmode."""
    serial = NumFreq(theory=StubBothProperties(), fragment=water_fragment(),
                     runmode="serial", numcores=1, printlevel=0, IR=True)
    parallel = NumFreq(theory=StubBothProperties(), fragment=water_fragment(),
                       runmode="parallel", numcores=2, printlevel=0, IR=True)

    assert np.allclose(np.array(serial.IR_intensities),
                       np.array(parallel.IR_intensities), atol=1e-8)


##############################################
# The same thing with a real QM program
##############################################

@pytest.mark.skipif(shutil.which("orca") is None, reason="ORCA not available")
def test_parallel_worker_keeps_both_properties_orca():
    """Same as the stub test, with a QM program that really returns both properties.

    ORCA is asked for the polarizability through "%elprop Polar 1", so
    get_dipole_moment and get_polarizability_tensor both succeed. Before the fix only
    the polarizability came back and NumFreq(runmode='parallel') produced
    IR_intensities=None for this theory while runmode='serial' produced them.
    """
    from ash import ORCATheory

    theory = ORCATheory(orcasimpleinput="! HF STO-3G TightSCF",
                        orcablocks="%elprop Polar 1 end", numcores=1, printlevel=0)
    fragments = [Fragment(elems=ELEMS, coords=COORDS, charge=0, mult=1,
                          label=f"disp{i}", printlevel=0) for i in range(2)]

    result = Job_parallel(fragments=fragments, theories=[theory], numcores=2,
                          allow_theory_parallelization=True, Grad=True,
                          printlevel=0, copytheory=True)

    assert len(result.displacement_dipole_dictionary) == len(fragments)
    assert len(result.displacement_polarizability_dictionary) == len(fragments)
    for properties in result.properties.values():
        assert "dipole_moment" in properties
        assert "polarizability" in properties


@pytest.mark.skipif(shutil.which("orca") is None, reason="ORCA not available")
def test_numfreq_parallel_gives_IR_intensities_orca():
    """End-to-end with ORCA: parallel mode must produce IR intensities, not None."""
    from ash import ORCATheory

    def orca_theory():
        return ORCATheory(orcasimpleinput="! HF STO-3G TightSCF",
                          orcablocks="%elprop Polar 1 end", numcores=1, printlevel=0)

    serial = NumFreq(theory=orca_theory(), fragment=water_fragment(),
                     runmode="serial", numcores=1, printlevel=0, IR=True)
    parallel = NumFreq(theory=orca_theory(), fragment=water_fragment(),
                       runmode="parallel", numcores=2, printlevel=0, IR=True)

    assert serial.IR_intensities is not None
    assert parallel.IR_intensities is not None
    # Loose tolerance: the displacements are separate SCF runs, so serial and parallel
    # do not start from the same orbital guess and the intensities differ slightly
    assert np.allclose(np.array(serial.IR_intensities),
                       np.array(parallel.IR_intensities), atol=0.1)


##############################################
# Hessian agreement with a real QM program
##############################################

@pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb executable not available")
def test_numfreq_hessian_serial_vs_parallel_xtb():
    """Full Hessian from a plain QM theory must not depend on the runmode."""
    from ash import xTBTheory

    hessians = {}
    for runmode, numcores in [("serial", 1), ("parallel", 2)]:
        result = NumFreq(theory=xTBTheory(xtbmethod="GFN1", numcores=1),
                         fragment=water_fragment(), runmode=runmode,
                         numcores=numcores, printlevel=0)
        hessians[runmode] = np.array(result.hessian)

    assert hessians["serial"].shape == (3 * len(ELEMS), 3 * len(ELEMS))
    assert np.allclose(hessians["serial"], hessians["parallel"], atol=1e-8)


@pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb executable not available")
def test_numfreq_qmmm_hessian_serial_vs_parallel():
    """Partial Hessian for QMMMTheory must not depend on the runmode either.

    Issue #331 expected this to fail because OpenMMTheory could not be pickled.
    """
    import os

    from ash import OpenMMTheory, QMMMTheory, xTBTheory, ashpath

    pytest.importorskip("openmm")
    pdbfile = f"{ashpath}/tests/pdbfiles/educt.pdb"
    xmlfile = f"{ashpath}/tests/extra_files/gaff_educt.xml"
    xyzfile = f"{ashpath}/tests/xyzfiles/educt.xyz"
    for path in (pdbfile, xmlfile, xyzfile):
        if not os.path.isfile(path):
            pytest.skip("educt test files not present")

    qmatoms = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
    hessatoms = [2, 3, 6]

    hessians = {}
    for runmode, numcores in [("serial", 1), ("parallel", 2)]:
        fragment = Fragment(xyzfile=xyzfile, charge=0, mult=1, printlevel=0)
        mm = OpenMMTheory(pdbfile=pdbfile, xmlfiles=[xmlfile],
                          autoconstraints=None, rigidwater=False)
        qmmm = QMMMTheory(qm_theory=xTBTheory(xtbmethod="GFN1", numcores=1),
                          mm_theory=mm, qmatoms=qmatoms, fragment=fragment,
                          embedding="elstat")
        result = NumFreq(theory=qmmm, fragment=fragment, runmode=runmode,
                         numcores=numcores, hessatoms=hessatoms, printlevel=0)
        hessians[runmode] = np.array(result.hessian)

    assert hessians["serial"].shape == (3 * len(hessatoms), 3 * len(hessatoms))
    assert np.allclose(hessians["serial"], hessians["parallel"], atol=1e-8)
