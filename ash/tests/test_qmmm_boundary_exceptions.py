"""QM/MM elstat embedding must clear stale 1-4 exception chargeprods at the boundary (#524).

`QMMMTheory` with `embedding="elstat"` zeroes the QM region's charges. OpenMM stores 1-4
exception `chargeprod` values separately, computed once from the original charges at
System creation, and `update_charges` deliberately does not touch them. Any 1-4 pair
straddling the QM/MM boundary therefore kept contributing its original Coulomb term on
top of the same interaction already handled by the QM Hamiltonian through the embedded
point charges, double-counting it.

The fix is the `self.mm_theory.delete_exceptions(self.qmatoms)` call in the elstat path.
It zeroes `chargeprod` only, so the 1-4 Lennard-Jones across the boundary must survive --
that is asserted here too, since zeroing sigma/epsilon as well would be a different and
equally wrong behaviour.

Uses the native `educt` fixtures from ash/tests, the same system as the covalent-boundary
tests in ash/tests/hybrid-theory-tests.
"""
import shutil

import pytest

from ash import Fragment, ashpath

XYZFILE = f"{ashpath}/tests/xyzfiles/educt.xyz"
PDBFILE = f"{ashpath}/tests/pdbfiles/educt.pdb"
XMLFILE = f"{ashpath}/tests/extra_files/gaff_educt.xml"

# Same QM region as ash/tests/hybrid-theory-tests/test_organic_covalboundary_qm_mm.py
QMATOMS = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13]

pytestmark = pytest.mark.skipif(shutil.which("xtb") is None,
                                reason="xtb executable not available")


def nonbonded_force(openmmtheory):
    import openmm
    for force in openmmtheory.system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return force
    raise AssertionError("no NonbondedForce in the OpenMM system")


def boundary_exceptions(openmmtheory):
    """{(atom1, atom2): (chargeprod, sigma, epsilon)} for pairs straddling the boundary."""
    force = nonbonded_force(openmmtheory)
    crossing = {}
    for index in range(force.getNumExceptions()):
        p1, p2, chargeprod, sigma, epsilon = force.getExceptionParameters(index)
        if (p1 in QMATOMS) != (p2 in QMATOMS):
            crossing[tuple(sorted((p1, p2)))] = (chargeprod._value, sigma._value,
                                                 epsilon._value)
    return crossing


def build_mm():
    from ash import OpenMMTheory
    return OpenMMTheory(pdbfile=PDBFILE, xmlfiles=[XMLFILE], autoconstraints=None,
                        rigidwater=False, printlevel=0)


def build_qmmm(mm):
    from ash import QMMMTheory, xTBTheory
    fragment = Fragment(xyzfile=XYZFILE, charge=0, mult=1, printlevel=0)
    return QMMMTheory(qm_theory=xTBTheory(xtbmethod="GFN1", numcores=1), mm_theory=mm,
                      qmatoms=QMATOMS, fragment=fragment, embedding="elstat",
                      printlevel=0)


def test_boundary_chargeprods_are_nonzero_before_qmmm():
    """Guards the test itself: the MM system must start with stale values to clear."""
    pytest.importorskip("openmm")
    before = boundary_exceptions(build_mm())

    assert len(before) > 0
    assert sum(1 for cp, _, _ in before.values() if abs(cp) > 0.0) > 0


def test_boundary_chargeprods_cleared_by_elstat_embedding():
    """The bug: these stayed at their original values and double-counted Coulomb."""
    pytest.importorskip("openmm")
    mm = build_mm()
    before = boundary_exceptions(mm)
    build_qmmm(mm)
    after = boundary_exceptions(mm)

    # the same pairs must still be listed, just with no charge product
    assert set(after) == set(before)
    assert all(abs(cp) == 0.0 for cp, _, _ in after.values())


def test_boundary_lennard_jones_is_preserved():
    """delete_exceptions must not take the 1-4 LJ with it."""
    pytest.importorskip("openmm")
    mm = build_mm()
    before = boundary_exceptions(mm)
    build_qmmm(mm)
    after = boundary_exceptions(mm)

    for pair, (_, sigma, epsilon) in before.items():
        assert after[pair][1] == pytest.approx(sigma, abs=1e-12)
        assert after[pair][2] == pytest.approx(epsilon, abs=1e-12)
    # and the LJ is actually present, so the check above is not vacuous
    assert sum(1 for _, _, eps in after.values() if eps > 0.0) > 0


def test_qm_internal_charges_are_zeroed():
    """The per-particle half of elstat embedding, for completeness."""
    pytest.importorskip("openmm")
    mm = build_mm()
    build_qmmm(mm)
    force = nonbonded_force(mm)

    for atom in QMATOMS:
        charge, _, _ = force.getParticleParameters(atom)
        assert charge._value == pytest.approx(0.0, abs=1e-12)
