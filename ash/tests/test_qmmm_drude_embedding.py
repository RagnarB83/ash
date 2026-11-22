"""Regression tests for self-consistent OpenMM Drude QM/MM embedding."""
import shutil
import os
from pathlib import Path

import numpy as np
import pytest

from ash import Fragment, OpenMMTheory, QMMMTheory, Singlepoint, xTBTheory


pytestmark = pytest.mark.skipif(shutil.which("xtb") is None,
                                reason="xTB executable is required for the QM/MM regression")

BOHR_PER_ANGSTROM = 1.8897261254578281


def build_drude_qmmm(fragment):
    """Build a small native OpenMM Drude system around an xTB water region."""
    openmm = pytest.importorskip("openmm")
    from openmm import unit
    mm = OpenMMTheory(dummysystem=True, fragment=fragment, autoconstraints=None,
                      rigidwater=False, printlevel=0)
    mm.system.setParticleMass(3, 12.011 * unit.amu)
    mm.system.setParticleMass(4, 0.4 * unit.amu)
    mm.system.setParticleMass(5, 0.4 * unit.amu)
    mm.system.setParticleMass(6, 0.0 * unit.amu)

    # Neutral polarizable MM site: a positive core and a Drude shell.
    nbforce = mm.nonbonded_force
    nbforce.setParticleParameters(0, 0.3, 0.0, 0.0)  # QM oxygen's classical charge
    nbforce.setParticleParameters(3, 0.5, 0.0, 0.0)
    nbforce.setParticleParameters(4, -0.5, 0.0, 0.0)
    nbforce.setParticleParameters(5, -0.3, 0.0, 0.0)  # Drude shell attached to QM oxygen
    nbforce.setParticleParameters(6, -0.1, 0.0, 0.0)  # QM-attached virtual lone pair
    nbforce.addException(3, 4, 0.0, 0.0, 0.0)
    nbforce.addException(0, 5, 0.0, 0.0, 0.0)
    nbforce.addException(3, 5, -0.15, 0.0, 0.0)  # stale QM-shell/MM 1-4 term
    mm.system.setVirtualSite(6, openmm.ThreeParticleAverageSite(0, 1, 2, 0.5, 0.25, 0.25))

    drude_force = openmm.DrudeForce()
    drude_force.addParticle(4, 3, -1, -1, -1, -0.5, 0.001, 1.0, 1.0)
    drude_force.addParticle(5, 0, -1, -1, -1, -0.3, 0.001, 1.0, 1.0)
    mm.system.addForce(drude_force)
    mm.getatomcharges()

    qm = xTBTheory(xtbmethod="GFN2", numcores=1, printlevel=0)
    theory = QMMMTheory(qm_theory=qm, mm_theory=mm, qmatoms=[0, 1, 2],
                        fragment=fragment, embedding="drude", printlevel=0,
                        drude_scf_tolerance=2.0e-6)
    return theory


def make_fragment():
    # Water is the QM region. Atoms 3/4 are an MM Drude core/shell pair;
    # Atom 5 is a QM Drude shell; atom 6 is its virtual lone pair. Both must be removed.
    coords = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.93, 0.0],
                       [3.2, 0.0, 0.0], [3.2, 0.0, 0.0], [0.0, 0.0, 0.0],
                       [0.24, 0.31, 0.0]])
    return Fragment(elems=["O", "H", "H", "C", "H", "H", "H"], coords=coords,
                    charge=0, mult=1, conncalc=False, printlevel=0)


def test_drude_embedding_relaxes_mm_shell_and_matches_finite_difference(tmp_path, monkeypatch):
    openmm = pytest.importorskip("openmm")
    monkeypatch.chdir(tmp_path)
    fragment = make_fragment()
    theory = build_drude_qmmm(fragment)

    # The QM oxygen's classical Drude shell is removed; the MM-region shell stays active.
    force = next(f for f in theory.mm_theory.system.getForces()
                 if isinstance(f, openmm.DrudeForce))
    assert force.getNumParticles() == 1
    assert theory.drude_particle_indices == [4]
    assert theory.charges[5] == pytest.approx(0.0)
    assert theory.charges[6] == pytest.approx(0.0)
    assert 6 in theory.mm_point_site_indices

    initial_shell = fragment.coords[4].copy()
    result = Singlepoint(theory=theory, fragment=fragment, charge=0, mult=1,
                         Grad=True, printlevel=0)
    relaxed_shell = theory._drude_scf_coords[0] / BOHR_PER_ANGSTROM
    assert np.linalg.norm(relaxed_shell - initial_shell) > 1.0e-3
    assert theory.drude_scf_residual <= theory.drude_scf_tolerance
    assert np.allclose(fragment.coords[4], relaxed_shell)

    nonbonded = theory.mm_theory.nonbonded_force
    qm_shell_crossing_exception = None
    for i in range(nonbonded.getNumExceptions()):
        p1, p2, chargeprod, _, _ = nonbonded.getExceptionParameters(i)
        if {p1, p2} == {3, 5}:
            qm_shell_crossing_exception = chargeprod._value
    assert qm_shell_crossing_exception == pytest.approx(0.0)

    # Re-relax the Drude shell at each displaced nuclear geometry. This checks
    # the derivative of the variationally minimized QM/MM energy.
    step = 1.0e-3
    displaced_energies = []
    for sign in (1.0, -1.0):
        coords = fragment.coords.copy()
        coords[0, 0] += sign * step
        displaced_energies.append(theory.run(current_coords=coords, elems=fragment.elems,
                                             charge=0, mult=1, Grad=False))
    numeric = ((displaced_energies[0] - displaced_energies[1])
               / (2.0 * step * BOHR_PER_ANGSTROM))
    assert result.gradient[0, 0] == pytest.approx(numeric, abs=1.0e-4)
    # Explicit-coordinate evaluations must not mutate their input array.
    assert fragment.coords[0, 0] == pytest.approx(0.0)


def test_drude_embedding_requires_openmm_drude_force(tmp_path, monkeypatch):
    pytest.importorskip("openmm")
    monkeypatch.chdir(tmp_path)
    fragment = make_fragment()
    mm = OpenMMTheory(dummysystem=True, fragment=fragment, autoconstraints=None,
                      rigidwater=False, printlevel=0)
    mm.getatomcharges()
    qm = xTBTheory(xtbmethod="GFN2", numcores=1, printlevel=0)
    with pytest.raises(ValueError, match="DrudeForce"):
        QMMMTheory(qm_theory=qm, mm_theory=mm, qmatoms=[0, 1, 2],
                   fragment=fragment, embedding="drude", printlevel=0)


def test_drude_embedding_on_native_covalent_boundary_regression(tmp_path, monkeypatch):
    """Use the educt covalent-cut geometry from ASH's QMMM gradient regressions."""
    openmm = pytest.importorskip("openmm")
    from openmm import unit
    monkeypatch.chdir(tmp_path)
    from ash import ashpath

    qmatoms = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
    fragment = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/educt.xyz", charge=0,
                        mult=1, printlevel=0)
    center = np.mean(fragment.coords[qmatoms], axis=0)
    fragment.add_coords(["C", "H", "H"],
                        [center + [4.0, 0.0, 0.0], center + [4.0, 0.0, 0.0],
                         fragment.coords[2]], conn=False)
    fragment.coords[17] = fragment.coords[1] + [0.05, 0.03, 0.0]

    mm = OpenMMTheory(dummysystem=True, fragment=fragment, autoconstraints=None,
                      rigidwater=False, printlevel=0)
    for atom_index, mass in ((16, 12.011), (17, 0.4), (18, 0.4)):
        mm.system.setParticleMass(atom_index, mass * unit.amu)
    nbforce = mm.nonbonded_force
    nbforce.setParticleParameters(1, 0.2, 0.0, 0.0)  # charge on the MM boundary atom
    nbforce.setParticleParameters(16, 0.0, 0.0, 0.0)
    nbforce.setParticleParameters(17, -0.5, 0.0, 0.0)
    nbforce.setParticleParameters(18, -0.3, 0.0, 0.0)
    nbforce.addException(1, 17, 0.0, 0.0, 0.0)
    nbforce.addException(2, 18, 0.0, 0.0, 0.0)
    nbforce.addException(16, 18, -0.15, 0.0, 0.0)
    drude_force = openmm.DrudeForce()
    drude_force.addParticle(17, 1, -1, -1, -1, -0.5, 0.001, 1.0, 1.0)
    drude_force.addParticle(18, 2, -1, -1, -1, -0.3, 0.001, 1.0, 1.0)
    mm.system.addForce(drude_force)
    mm.getatomcharges()

    qm = xTBTheory(xtbmethod="GFN2", numcores=1, printlevel=0)
    theory = QMMMTheory(qm_theory=qm, mm_theory=mm, qmatoms=qmatoms,
                        fragment=fragment, embedding="drude", printlevel=0,
                        drude_scf_tolerance=5.0e-4)
    assert theory.boundaryatoms == {2: [1]}
    assert theory.drude_particle_indices == [17]
    assert theory.charges[18] == pytest.approx(0.0)

    result = Singlepoint(theory=theory, fragment=fragment, charge=0, mult=1,
                         Grad=True, printlevel=0)
    for site in (1, 17):
        pointcharge_index = list(theory.mmatoms).index(site)
        assert theory.pointcharges[pointcharge_index] == pytest.approx(0.0)
    assert 17 in theory.dipole_source_indices
    step = 1.0e-3
    energies = []
    for sign in (1.0, -1.0):
        coords = fragment.coords.copy()
        coords[2, 0] += sign * step
        energies.append(theory.run(current_coords=coords, elems=fragment.elems,
                                   charge=0, mult=1, Grad=False))
    numeric = ((energies[0] - energies[1])
               / (2.0 * step * BOHR_PER_ANGSTROM))
    assert result.gradient[2, 0] == pytest.approx(numeric, abs=1.0e-4)


def test_full_charmm_drude_qmmm_gradient_regression(tmp_path, monkeypatch):
    """Exercise a full CHARMM Drude PSF when local test assets are configured."""
    psf_file = os.environ.get("ASH_CHARMM_DRUDE_TEST_PSF")
    top_files = os.environ.get("ASH_CHARMM_DRUDE_TEST_TOP_FILES")
    parameter_files = os.environ.get("ASH_CHARMM_DRUDE_TEST_PARAM_FILES")
    charmm_cwd = os.environ.get("ASH_CHARMM_DRUDE_TEST_CWD")
    if not all((psf_file, top_files, parameter_files, charmm_cwd)):
        pytest.skip("Set ASH_CHARMM_DRUDE_TEST_PSF, _TOP_FILES, _PARAM_FILES and _CWD for the full CHARMM Drude regression")

    openmm = pytest.importorskip("openmm")
    import openmm.app as openmm_app
    from openmm import unit

    psf_file = Path(psf_file)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "toppar_drude").symlink_to(Path(charmm_cwd) / "toppar_drude")
    mm = OpenMMTheory(
        CHARMMfiles=True,
        psffile=str(psf_file),
        charmmtopfile=top_files.split(os.pathsep),
        charmmprmfile=parameter_files.split(os.pathsep),
        use_parmed=False,
        autoconstraints=None,
        rigidwater=False,
        printlevel=0,
    )
    crd_file = psf_file.with_suffix(".crd")
    coordinates = openmm_app.CharmmCrdFile(str(crd_file)).positions.value_in_unit(unit.angstrom)
    topology_atoms = list(mm.psf.topology.atoms())
    elems = [atom.element.symbol if atom.element is not None else "M" for atom in topology_atoms]
    fragment = Fragment(elems=elems, coords=np.asarray(coordinates), charge=0, mult=1,
                        conncalc=False, printlevel=0)
    drude_shells = {
        int(force.getParticleParameters(i)[0])
        for force in mm.system.getForces() if isinstance(force, openmm.DrudeForce)
        for i in range(force.getNumParticles())
    }
    qmatoms = [i for i, atom in enumerate(topology_atoms)
               if atom.residue.index == 0 and i not in drude_shells
               and not mm.system.isVirtualSite(i)]
    qm = xTBTheory(xtbmethod="GFN2", numcores=1, printlevel=0)
    theory = QMMMTheory(qm_theory=qm, mm_theory=mm, qmatoms=qmatoms,
                        fragment=fragment, embedding="drude", unusualboundary=True,
                        printlevel=0)

    result = Singlepoint(theory=theory, fragment=fragment, charge=0, mult=1,
                         Grad=True, printlevel=0)
    assert len(theory.drude_particle_indices) == 12
    assert theory.drude_scf_residual <= 1.0e-5
    assert result.gradient.shape == fragment.coords.shape

    # Relaxed-shell finite difference on the QM atom at the covalent boundary.
    # xTB's PC gradient noise is above 1e-5 for displaced geometries, so use
    # this looser SCF stopping threshold only for the two energy evaluations.
    theory.drude_scf_tolerance = 2.0e-4
    base_coords = fragment.coords.copy()
    step = 1.0e-3
    energies = []
    for sign in (1.0, -1.0):
        displaced = base_coords.copy()
        displaced[17, 0] += sign * step
        energies.append(theory.run(current_coords=displaced, elems=fragment.elems,
                                   charge=0, mult=1, Grad=False))
    numeric = ((energies[0] - energies[1])
               / (2.0 * step * BOHR_PER_ANGSTROM))
    assert result.gradient[17, 0] == pytest.approx(numeric, abs=5.0e-5)
