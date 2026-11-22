from ash import *
from ash.interfaces.interface_OpenMM import OpenMM_MDclass, _normalize_charmm_files
import os
import pytest


def test_openmm_md_rejects_unknown_trajectory_option():
    # Validate before dereferencing the fragment or theory, so this test needs
    # no OpenMM system construction and cannot launch an MD simulation.
    with pytest.raises(SystemExit):
        OpenMM_MDclass(fragment=None, theory=None, trajectory_file_option="NetCDF")


def test_normalize_charmm_file_arguments(tmp_path):
    top = tmp_path / "top.rtf"
    prm = tmp_path / "params.prm"
    assert _normalize_charmm_files(top, [prm]) == [str(top), str(prm)]
    with pytest.raises(TypeError, match="Each CHARMM"):
        _normalize_charmm_files([top, None])


def test_openmm_charmm_drude_file_lists(monkeypatch):
    """Load a real CHARMM Drude PSF when paths are supplied by the test environment."""
    psf = os.environ.get("ASH_CHARMM_DRUDE_TEST_PSF")
    top_files = os.environ.get("ASH_CHARMM_DRUDE_TEST_TOP_FILES")
    parameter_files = os.environ.get("ASH_CHARMM_DRUDE_TEST_PARAM_FILES")
    working_directory = os.environ.get("ASH_CHARMM_DRUDE_TEST_CWD")
    if not all((psf, top_files, parameter_files)):
        pytest.skip("Set ASH_CHARMM_DRUDE_TEST_PSF, _TOP_FILES and _PARAM_FILES for the CHARMM Drude integration test")

    openmm = pytest.importorskip("openmm")
    if working_directory:
        monkeypatch.chdir(working_directory)
    theory = OpenMMTheory(
        CHARMMfiles=True,
        psffile=psf,
        charmmtopfile=top_files.split(os.pathsep),
        charmmprmfile=parameter_files.split(os.pathsep),
        use_parmed=False,
        autoconstraints=None,
        rigidwater=False,
        printlevel=0,
    )
    drude_forces = [force for force in theory.system.getForces()
                    if isinstance(force, openmm.DrudeForce)]
    assert theory.system.getNumParticles() > 0
    assert len(drude_forces) == 1
    assert drude_forces[0].getNumParticles() > 0


#Read solvated PDB-file, create OpenMMTheory job and run MM singlepoint
def test_openmm_basic():

    #Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment or PDB-file)
    pdbfile=f"{ashpath}/tests/pdbfiles/1aki_solvated.pdb"
    fragment=Fragment(pdbfile=pdbfile)

    #Creating new OpenMM object from OpenMM full system file
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"], pdbfile=pdbfile, periodic=True,
            autoconstraints=None, rigidwater=False)
    #Singlepoint MM energy
    Singlepoint(theory=omm, fragment=fragment, Grad=True)

#Read raw PDB-file, fix using pdbfixer, setup using Modeller and optimize
def test_openmm_modeller():

    pdbfile=f"{ashpath}/tests/pdbfiles/1aki.pdb"

    #Setting up new system, adding hydrogens, solvent, ions and defining forcefield, topology
    openmmobject, ashfragment = OpenMM_Modeller(pdbfile=pdbfile, forcefield='CHARMM36', watermodel="tip3p", pH=7.0,
        solvent_padding=10.0, ionicstrength=0.1, platform='CPU')

    #MM minimization to get rid the worst contacts
    #OpenMM_Opt(fragment=ashfragment, theory=openmmobject, maxiter=100, tolerance=1000)

#test_openmm_basic()
#test_openmm_modeller()
