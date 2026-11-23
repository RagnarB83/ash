"""OpenMM_MD must not leave skip_openmmtheory_run set on the user's WrapTheory object.

OpenMM_MD sets skip_openmmtheory_run=True on the WrapTheory it drives, because the MM
gradient comes from the OpenMM simulation context during MD. The flag was never put back,
so the same WrapTheory object used afterwards for a singlepoint or an optimization
silently contributed zero energy and zero gradient for its OpenMMTheory component.

A stub QM theory is used: the behaviour under test is in OpenMM_MD, not in any QM program.
"""
import os

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from ash import (Fragment, OpenMMTheory, OpenMM_MD, Singlepoint, WrapTheory,
                 ashpath)

XYZFILE = os.path.join(ashpath, "tests", "xyzfiles", "h2o_MeOH.xyz")
XMLFILE = os.path.join(ashpath, "tests", "extra_files", "MeOH_H2O-sigma.xml")
STUB_ENERGY = -1.0


class StubQM:
    """Minimal QM theory: fixed energy, zero gradient."""
    theorytype = "QM"
    numcores = 1
    theorynamelabel = "StubQM"
    label = "StubQM"
    filename = "stub"

    def run(self, current_coords=None, elems=None, qm_elems=None, Grad=False, PC=False,
            numcores=None, charge=None, mult=None, label=None, restart=False,
            current_MM_coords=None, MMcharges=None, mm_elems=None):
        if Grad:
            return STUB_ENERGY, np.zeros((len(current_coords), 3))
        return STUB_ENERGY

    def cleanup(self):
        pass


@pytest.fixture
def fragment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frag = Fragment(xyzfile=XYZFILE, charge=0, mult=1, printlevel=0)
    frag.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    return frag


def build_wraptheory():
    mm = OpenMMTheory(xmlfiles=[XMLFILE], pdbfile="h2o_MeOH.pdb", autoconstraints=None,
                      rigidwater=False, printlevel=0)
    return WrapTheory(theories=[StubQM(), mm], printlevel=0)


def run_short_md(fragment, wraptheory):
    OpenMM_MD(fragment=fragment, theory=wraptheory, timestep=0.0005, simulation_time=0.001,
              traj_frequency=1000, temperature=300, integrator='LangevinMiddleIntegrator',
              trajectory_file_option='DCD', printlevel=0)


@pytest.mark.skipif(not os.path.isfile(XYZFILE) or not os.path.isfile(XMLFILE),
                    reason="test input files not present")
def test_flag_is_restored_after_md(fragment):
    wrap = build_wraptheory()
    assert wrap.skip_openmmtheory_run is False
    run_short_md(fragment, wrap)
    assert wrap.skip_openmmtheory_run is False, \
        "OpenMM_MD left skip_openmmtheory_run set on the WrapTheory object"


@pytest.mark.skipif(not os.path.isfile(XYZFILE) or not os.path.isfile(XMLFILE),
                    reason="test input files not present")
def test_mm_contribution_survives_an_md_run(fragment):
    """A singlepoint after MD must still include the OpenMM component."""
    wrap = build_wraptheory()
    run_short_md(fragment, wrap)
    reused = Singlepoint(theory=wrap, fragment=fragment, charge=0, mult=1).energy
    # Same coordinates, but a WrapTheory that has not been through MD
    fresh_energy = Singlepoint(theory=build_wraptheory(), fragment=fragment,
                               charge=0, mult=1).energy
    assert reused == pytest.approx(fresh_energy), \
        f"MM contribution lost after MD: {reused} vs {fresh_energy}"
    assert reused != pytest.approx(STUB_ENERGY), \
        "energy equals the stub QM energy alone, so the MM part contributed nothing"


@pytest.mark.skipif(not os.path.isfile(XYZFILE) or not os.path.isfile(XMLFILE),
                    reason="test input files not present")
def test_user_set_flag_is_not_overwritten(fragment):
    """A user who deliberately set the flag keeps it after the MD run."""
    wrap = build_wraptheory()
    wrap.skip_openmmtheory_run = True
    run_short_md(fragment, wrap)
    assert wrap.skip_openmmtheory_run is True
