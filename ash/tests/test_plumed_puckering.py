"""Test plumed_ASH PUCKERING CV support for 6-membered ring Cremer-Pople metadynamics.

Requirements: plumed (pip install plumed), libplumedKernel.so
"""
from ash import *
import numpy as np
import os, glob, tempfile, shutil

# Locate PLUMED kernel
_kernel_candidates = glob.glob(os.path.expanduser("~/software/plumed2/*/lib/libplumedKernel.so"))
_plumed_kernel = _kernel_candidates[0] if _kernel_candidates else None


def _make_ring_fragment():
    """Flat 6-membered ring mimicking a pyranose in 4C1-like geometry."""
    # Approximate chair coordinates (O5, C1, C2, C3, C4, C5)
    coords = np.array([
        [ 1.200,  0.700,  0.300],  # O5
        [ 1.500,  0.000,  0.000],  # C1
        [ 0.750, -1.300,  0.000],  # C2
        [-0.750, -1.300,  0.000],  # C3
        [-1.500,  0.000,  0.300],  # C4
        [-0.750,  1.300,  0.300],  # C5
    ])
    return Fragment(elems=["O","C","C","C","C","C"], coords=coords)


def _skip_if_no_plumed():
    """Skip test if plumed is not available."""
    if _plumed_kernel is None:
        import pytest
        pytest.skip("libplumedKernel.so not found")
    try:
        import plumed
    except ImportError:
        import pytest
        pytest.skip("plumed python module not installed")


def test_plumed_puckering_1d():
    """1D metadynamics setup with PUCKERING theta as CV1."""
    _skip_if_no_plumed()
    from ash.interfaces.interface_plumed import plumed_ASH

    frag = _make_ring_fragment()
    workdir = tempfile.mkdtemp(prefix="test_puck_1d_")
    origdir = os.getcwd()
    os.chdir(workdir)

    try:
        p = plumed_ASH(
            path_to_plumed_kernel=_plumed_kernel,
            dynamics_program="ASE",
            fragment=frag,
            CV1_type="puckering",
            CV1_indices=[0, 1, 2, 3, 4, 5],
            CV1_component="theta",
            sigma1=0.05,
            height=0.001,
            biasfactor=10,
            timestep=0.001,
            pace_num=500,
        )

        # Check info file (written at init, no flush needed)
        assert os.path.exists("plumed_ash.info"), "plumed_ash.info not written"
        info = open("plumed_ash.info").read()
        assert "CV1_type puckering" in info
        assert "CV1_component theta" in info

        # Flush plumed.log before reading
        p.close()

        log = open("plumed.log").read()
        assert "Action PUCKERING" in log, "PUCKERING action not in plumed.log"
        assert "cv1.theta" in log, "cv1.theta not in plumed.log"
        assert "Action METAD" in log, "METAD action not in plumed.log"
    finally:
        os.chdir(origdir)
        shutil.rmtree(workdir)


def test_plumed_puckering_2d():
    """2D metadynamics setup with PUCKERING theta + TORSION."""
    _skip_if_no_plumed()
    from ash.interfaces.interface_plumed import plumed_ASH

    frag = _make_ring_fragment()
    workdir = tempfile.mkdtemp(prefix="test_puck_2d_")
    origdir = os.getcwd()
    os.chdir(workdir)

    try:
        p = plumed_ASH(
            path_to_plumed_kernel=_plumed_kernel,
            dynamics_program="ASE",
            fragment=frag,
            CV1_type="puckering",
            CV1_indices=[0, 1, 2, 3, 4, 5],
            CV1_component="theta",
            CV2_type="torsion",
            CV2_indices=[2, 3, 4, 5],  # C2-C3-C4-C5
            sigma1=0.05,
            sigma2=0.1,
            height=0.001,
            biasfactor=10,
            timestep=0.001,
            pace_num=500,
        )

        info = open("plumed_ash.info").read()
        assert "CV1_component theta" in info
        assert "CV2_type torsion" in info
        assert "CV2_component" not in info

        p.close()

        log = open("plumed.log").read()
        assert "cv1.theta" in log
        assert "Action TORSION" in log
    finally:
        os.chdir(origdir)
        shutil.rmtree(workdir)


def test_plumed_puckering_phi():
    """PUCKERING with phi component instead of theta."""
    _skip_if_no_plumed()
    from ash.interfaces.interface_plumed import plumed_ASH

    frag = _make_ring_fragment()
    workdir = tempfile.mkdtemp(prefix="test_puck_phi_")
    origdir = os.getcwd()
    os.chdir(workdir)

    try:
        p = plumed_ASH(
            path_to_plumed_kernel=_plumed_kernel,
            dynamics_program="ASE",
            fragment=frag,
            CV1_type="puckering",
            CV1_indices=[0, 1, 2, 3, 4, 5],
            CV1_component="phi",
            sigma1=0.1,
            height=0.001,
            biasfactor=10,
            timestep=0.001,
            pace_num=500,
        )

        assert "CV1_component phi" in open("plumed_ash.info").read()

        p.close()

        log = open("plumed.log").read()
        assert "cv1.phi" in log
    finally:
        os.chdir(origdir)
        shutil.rmtree(workdir)


def test_plumed_puckering_missing_component():
    """PUCKERING without component should exit with error."""
    _skip_if_no_plumed()
    from ash.interfaces.interface_plumed import plumed_ASH

    frag = _make_ring_fragment()
    workdir = tempfile.mkdtemp(prefix="test_puck_nocomp_")
    origdir = os.getcwd()
    os.chdir(workdir)

    try:
        raised = False
        try:
            p = plumed_ASH(
                path_to_plumed_kernel=_plumed_kernel,
                dynamics_program="ASE",
                fragment=frag,
                CV1_type="puckering",
                CV1_indices=[0, 1, 2, 3, 4, 5],
                # CV1_component intentionally omitted
                sigma1=0.05,
                height=0.001,
                biasfactor=10,
                timestep=0.001,
                pace_num=500,
            )
        except SystemExit:
            raised = True
        assert raised, "plumed_ASH should exit when PUCKERING CV has no component"
    finally:
        os.chdir(origdir)
        shutil.rmtree(workdir)


def test_plumed_torsion_backward_compat():
    """Existing TORSION CV still works (no component needed)."""
    _skip_if_no_plumed()
    from ash.interfaces.interface_plumed import plumed_ASH

    frag = _make_ring_fragment()
    workdir = tempfile.mkdtemp(prefix="test_puck_compat_")
    origdir = os.getcwd()
    os.chdir(workdir)

    try:
        p = plumed_ASH(
            path_to_plumed_kernel=_plumed_kernel,
            dynamics_program="ASE",
            fragment=frag,
            CV1_type="torsion",
            CV1_indices=[0, 1, 2, 3],
            sigma1=0.1,
            height=0.001,
            biasfactor=10,
            timestep=0.001,
            pace_num=500,
        )

        info = open("plumed_ash.info").read()
        assert "CV1_component" not in info

        p.close()

        log = open("plumed.log").read()
        assert "Action TORSION" in log
        assert "cv1.None" not in log
    finally:
        os.chdir(origdir)
        shutil.rmtree(workdir)
