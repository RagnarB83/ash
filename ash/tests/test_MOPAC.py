"""Integration checks for the external MOPAC executable."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from ash import Fragment, MOPACTheory, Singlepoint


def _find_mopacdir():
    configured_dir = os.environ.get("MOPACDIR")
    if configured_dir and os.path.isfile(os.path.join(configured_dir, "bin", "mopac")):
        return configured_dir
    executable = shutil.which("mopac")
    if executable:
        return os.path.dirname(os.path.dirname(os.path.abspath(executable)))
    pytest.skip("MOPAC executable not available (set MOPACDIR or add mopac to PATH)")


def _water():
    return Fragment(
        elems=["H", "O", "H"],
        coords=[[-0.02110, -0.00200, 0.0], [0.83450, 0.45190, 0.0], [1.47690, -0.27300, 0.0]],
        charge=0,
        mult=1,
        printlevel=0,
    )


@pytest.mark.parametrize(
    "keywords,energy,charges,dipole",
    [
        (None, -0.0919353475, [0.32046258, -0.64093238, 0.32046980], [-0.38486150, -2.12796373, 0.0]),
        ("EPS=78.4", -0.1039321748, [0.37393000, -0.74771029, 0.37378029], [-0.43806200, -2.41901946, 5.37e-6]),
    ],
)
def test_mopac_pm7_energy_gradient_and_properties(tmp_path, monkeypatch, keywords, energy, charges, dipole):
    mopacdir = _find_mopacdir()
    monkeypatch.chdir(tmp_path)
    theory = MOPACTheory(
        mopacdir=mopacdir,
        method="PM7",
        filename="mopac_test",
        precise=True,
        scfcrt=1e-10,
        keywords=keywords,
        printlevel=0,
    )

    result = Singlepoint(theory=theory, fragment=_water(), Grad=True)

    assert result.energy == pytest.approx(energy, abs=2e-8)
    assert result.gradient.shape == (3, 3)
    assert np.all(np.isfinite(result.gradient))
    assert np.allclose(theory.charges, charges, atol=2e-6)
    assert np.allclose(theory.properties["dipole"], dipole, atol=2e-5)


def test_mopac_qmmm_gradients_match_finite_differences(tmp_path, monkeypatch):
    mopacdir = _find_mopacdir()
    monkeypatch.chdir(tmp_path)
    elems = ["O", "H", "H"]
    qm_coords = np.array([[0.0, 0.0, 0.0], [0.0, -0.757, 0.587], [0.0, 0.757, 0.587]])
    pc_coords = np.array([[2.5, 0.0, 1.5], [-2.5, 0.0, -1.5]])
    pc_charges = np.array([0.3, -0.2])
    theory = MOPACTheory(
        mopacdir=mopacdir,
        method="PM7",
        filename="mopac_qmmm_test",
        precise=True,
        scfcrt=1e-10,
        printlevel=0,
    )
    sentinel_mol_in = "keep this pre-existing file"
    Path("mol.in").write_text(sentinel_mol_in)

    energy, qm_gradient, pc_gradient = theory.run(
        current_coords=qm_coords,
        current_MM_coords=pc_coords,
        MMcharges=pc_charges,
        qm_elems=elems,
        charge=0,
        mult=1,
        Grad=True,
        PC=True,
    )
    assert np.isfinite(energy)

    displacement = 0.001  # Angstrom
    bohr_to_angstrom = 0.529177210903

    def energy_at(qm, pc):
        return theory.run(
            current_coords=qm,
            current_MM_coords=pc,
            MMcharges=pc_charges,
            qm_elems=elems,
            charge=0,
            mult=1,
            PC=True,
        )

    qm_fd = np.zeros_like(qm_coords)
    for atom in range(len(qm_coords)):
        for axis in range(3):
            plus, minus = qm_coords.copy(), qm_coords.copy()
            plus[atom, axis] += displacement
            minus[atom, axis] -= displacement
            qm_fd[atom, axis] = (energy_at(plus, pc_coords) - energy_at(minus, pc_coords)) / (2 * displacement) * bohr_to_angstrom

    pc_fd = np.zeros_like(pc_coords)
    for atom in range(len(pc_coords)):
        for axis in range(3):
            plus, minus = pc_coords.copy(), pc_coords.copy()
            plus[atom, axis] += displacement
            minus[atom, axis] -= displacement
            pc_fd[atom, axis] = (energy_at(qm_coords, plus) - energy_at(qm_coords, minus)) / (2 * displacement) * bohr_to_angstrom

    assert np.max(np.abs(qm_gradient - qm_fd)) < 2e-7
    assert np.max(np.abs(pc_gradient - pc_fd)) < 2e-8
    assert Path("mol.in").read_text() == sentinel_mol_in


def test_mopac_generic_optimization_and_numerical_frequencies(tmp_path, monkeypatch):
    pytest.importorskip("geometric")
    from ash.interfaces.interface_geometric_new import geomeTRICOptimizer
    from ash.modules.module_freq import NumFreq

    mopacdir = _find_mopacdir()
    monkeypatch.chdir(tmp_path)
    settings = dict(mopacdir=mopacdir, method="PM7", precise=True, scfcrt=1e-10, printlevel=0)
    water = _water()
    theory = MOPACTheory(filename="mopac_opt_test", **settings)
    opt_result = geomeTRICOptimizer(
        theory=theory,
        fragment=water,
        coordsystem="cart",
        force_coordsystem=True,
        maxiter=100,
        printlevel=0,
        result_write_to_disk=False,
    )
    gradient_result = Singlepoint(theory=theory, fragment=water, Grad=True)
    assert np.isfinite(opt_result.energy)
    assert np.max(np.abs(gradient_result.gradient)) < 3e-5

    relaxed_water = Fragment(
        elems=["H", "O", "H"],
        coords=[
            [-0.016619016, 0.009603882, 0.0],
            [0.833396536, 0.445864077, 0.0],
            [1.476815457, -0.260555447, 0.0],
        ],
        charge=0,
        mult=1,
        printlevel=0,
    )
    frequency_result = NumFreq(
        fragment=relaxed_water,
        theory=MOPACTheory(filename="mopac_numfreq_test", **settings),
        npoint=2,
        displacement=0.005,
        printlevel=0,
        IR=False,
    )
    assert np.allclose(frequency_result.frequencies[-3:], [1397.10, 2810.32, 2859.11], atol=0.1)
