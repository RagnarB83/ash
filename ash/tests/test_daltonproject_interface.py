import importlib.util
import sys
import os
import shutil
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ash.interfaces.interface_daltonproject import DaltonProjectTheory


@pytest.mark.parametrize("program", ["dalton", "lsdalton"])
def test_daltonproject_adapter_energy_and_gradient(monkeypatch, tmp_path, program):
    calls = {}

    class FakeMolecule:
        def __init__(self, **kwargs):
            calls["molecule"] = kwargs

    class FakeBasis:
        def __init__(self, **kwargs):
            calls["basis"] = kwargs

    class FakeQCMethod:
        def __init__(self, **kwargs):
            calls["qc_method"] = kwargs

    class FakeProperty:
        def __init__(self, **kwargs):
            calls["properties"] = kwargs

    class FakeComputeSettings:
        def __init__(self, **kwargs):
            calls["settings"] = kwargs

    def fake_compute(*args, **kwargs):
        calls["compute"] = (args, kwargs)
        return SimpleNamespace(
            energy=-1.234,
            gradients=np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]),
        )

    fake_dp = ModuleType("daltonproject")
    fake_dp.Molecule = FakeMolecule
    fake_dp.Basis = FakeBasis
    fake_dp.QCMethod = FakeQCMethod
    fake_dp.Property = FakeProperty
    fake_dp.ComputeSettings = FakeComputeSettings
    fake_dp.dalton = SimpleNamespace(compute=fake_compute)
    fake_dp.lsdalton = SimpleNamespace(compute=fake_compute)
    monkeypatch.setitem(sys.modules, "daltonproject", fake_dp)

    theory = DaltonProjectTheory(
        basis="STO-3G", qc_method="DFT", functional="B3LYP", program=program,
        numcores=4, memory=1500, scf_threshold=1e-9, work_dir=str(tmp_path),
    )
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    energy, gradient = theory.run(
        current_coords=coords, elems=["O", "H"], charge=0, mult=1, Grad=True
    )

    assert energy == pytest.approx(-1.234)
    np.testing.assert_allclose(
        gradient, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
    )
    np.testing.assert_allclose(theory.gradient, gradient)
    assert calls["molecule"]["charge"] == 0
    assert calls["molecule"]["multiplicity"] == 1
    assert "O 0.000000000000 0.000000000000 0.000000000000" in calls["molecule"]["atoms"]
    assert calls["basis"] == {"basis": "STO-3G"}
    assert calls["qc_method"] == {
        "qc_method": "DFT", "xc_functional": "B3LYP", "scf_threshold": 1e-9
    }
    assert calls["properties"] == {"energy": True, "gradients": True}
    assert calls["settings"]["mpi_num_procs"] == 4
    assert calls["settings"]["memory"] == 1500


def test_daltonproject_adapter_rejects_missing_required_inputs():
    theory = DaltonProjectTheory(basis="STO-3G")
    with pytest.raises(ValueError, match="current_coords"):
        theory.run(elems=["H"], charge=0, mult=2)
    with pytest.raises(ValueError, match="charge and mult"):
        theory.run(current_coords=[[0.0, 0.0, 0.0]], elems=["H"])
    with pytest.raises(NotImplementedError, match="embedding settings"):
        theory.run(
            current_coords=[[0.0, 0.0, 0.0]], elems=["H"],
            charge=0, mult=2, MMcharges=[0.0], current_MM_coords=[[5.0, 0.0, 0.0]],
        )


def test_daltonproject_adapter_rejects_unsupported_program():
    with pytest.raises(ValueError, match="program must"):
        DaltonProjectTheory(basis="STO-3G", program="other")


@pytest.mark.skipif(
    os.environ.get("ASH_DALTONPROJECT_NATIVE") != "1",
    reason="set ASH_DALTONPROJECT_NATIVE=1 to run the external Dalton regression",
)
def test_daltonproject_native_water_gradient_matches_finite_difference(tmp_path):
    assert shutil.which("dalton"), "Dalton Project native regression requires dalton on PATH"
    assert importlib.util.find_spec("daltonproject") is not None

    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    theory = DaltonProjectTheory(
        basis="STO-3G",
        qc_method="HF",
        work_dir=str(tmp_path),
        scratch_dir=str(scratch_dir),
        memory=1500,
        printlevel=0,
    )
    coords = np.array([
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.958400],
        [0.000000, 0.926627, -0.239987],
    ])
    elems = ["O", "H", "H"]
    energy, gradient = theory.run(
        current_coords=coords, elems=elems, Grad=True, charge=0, mult=1,
        label="water_gradient",
    )

    displacement_angstrom = 0.001
    displacement_bohr = displacement_angstrom * 1.8897261254578281
    plus_coords = coords.copy()
    minus_coords = coords.copy()
    plus_coords[1, 2] += displacement_angstrom
    minus_coords[1, 2] -= displacement_angstrom
    energy_plus = theory.run(
        current_coords=plus_coords, elems=elems, Grad=False, charge=0, mult=1,
        label="water_plus",
    )
    energy_minus = theory.run(
        current_coords=minus_coords, elems=elems, Grad=False, charge=0, mult=1,
        label="water_minus",
    )
    numeric_gradient = (energy_plus - energy_minus) / (2 * displacement_bohr)

    assert energy == pytest.approx(-74.96301305, abs=2e-6)
    assert gradient.shape == coords.shape
    assert gradient[1, 2] == pytest.approx(numeric_gradient, abs=2e-6)
