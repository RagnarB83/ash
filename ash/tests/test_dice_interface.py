from types import SimpleNamespace

import numpy as np
import pytest

from ash.interfaces.interface_dice import DiceTheory


def test_dice_nevpt2_result_is_read_from_native_qmcutils():
    calls = {}

    def get_nevpt_energy(fname, printNevpt2):
        calls["args"] = (fname, printNevpt2)
        return -75.123456, 0.000031

    theory = DiceTheory.__new__(DiceTheory)
    theory.QMCUtils = SimpleNamespace(get_nevptEnergy=get_nevpt_energy)
    theory.grab_nevpt2_energy()

    assert calls["args"] == ("nevpt2.out", True)
    assert theory.energy == pytest.approx(-75.123456)
    assert theory.error == pytest.approx(0.000031)


def test_dice_nevpt2_marks_single_rank_error_estimate_unavailable(capsys):
    theory = DiceTheory.__new__(DiceTheory)
    theory.QMCUtils = SimpleNamespace(
        get_nevptEnergy=lambda fname, printNevpt2: (-75.1, float("nan"))
    )

    theory.grab_nevpt2_energy()

    assert theory.energy == pytest.approx(-75.1)
    assert theory.error is None
    assert "at least two MPI processes" in capsys.readouterr().out


def test_dice_nevpt2_rejects_nonfinite_energy():
    theory = DiceTheory.__new__(DiceTheory)
    theory.QMCUtils = SimpleNamespace(
        get_nevptEnergy=lambda fname, printNevpt2: (float("nan"), 0.001)
    )

    with pytest.raises(ValueError, match="non-finite NEVPT2 energy"):
        theory.grab_nevpt2_energy()


def test_dice_nevpt2_enables_required_shci_two_particle_rdm(monkeypatch, tmp_path):
    monkeypatch.setattr(DiceTheory, "load_qmcutils", lambda self: None)

    theory = DiceTheory(
        dicedir=str(tmp_path),
        Dice_SHCI_direct=True,
        NEVPT2=True,
        NEVPT2_numdets=65,
        SHCI_DoRDM=False,
    )

    assert theory.SHCI_DoRDM is True
    assert theory.NEVPT2_numdets == 65


def test_dice_shci_sets_the_requested_mo_coefficients():
    theory = DiceTheory.__new__(DiceTheory)
    theory.mch = SimpleNamespace(mo_coeff=None)
    mo_coefficients = np.eye(3)

    theory.SHCI_object_set_mos(mo_coeffs=mo_coefficients)

    np.testing.assert_array_equal(theory.mch.mo_coeff, mo_coefficients)


def test_dice_nevpt2_requires_external_virtual_orbitals():
    theory = DiceTheory.__new__(DiceTheory)
    theory.mch = SimpleNamespace(mo_coeff=np.eye(7), ncore=1, ncas=6)

    with pytest.raises(ValueError, match="at least one external virtual orbital"):
        theory.validate_nevpt2_space()


def test_dice_nevpt2_determinant_adapter_writes_vmc_format(tmp_path, monkeypatch):
    output = tmp_path / "output.dat"
    output.write_text(
        "Printing most important determinants\n"
        " Det     weight  Determinant string\n"
        "State :   0\n"
        "   0       0.9893832826  2 2 2 2 0 0\n"
        "   1      -0.0678497382  2 0 2 2 0 2\n\n"
        "Next section\n"
    )
    theory = DiceTheory.__new__(DiceTheory)
    monkeypatch.chdir(tmp_path)

    theory.write_nevpt2_dets(str(output), nroot=0)

    assert (tmp_path / "dets").read_text().splitlines() == [
        "0.9893832826 2 2 2 2 0 0",
        "-0.0678497382 2 0 2 2 0 2",
    ]


def test_dice_variational_energy_parser_reads_dice_output(tmp_path):
    output = tmp_path / "output.dat"
    output.write_text(
        "Variational calculation result\n"
        "Root             Energy     Time(s)\n"
        "   0       -1.1372838345        0.00\n"
        "   1       -1.1000000000        0.01\n"
        "\nPrinting most important determinants\n"
    )
    theory = DiceTheory.__new__(DiceTheory)

    assert theory.grab_dice_variational_energy(str(output)) == pytest.approx(-1.1372838345)


def test_free_projection_energy_parser_uses_last_sampling_row(tmp_path):
    output = tmp_path / "dqmc.out"
    output.write_text(
        "header\n"
        " 0.05 ( -1.11886020, -0.00000968) (3.58e-04 (1.68e-04))\n"
        " 0.10 ( -1.12046300, 0.00003049) (6.32e-04 (2.97e-04))\n"
    )
    theory = DiceTheory.__new__(DiceTheory)

    theory.grab_fpafqmc_energy(filename=str(output))

    assert theory.energy == pytest.approx(-1.120463)
    assert theory.error == pytest.approx(6.32e-4)
    assert theory.imaginary_energy == pytest.approx(3.049e-5)


def test_free_projection_energy_parser_rejects_missing_energy(tmp_path):
    output = tmp_path / "dqmc.out"
    output.write_text("Dice failed before sampling\n")
    theory = DiceTheory.__new__(DiceTheory)

    with pytest.raises(ValueError, match="No free-projection AFQMC energy rows"):
        theory.grab_fpafqmc_energy(filename=str(output))


def test_free_projection_run_does_not_reuse_stale_output(tmp_path, monkeypatch):
    output = tmp_path / "dqmc.out"
    output.write_text("0.05 ( -1.0, 0.0) (0.001)\n")
    theory = DiceTheory.__new__(DiceTheory)
    theory.fp_helper = SimpleNamespace(run_fpafqmc=lambda *args, **kwargs: None)
    theory.frozen_core_orbs = 0
    theory.dicedir = str(tmp_path)
    theory.numcores = 1
    theory.FP_AFQMC_dt = 0.05
    theory.FP_AFQMC_nsteps = 1
    theory.FP_AFQMC_eneSteps = (1,)
    theory.FP_AFQMC_errorTargets = (0.001,)
    theory.FP_AFQMC_stochasticIter = 1
    theory.FP_AFQMC_choleskyThreshold = 0.00005
    theory.FP_AFQMC_orthoSteps = 1
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="did not produce dqmc.out"):
        theory.run_fp_afqmc(trial_object=None, energy_guess=-1.0)


def test_free_projection_option_requires_afqmc(tmp_path):
    with pytest.raises(ValueError, match="requires AFQMC=True"):
        DiceTheory(
            dicedir=str(tmp_path),
            Dice_SHCI_direct=True,
            AFQMC_free_projection=True,
        )


def test_free_projection_sampling_arrays_must_match(tmp_path):
    with pytest.raises(ValueError, match="same nonzero length"):
        DiceTheory(
            dicedir=str(tmp_path),
            Dice_SHCI_direct=True,
            AFQMC=True,
            AFQMC_free_projection=True,
            FP_AFQMC_eneSteps=(1, 2),
            FP_AFQMC_errorTargets=(0.001,),
        )


def test_dice_initial_orbital_count_handles_two_orbital_rhf(capsys):
    occupations = np.array([2.0, 0.0])
    coefficients = np.eye(2)
    theory = DiceTheory.__new__(DiceTheory)
    theory.pyscftheoryobject = SimpleNamespace(
        mf=SimpleNamespace(mo_occ=occupations, mo_coeff=coefficients)
    )
    theory.moreadfile = None
    theory.initial_orbitals = "RHF"
    theory.numcores = 1

    returned_coefficients, returned_occupations = theory.setup_initial_orbitals(
        elems=["H", "H"]
    )

    assert "There are 2 orbitals in the system" in capsys.readouterr().out
    np.testing.assert_array_equal(returned_coefficients, coefficients)
    np.testing.assert_array_equal(returned_occupations, occupations)
