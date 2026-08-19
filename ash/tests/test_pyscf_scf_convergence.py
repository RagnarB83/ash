import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ash.interfaces.interface_pyscf import PySCFTheory


class FakeRHF:
    def __init__(self, *, converged, max_cycle=50):
        self.converged = converged
        self.max_cycle = max_cycle
        self.e_tot = -75.0
        self.mo_occ = np.array([2.0])
        self.make_rdm1_calls = 0
        self.nuc_grad_method_calls = 0
        self.grad_hcore_mm_calls = 0
        self.grad_nuc_mm_calls = 0

    def run(self, dm):
        return self

    def make_rdm1(self):
        self.make_rdm1_calls += 1
        return np.array([[2.0]])

    def nuc_grad_method(self):
        self.nuc_grad_method_calls += 1
        return FakeGradient(self)


class FakeGradient:
    def __init__(self, mf):
        self.mf = mf

    def kernel(self):
        return np.array([[0.0, 0.0, 0.0]])

    def grad_hcore_mm(self, dm):
        self.mf.grad_hcore_mm_calls += 1
        return np.array([[0.0, 0.0, 0.0]])

    def grad_nuc_mm(self):
        self.mf.grad_nuc_mm_calls += 1
        return np.array([[0.0, 0.0, 0.0]])


def install_fake_pyscf(monkeypatch):
    pyscf_module = ModuleType("pyscf")
    pyscf_dft_module = ModuleType("pyscf.dft")
    pyscf_dft_module.rks = SimpleNamespace(RKS=type("FakeRKS", (), {}))
    pyscf_module.dft = pyscf_dft_module
    pyscf_module.scf = SimpleNamespace(hf=SimpleNamespace(RHF=FakeRHF))
    monkeypatch.setitem(sys.modules, "pyscf", pyscf_module)
    monkeypatch.setitem(sys.modules, "pyscf.dft", pyscf_dft_module)


def make_theory(mf):
    theory = PySCFTheory.__new__(PySCFTheory)
    theory.printlevel = 0
    theory.mf = mf
    theory.scf_type = "RHF"
    theory.functional = None
    theory.platform = "CPU"
    theory.periodic = False
    return theory


def test_run_scf_aborts_before_density_for_unconverged_result(monkeypatch):
    install_fake_pyscf(monkeypatch)
    mf = FakeRHF(converged=False)
    theory = make_theory(mf)

    with pytest.raises(SystemExit):
        theory.run_SCF()

    assert mf.make_rdm1_calls == 0


def test_run_scf_allows_explicit_zero_cycle_evaluation(monkeypatch):
    install_fake_pyscf(monkeypatch)
    mf = FakeRHF(converged=False)
    theory = make_theory(mf)

    result = theory.run_SCF(max_cycle=0)

    assert result is mf
    assert mf.max_cycle == 0
    assert mf.make_rdm1_calls == 1
    np.testing.assert_array_equal(theory.dm, np.array([[2.0]]))


def test_run_scf_allows_explicit_negative_cycle_evaluation(monkeypatch):
    install_fake_pyscf(monkeypatch)
    mf = FakeRHF(converged=False, max_cycle=-1)
    theory = make_theory(mf)

    result = theory.run_SCF()

    assert result is mf
    assert mf.make_rdm1_calls == 1


def test_actualrun_aborts_before_gradient_and_pc_gradient_for_unconverged_scf(monkeypatch):
    install_fake_pyscf(monkeypatch)
    mf = FakeRHF(converged=False)
    theory = make_theory(mf)
    theory.SCF = True
    theory.verbose_setting = 0
    theory.write_chkfile_name = None
    theory.BS = False
    theory.setup_guess = lambda: None
    theory.run_stability_analysis = lambda: None
    theory.do_pop_analysis = False
    theory.get_dipole_moment = lambda: None
    theory.dispersion = None
    theory.NMF = False
    theory.losc = False
    theory.postSCF = False
    theory.PC_gradient_code = "new"

    with pytest.raises(SystemExit):
        theory.actualrun(
            Grad=True,
            PC=True,
            current_MM_coords=np.array([[1.0, 0.0, 0.0]]),
            MMcharges=np.array([1.0]),
        )

    assert mf.nuc_grad_method_calls == 0
    assert mf.grad_hcore_mm_calls == 0
    assert mf.grad_nuc_mm_calls == 0


def test_run_scf_preserves_density_for_converged_result(monkeypatch):
    install_fake_pyscf(monkeypatch)
    mf = FakeRHF(converged=True)
    theory = make_theory(mf)

    result = theory.run_SCF()

    assert result is mf
    assert mf.make_rdm1_calls == 1
    np.testing.assert_array_equal(theory.dm, np.array([[2.0]]))
