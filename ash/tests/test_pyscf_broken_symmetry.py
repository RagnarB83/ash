"""
Broken-symmetry SCF (PySCFTheory BS=True) on Li2 stretched to 6 A (UHF, HS triplet -> flip Li1 -> Ms=0).
Li2 rather than H2 so that both spin channels stay occupied in the triplet (gpu4pyscf needs that).

The second test reproduces what happens on GPU, where mf.to_gpu() deep-copies the Mole so that
self.mf.mol is not self.mol: the spin must be changed on the Mole used by the SCF, and the flipped
density must reach the SCF as one (2,nao,nao) array (gpu4pyscf's cupy.asarray cannot take a list
of cupy arrays).
"""
import numpy as np
import pytest

pytest.importorskip("pyscf")

from ash.interfaces.interface_pyscf import PySCFTheory

ELEMS = ["Li", "Li"]
COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])


def make_theory():
    return PySCFTheory(scf_type="UHF", basis="def2-SVP", printlevel=0, conv_tol=1e-10,
                       scf_maxiter=200, do_pop_analysis=False, BS=True, HSmult=3, atomstoflip=[1])


def run_and_record(theory):
    calls = []
    run_SCF = theory.run_SCF
    def recording_run_SCF(mf=None, dm=None, max_cycle=None):
        calls.append((theory.mf.mol.spin, dm))
        return run_SCF(mf=mf, dm=dm, max_cycle=max_cycle)
    theory.run_SCF = recording_run_SCF
    energy = theory.run(current_coords=COORDS, qm_elems=ELEMS, charge=0, mult=1)
    return energy, calls


def check(theory, calls):
    # HS step at 2S = 2, BS step at 2S = 0, both on the Mole the SCF actually uses
    assert [spin for spin, _ in calls] == [2, 0]
    dm = calls[1][1]
    assert not isinstance(dm, (list, tuple))
    assert dm.shape[0] == 2
    # two antiparallel Li 2s electrons: <S**2> ~ 1
    s2 = theory.mf.spin_square()[0]
    assert abs(s2 - 1.0) < 5e-2


def test_broken_symmetry_cpu():
    theory = make_theory()
    energy, calls = run_and_record(theory)
    check(theory, calls)


def test_broken_symmetry_when_mf_has_its_own_mole(monkeypatch):
    # Emulate mf.to_gpu(): the SCF object ends up with a deep copy of self.mol
    prepare_run = PySCFTheory.prepare_run
    def prepare_run_copy_mol(self, *args, **kwargs):
        prepare_run(self, *args, **kwargs)
        self.mf.mol = self.mf.mol.copy()
    monkeypatch.setattr(PySCFTheory, "prepare_run", prepare_run_copy_mol)
    theory = make_theory()
    energy, calls = run_and_record(theory)
    assert theory.mf.mol is not theory.mol
    check(theory, calls)
