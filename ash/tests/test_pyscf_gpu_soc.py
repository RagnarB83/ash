import sys
from types import ModuleType

import pytest

from ash.interfaces.interface_pyscf import _validate_gpu_x2c_soc, _validate_soc_cas


def test_cpu_x2c_soc_does_not_require_gpu4pyscf():
    _validate_gpu_x2c_soc("CPU", "GHF", True)


def test_gpu_x2c_soc_accepts_supported_ghf(monkeypatch):
    gpu4pyscf = ModuleType("gpu4pyscf")
    gpu4pyscf.__version__ = "1.8.0"
    monkeypatch.setitem(sys.modules, "gpu4pyscf", gpu4pyscf)

    _validate_gpu_x2c_soc("GPU", "GHF", True)


def test_gpu_x2c_soc_rejects_gks_until_gpu_conversion_is_supported():
    with pytest.raises(NotImplementedError, match="scf_type='GHF'"):
        _validate_gpu_x2c_soc("GPU", "GKS", True)


def test_gpu_x2c_soc_allows_cpu_post_scf_after_gpu_ghf(monkeypatch):
    gpu4pyscf = ModuleType("gpu4pyscf")
    gpu4pyscf.__version__ = "1.9.0"
    monkeypatch.setitem(sys.modules, "gpu4pyscf", gpu4pyscf)

    _validate_gpu_x2c_soc("GPU", "GHF", True)


def test_soc_cas_rejected_for_generalized_x2c_reference():
    with pytest.raises(NotImplementedError, match="generalized CAS/CASSCF"):
        _validate_soc_cas("GHF", True, True)


def test_soc_cas_guard_only_applies_to_generalized_x2c():
    _validate_soc_cas("GHF", False, True)
    _validate_soc_cas("RHF", True, True)


def test_gpu_x2c_soc_requires_gpu4pyscf_18(monkeypatch):
    gpu4pyscf = ModuleType("gpu4pyscf")
    gpu4pyscf.__version__ = "1.7.9"
    monkeypatch.setitem(sys.modules, "gpu4pyscf", gpu4pyscf)

    with pytest.raises(RuntimeError, match="GPU4PySCF >= 1.8.0"):
        _validate_gpu_x2c_soc("GPU", "GHF", True)
