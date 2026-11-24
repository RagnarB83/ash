import sys
from types import ModuleType

import pytest

from ash.interfaces.interface_BSE import get_basis_from_bse, write_basis_from_bse


def test_get_basis_from_bse_requests_selected_elements_and_format(monkeypatch):
    calls = []
    fake_bse = ModuleType("basis_set_exchange")
    fake_bse.get_formats = lambda: {"nwchem": "NWChem", "orca": "ORCA"}

    def fake_get_basis(name, **kwargs):
        calls.append((name, kwargs))
        return "basis text"

    fake_bse.get_basis = fake_get_basis
    monkeypatch.setitem(sys.modules, "basis_set_exchange", fake_bse)

    result = get_basis_from_bse("def2-SVP", elements=["H", "O"], fmt="orca")

    assert result == "basis text"
    assert calls == [("def2-SVP", {"elements": ["H", "O"], "fmt": "orca", "header": True})]


def test_get_basis_from_bse_rejects_unknown_format(monkeypatch):
    fake_bse = ModuleType("basis_set_exchange")
    fake_bse.get_formats = lambda: {"nwchem": "NWChem"}
    monkeypatch.setitem(sys.modules, "basis_set_exchange", fake_bse)

    with pytest.raises(ValueError, match="Unknown BSE format"):
        get_basis_from_bse("def2-SVP", fmt="not-a-format")


def test_write_basis_from_bse_writes_text_to_requested_path(monkeypatch, tmp_path):
    fake_bse = ModuleType("basis_set_exchange")
    fake_bse.get_formats = lambda: {"nwchem": "NWChem"}
    fake_bse.get_basis = lambda *args, **kwargs: "basis text\n"
    monkeypatch.setitem(sys.modules, "basis_set_exchange", fake_bse)
    destination = tmp_path / "custom.basis"

    result = write_basis_from_bse("def2-SVP", destination, elements=["H"])

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "basis text\n"
