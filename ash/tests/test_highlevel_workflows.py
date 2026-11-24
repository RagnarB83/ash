import pytest

from ash.modules.module_highlevel_workflows import (
    ORCA_CC_CBS_Theory,
    _build_locally_dense_basis_overrides,
    basis_for_element,
)
from ash.interfaces.interface_ORCA import create_orca_input_plain


@pytest.mark.parametrize(
    "element, basis_family, cardinal, expected",
    [
        ("C", "cc", 2, ("cc-pVDZ", None)),
        ("C", "aug-cc", 4, ("aug-cc-pVQZ", None)),
        ("Mo", "cc", 2, ("cc-pVDZ-PP", "SK-MCDHF-RSC")),
        ("Ru", "def2", 3, ("def2-TZVPP", "def2-ECP")),
        ("Ru", "ma-def2", 2, ("ma-def2-SVP", "def2-ECP")),
        ("Kr", "cc-f12", 3, ("cc-pVTZ-PP-F12", "SK-MCDHF-RSC")),
    ],
)
def test_basis_for_element_selects_basis_and_ecp(element, basis_family, cardinal, expected):
    assert basis_for_element(element, basis_family, cardinal) == expected


def test_basis_for_element_rejects_unknown_basis_family():
    with pytest.raises(SystemExit):
        basis_for_element("C", "unknown-family", 2)


def test_basis_for_element_rejects_def2_cardinal_above_qz():
    with pytest.raises(SystemExit):
        basis_for_element("C", "def2", 5)


def test_locally_dense_basis_override_uses_next_cardinal_only_for_selected_atom():
    overrides = _build_locally_dense_basis_overrides(
        ["H", "O"], atom_indices=(1,), basisfamily="cc", cardinal=2,
        cardinal_offset=1,
    )

    assert overrides == {("O", 1): ['newgto "cc-pVTZ" end\n']}


def test_locally_dense_override_rejects_missing_atom_or_basis():
    with pytest.raises(ValueError, match="only 2 atoms"):
        _build_locally_dense_basis_overrides(
            ["H", "O"], atom_indices=(2,), basisfamily="cc", cardinal=2,
            cardinal_offset=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"locally_dense_atoms": [-1]},
        {"locally_dense_atoms": [0, 0]},
        {"locally_dense_atoms": [True]},
        {"locally_dense_atoms": [0], "locally_dense_cardinal_offset": 0},
    ],
)
def test_locally_dense_constructor_validates_indices_and_offset(kwargs):
    params = {"elements": ["H"], "cardinals": [2], "basisfamily": "cc"}
    params.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        ORCA_CC_CBS_Theory(**params)


def test_locally_dense_basis_is_updated_for_each_cbs_stage():
    class Stage:
        atom_specific_basis_dict = None

    theory = ORCA_CC_CBS_Theory.__new__(ORCA_CC_CBS_Theory)
    theory.locally_dense_atoms = (0,)
    theory.Calc1_basis_dict = {"H": ("cc-pVDZ", None)}
    theory._basis_stages = [(Stage(), 2), (Stage(), 3)]
    theory.basisfamily = "cc"
    theory.locally_dense_cardinal_offset = 1

    theory._set_locally_dense_basis_overrides(["H", "H"])

    assert theory._basis_stages[0][0].atom_specific_basis_dict == {
        ("H", 0): ['newgto "cc-pVTZ" end\n']
    }
    assert theory._basis_stages[1][0].atom_specific_basis_dict == {
        ("H", 0): ['newgto "cc-pVQZ" end\n']
    }
    with pytest.raises(ValueError, match="No locally dense"):
        _build_locally_dense_basis_overrides(
            ["C"], atom_indices=(0,), basisfamily="def2", cardinal=4,
            cardinal_offset=1,
        )


def test_orca_plain_input_applies_sparse_atom_specific_basis(tmp_path):
    input_name = str(tmp_path / "locally_dense")
    create_orca_input_plain(
        name=input_name,
        elems=["H", "O"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        orcasimpleinput="! HF",
        orcablockinput="",
        charge=0,
        mult=1,
        atom_specific_basis_dict={("O", 1): ['newgto "cc-pVTZ" end\n']},
    )

    text = (tmp_path / "locally_dense.inp").read_text()
    assert "H 0.0 0.0 0.0" in text
    assert "O 0.0 0.0 1.0" in text
    assert 'newgto "cc-pVTZ" end' in text
