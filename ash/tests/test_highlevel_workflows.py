import pytest

from ash.modules.module_highlevel_workflows import basis_for_element


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
