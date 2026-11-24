import numpy as np

from ash.modules.module_highlevel_workflows import (
    Extrapolation_twopoint_SCF,
    Extrapolation_twopoint_corr,
)
from ash.interfaces.interface_ORCA import grab_HF_and_corr_energies


def test_scf_extrapolation_accepts_gradient_arrays():
    gradients = [
        np.array([[1.0, -2.0, 3.0]]),
        np.array([[4.0, 5.0, -6.0]]),
    ]
    alpha = 2.1
    x = np.exp(-alpha * np.sqrt(2))
    y = np.exp(-alpha * np.sqrt(3))
    expected = (gradients[0] * y - gradients[1] * x) / (y - x)

    result = Extrapolation_twopoint_SCF(
        gradients, [2, 3], "cc", alpha=alpha, SCFextrapolation=True
    )

    np.testing.assert_allclose(result, expected)


def test_scf_gradient_uses_largest_basis_when_extrapolation_disabled():
    gradients = [np.zeros((2, 3)), np.ones((2, 3))]

    result = Extrapolation_twopoint_SCF(
        gradients, [2, 3], "cc", SCFextrapolation=False
    )

    np.testing.assert_array_equal(result, gradients[1])


def test_correlation_extrapolation_accepts_gradient_arrays():
    gradients = [
        np.array([[1.0, -2.0, 3.0]]),
        np.array([[4.0, 5.0, -6.0]]),
    ]
    beta = 2.5
    x = 2**beta
    y = 3**beta
    expected = (x * gradients[0] - y * gradients[1]) / (x - y)

    result = Extrapolation_twopoint_corr(gradients, [2, 3], "cc", beta=beta)

    np.testing.assert_allclose(result, expected)


def test_orca_correlation_energy_parser_accepts_optional_eh_unit(tmp_path):
    output = tmp_path / "orca.out"
    output.write_text(
        "E(0)                                       ...     -1.100000000 Eh\n"
        "E(0)                                       ...     -1.200000000\n"
        "E(CORR)                                    ...     -0.030000000\n"
        "(T) correction                             ...     -0.001000000\n"
        "T1 diagnostic                              ...      0.005000000\n"
    )

    result = grab_HF_and_corr_energies(str(output))

    assert result["HF"] == -1.2
    assert result["CCSD_corr"] == -0.03
    assert result["CCSD(T)_corr"] == -0.001
    assert result["full_corr"] == -0.031
    assert result["T1diag"] == 0.005
