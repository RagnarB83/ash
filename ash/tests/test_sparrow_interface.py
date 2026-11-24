import numpy as np
import pytest

pytest.importorskip("scine_utilities")
pytest.importorskip("scine_sparrow")

from ash import Fragment, Singlepoint, SparrowTheory
from ash.constants import ang2bohr


def h2_fragment():
    return Fragment(
        coordsstring="H 0.000000 0.000000 0.000000\nH 0.000000 0.000000 0.740000",
        charge=0,
        mult=1,
    )


def test_sparrow_pm6_energy_gradient_and_finite_difference():
    fragment = h2_fragment()
    theory = SparrowTheory(method="PM6", printlevel=0)

    result = Singlepoint(
        fragment=fragment, theory=theory, Grad=True, printlevel=0,
        result_write_to_disk=False,
    )

    assert np.isfinite(result.energy)
    assert result.gradient.shape == (2, 3)
    assert np.all(np.isfinite(result.gradient))

    step_bohr = 1.0e-4
    displaced = np.array(fragment.coords, dtype=float)
    displaced[1, 2] += step_bohr / ang2bohr
    energy_plus = theory.run(
        current_coords=displaced, qm_elems=fragment.elems,
        charge=fragment.charge, mult=fragment.mult,
    )
    displaced[1, 2] -= 2 * step_bohr / ang2bohr
    energy_minus = theory.run(
        current_coords=displaced, qm_elems=fragment.elems,
        charge=fragment.charge, mult=fragment.mult,
    )
    numeric_gradient = (energy_plus - energy_minus) / (2 * step_bohr)

    assert result.gradient[1, 2] == pytest.approx(numeric_gradient, abs=1.0e-6)


def test_sparrow_respects_charge_and_multiplicity():
    fragment = h2_fragment()
    theory = SparrowTheory(method="PM6", printlevel=0)
    neutral_energy = theory.run(
        current_coords=fragment.coords, qm_elems=fragment.elems, charge=0, mult=1
    )
    cation_energy = theory.run(
        current_coords=fragment.coords, qm_elems=fragment.elems, charge=1, mult=2
    )

    assert np.isfinite(neutral_energy)
    assert np.isfinite(cation_energy)
    assert cation_energy != pytest.approx(neutral_energy, abs=1.0e-5)


def test_sparrow_provides_native_hessian():
    fragment = h2_fragment()
    theory = SparrowTheory(method="PM6", printlevel=0)

    hessian = np.asarray(theory.Hessian(fragment=fragment))

    assert theory.analytic_hessian is True
    assert hessian.shape == (6, 6)
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-10)

    step_bohr = 1.0e-4
    displaced = np.array(fragment.coords, dtype=float)
    gradients = []
    for sign in (-1, 1):
        displaced[1, 2] = fragment.coords[1, 2] + sign * step_bohr / ang2bohr
        _, gradient = theory.run(
            current_coords=displaced, qm_elems=fragment.elems, Grad=True,
            charge=fragment.charge, mult=fragment.mult,
        )
        gradients.append(gradient[1, 2])
    numeric_hessian_element = (gradients[1] - gradients[0]) / (2 * step_bohr)

    assert hessian[5, 5] == pytest.approx(numeric_hessian_element, abs=5.0e-4)


def test_sparrow_rejects_unsupported_point_charge_embedding():
    fragment = h2_fragment()
    theory = SparrowTheory(method="PM6", printlevel=0)

    with pytest.raises(NotImplementedError, match="does not support external point-charge"):
        theory.run(
            current_coords=fragment.coords, qm_elems=fragment.elems,
            charge=fragment.charge, mult=fragment.mult, PC=True,
        )
