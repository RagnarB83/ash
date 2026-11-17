import numpy as np
import pytest

import ash
from ash import Fragment, seminario


def test_seminario_uses_fragment_bonds_and_converts_force_constant_units():
    fragment = Fragment(
        elems=["H", "H"], coords=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
        charge=0, mult=1, printlevel=0,
    )
    force_constant_au = 0.25
    hessian = np.zeros((6, 6))
    hessian[0, 0] = hessian[3, 3] = force_constant_au
    hessian[0, 3] = hessian[3, 0] = -force_constant_au

    result = seminario(fragment=fragment, hessian=hessian)

    expected = force_constant_au * ash.constants.harkcal / ash.constants.bohr2ang**2
    assert len(result["bonds"]) == 1
    assert result["bonds"][0][:2] == (0, 1)
    assert result["bonds"][0][2] == pytest.approx(expected)
    assert result["angles"] == []
    assert result["dihedrals"] == []


def test_seminario_rejects_hessian_with_wrong_shape():
    fragment = Fragment(
        elems=["H", "H"], coords=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
        charge=0, mult=1, printlevel=0,
    )

    with pytest.raises(ValueError, match="Hessian shape"):
        seminario(fragment=fragment, hessian=np.zeros((3, 3)))


def test_seminario_angle_constant_includes_bond_length_factors():
    coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    fragment = Fragment(
        elems=["O", "H", "H"], coords=coords, charge=0, mult=1, printlevel=0,
    )
    coords_bohr = coords * ash.constants.ang2bohr
    vec_i = coords_bohr[1] - coords_bohr[0]
    vec_k = coords_bohr[2] - coords_bohr[0]
    normal = np.cross(vec_i, vec_k)
    normal /= np.linalg.norm(normal)
    tangent_i = np.cross(normal, vec_i)
    tangent_i /= np.linalg.norm(tangent_i)
    tangent_k = np.cross(vec_k, normal)
    tangent_k /= np.linalg.norm(tangent_k)
    force_constant_au = 0.25
    hessian = np.zeros((9, 9))
    for atom, tangent in ((1, tangent_i), (2, tangent_k)):
        block = -force_constant_au * np.outer(tangent, tangent)
        hessian[3 * atom:3 * atom + 3, 0:3] = block
        hessian[0:3, 3 * atom:3 * atom + 3] = block.T

    result = seminario(fragment=fragment, hessian=hessian)

    r_i = np.linalg.norm(vec_i)
    r_k = np.linalg.norm(vec_k)
    expected = ash.constants.harkcal / (
        1.0 / (r_i**2 * force_constant_au) + 1.0 / (r_k**2 * force_constant_au)
    )
    assert len(result["angles"]) == 1
    assert result["angles"][0] == pytest.approx((1, 0, 2, expected))


def test_seminario_rejects_unsupported_method():
    fragment = Fragment(
        elems=["H", "H"], coords=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
        charge=0, mult=1, printlevel=0,
    )

    with pytest.raises(ValueError, match="supports only method='original'"):
        seminario(fragment=fragment, hessian=np.zeros((6, 6)), method="modified")
