import numpy as np

from ash import Fragment
from ash.modules.module_freq import approximate_full_Hessian_from_smaller


def make_fragment():
    # 5-atom test system
    elems = ["O", "H", "H", "C", "N"]
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0], [4.2, 0.0, 0.0]]
    return Fragment(elems=elems, coords=coords, charge=0, mult=1, printlevel=0)


def symmetric_matrix(dim):
    matrix = np.arange(1, dim * dim + 1, dtype=float).reshape(dim, dim)
    return (matrix + matrix.T) / 2


def hessian_indices(atomindices):
    return [3 * i + j for i in atomindices for j in (0, 1, 2)]


def test_partial_hessian_into_full_system():
    # Partial Hessian for atoms 1 and 3, expanded to the full system.
    # This is the case used by the DL-FIND interface (large_atomindices=all atoms).
    fragment = make_fragment()
    small_atoms = [1, 3]
    hessian_small = symmetric_matrix(3 * len(small_atoms))

    fullhessian = approximate_full_Hessian_from_smaller(fragment, hessian_small, small_atoms,
                                                        large_atomindices=fragment.allatoms,
                                                        restHessian=None)

    assert fullhessian.shape == (3 * fragment.numatoms, 3 * fragment.numatoms)
    # The small Hessian must end up at the Hessian-indices of its own atoms
    indices = hessian_indices(small_atoms)
    assert np.allclose(fullhessian[np.ix_(indices, indices)], hessian_small)
    # and everything else must still be zero
    rest = np.ones(fullhessian.shape, dtype=bool)
    rest[np.ix_(indices, indices)] = False
    assert np.allclose(fullhessian[rest], 0.0)


def test_partial_hessian_into_active_region():
    # Partial Hessian for atoms 1 and 3, expanded to an active region (atoms 1,2,3,4) only.
    # The atomindices of the small Hessian then have to be translated to the position of
    # those atoms within the active region, i.e. atom 1 -> 0 and atom 3 -> 2.
    fragment = make_fragment()
    small_atoms = [1, 3]
    large_atoms = [1, 2, 3, 4]
    hessian_small = symmetric_matrix(3 * len(small_atoms))

    fullhessian = approximate_full_Hessian_from_smaller(fragment, hessian_small, small_atoms,
                                                        large_atomindices=large_atoms,
                                                        restHessian=None)

    assert fullhessian.shape == (3 * len(large_atoms), 3 * len(large_atoms))
    indices = hessian_indices([large_atoms.index(i) for i in small_atoms])
    assert np.allclose(fullhessian[np.ix_(indices, indices)], hessian_small)
    rest = np.ones(fullhessian.shape, dtype=bool)
    rest[np.ix_(indices, indices)] = False
    assert np.allclose(fullhessian[rest], 0.0)


def test_partial_hessian_unit_restHessian():
    # restHessian='unit' should give a unit matrix outside the partial-Hessian block
    fragment = make_fragment()
    small_atoms = [0]
    hessian_small = symmetric_matrix(3 * len(small_atoms))

    fullhessian = approximate_full_Hessian_from_smaller(fragment, hessian_small, small_atoms,
                                                        large_atomindices=fragment.allatoms,
                                                        restHessian='unit')

    indices = hessian_indices(small_atoms)
    assert np.allclose(fullhessian[np.ix_(indices, indices)], hessian_small)
    # Diagonal outside the small-Hessian block is 1.0, off-diagonal 0.0
    outside = [i for i in range(3 * fragment.numatoms) if i not in indices]
    assert np.allclose(fullhessian[np.ix_(outside, outside)], np.identity(len(outside)))


def test_partial_hessian_is_symmetric():
    fragment = make_fragment()
    small_atoms = [2, 4]
    hessian_small = symmetric_matrix(3 * len(small_atoms))

    fullhessian = approximate_full_Hessian_from_smaller(fragment, hessian_small, small_atoms,
                                                        large_atomindices=fragment.allatoms,
                                                        restHessian=None)

    assert np.allclose(fullhessian, fullhessian.T)
