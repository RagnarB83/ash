"""Tests for the ActiveRegion option of the DL-FIND interface (ASH issue #520).

Without it, DL-FIND is handed the whole system with the inactive atoms frozen, and a
partial NumFreq Hessian is expanded to 3*numatoms x 3*numatoms before being passed on
(1.5e10 elements, ~114 GiB, for the 41318-atom system in the report). With
ActiveRegion=True only the active-region representation is passed, as in the geomeTRIC
interface, so the Hessian covers the active region instead.

These tests cover the index bookkeeping and the dimensions; an end-to-end DL-FIND run is
in examples/interfaces/DLFIND/DLFIND-ActiveRegion.py.
"""
import numpy as np
import pytest

pytest.importorskip("libdlfind")

from ash import Fragment
from ash.interfaces.interface_dlfind import DLFIND_optimizerClass
from ash.modules.module_freq import approximate_full_Hessian_from_smaller

COORDS = """
C   -1.9000   0.0000   0.0000
C   -0.4000   0.0000   0.0000
C    0.2000   1.4000   0.0000
O    1.6000   1.4500   0.0000
H   -2.3000   1.0200   0.0000
H   -2.3000  -0.5100   0.8800
H   -2.3000  -0.5100  -0.8800
H   -0.0300  -0.5300   0.8800
H   -0.0300  -0.5300  -0.8800
H   -0.1500   1.9500   0.8800
H   -0.1500   1.9500  -0.8800
H    1.9500   2.3500   0.0000
"""
ACTATOMS = [0, 1, 2, 3, 4, 5]


@pytest.fixture
def fragment():
    return Fragment(coordsstring=COORDS, charge=0, mult=1, printlevel=0)


def make_optimizer(fragment, **kwargs):
    kwargs.setdefault("jobtype", "opt")
    kwargs.setdefault("printlevel", 0)
    return DLFIND_optimizerClass(fragment=fragment, theory=None, **kwargs)


def test_nvar_is_active_region_only(fragment):
    """The number of variables DL-FIND works with must come from the active region."""
    opt = make_optimizer(fragment, ActiveRegion=True, actatoms=list(ACTATOMS))
    opt.setup_active_region_geometry()
    assert opt.dlfind_nvar() == 3 * len(ACTATOMS)
    # Default behaviour is unchanged: the whole fragment
    plain = make_optimizer(fragment, actatoms=list(ACTATOMS))
    assert plain.dlfind_nvar() == 3 * fragment.numatoms


def test_scatter_back_to_full_system(fragment):
    """Active-region coordinates go back to their own atoms; the rest is untouched."""
    opt = make_optimizer(fragment, ActiveRegion=True, actatoms=list(ACTATOMS))
    opt.setup_active_region_geometry()
    original = np.array(fragment.coords)
    act_coords = np.array(fragment.coords)[ACTATOMS] + 0.25
    full = opt.actregion_to_full_coords(act_coords)
    assert full.shape == original.shape
    for act_i, act_index in enumerate(ACTATOMS):
        assert np.allclose(full[act_index], act_coords[act_i])
    inactive = [i for i in range(fragment.numatoms) if i not in ACTATOMS]
    assert np.allclose(full[inactive], original[inactive])


def test_partial_hessian_expands_into_active_region(fragment):
    """The #520 dimension: a partial Hessian becomes 3*nact, not 3*numatoms."""
    hessatoms = [0, 1, 2]
    small = np.eye(3 * len(hessatoms))
    into_active = approximate_full_Hessian_from_smaller(fragment, small, hessatoms,
                                                        large_atomindices=ACTATOMS,
                                                        restHessian=None)
    into_full = approximate_full_Hessian_from_smaller(fragment, small, hessatoms,
                                                      large_atomindices=fragment.allatoms,
                                                      restHessian=None)
    assert into_active.shape == (3 * len(ACTATOMS), 3 * len(ACTATOMS))
    assert into_full.shape == (3 * fragment.numatoms, 3 * fragment.numatoms)
    # The partial block itself is identical in both, it is only the padding that differs
    block = [3 * i + j for i in range(len(hessatoms)) for j in range(3)]
    assert np.allclose(into_active[np.ix_(block, block)], into_full[np.ix_(block, block)])


def test_spec_all_active_with_nuccharges(fragment):
    """Every atom DL-FIND sees in ActiveRegion mode is active (spec 1)."""
    opt = make_optimizer(fragment, ActiveRegion=True, actatoms=list(ACTATOMS))
    opt.setup_active_region_geometry()
    opt.setup_constraints_act_frozen()
    nact = len(ACTATOMS)
    assert opt.spec[:nact] == [1] * nact
    # Nuclear charges of the active-region elements: C C C O H H
    assert opt.spec[nact:2 * nact] == [6, 6, 6, 8, 1, 1]
    # Microiterative block at the end
    assert opt.spec[2 * nact:] == [1] * nact
    assert opt.nspec == len(opt.spec)
    assert opt.numcons == 0


def test_residues_are_renumbered_for_the_active_region(fragment):
    """Residues are given in full-system indices and must come out as 1,2,3,... """
    residues = [[0, 1, 4, 5, 6], [2, 3, 7, 8], [9, 10, 11]]
    opt = make_optimizer(fragment, ActiveRegion=True, actatoms=list(ACTATOMS),
                         residues=residues)
    opt.setup_active_region_geometry()
    opt.setup_constraints_act_frozen()
    # actatoms 0,1,4,5 are in residue 0 -> 1 ; actatoms 2,3 are in residue 1 -> 2
    assert opt.spec[:len(ACTATOMS)] == [1, 1, 2, 2, 1, 1]


def test_constraints_are_reindexed(fragment):
    """Constraint atom indices must be translated into active-region indices."""
    actatoms = [0, 2, 5, 7]
    constraints = {'bond': [[0, 5]], 'angle': [[0, 2, 5]]}
    opt = make_optimizer(fragment, ActiveRegion=True, actatoms=list(actatoms),
                         constraints=constraints)
    opt.setup_active_region_geometry()
    opt.setup_constraints_act_frozen()
    nact = len(actatoms)
    conlist = opt.spec[2 * nact:-nact]
    # bond: type 1, atoms 0 and 5 -> act-indices 0 and 2, 1-based -> 1 and 3
    # angle: type 2, atoms 0,2,5 -> act-indices 0,1,2, 1-based -> 1,2,3
    assert conlist == [1, 1, 3, 0, 0, 2, 1, 2, 3, 0]
    assert opt.numcons == 2


def test_constraint_outside_active_region_is_rejected(fragment):
    opt = make_optimizer(fragment, ActiveRegion=True, actatoms=list(ACTATOMS),
                         constraints={'bond': [[0, 11]]})
    opt.setup_active_region_geometry()
    with pytest.raises(SystemExit):
        opt.setup_constraints_act_frozen()


def test_numfreq_hessatoms_defaults_to_active_region(fragment):
    opt = make_optimizer(fragment, ActiveRegion=True, actatoms=list(ACTATOMS),
                         hessian_choice="numfreq")
    opt.setup_active_region_geometry()
    assert opt.numfreq_hessatoms == ACTATOMS


def test_numfreq_hessatoms_outside_active_region_is_rejected(fragment):
    opt = make_optimizer(fragment, ActiveRegion=True, actatoms=list(ACTATOMS),
                         hessian_choice="numfreq", numfreq_hessatoms=[0, 11])
    with pytest.raises(SystemExit):
        opt.setup_active_region_geometry()


@pytest.mark.parametrize("kwargs", [
    {"actatoms": []},
    {"actatoms": None},
    {"actatoms": [0, 1, 1]},
    {"actatoms": [0, 1, 99]},
    {"actatoms": [0, 1], "frozenatoms": [5]},
])
def test_invalid_active_region_input_exits(fragment, kwargs):
    opt = make_optimizer(fragment, ActiveRegion=True, **kwargs)
    with pytest.raises(SystemExit):
        opt.setup_active_region_geometry()


def test_neb_with_active_region_is_rejected(fragment):
    opt = make_optimizer(fragment, jobtype="neb", nimage=5, ActiveRegion=True,
                         actatoms=list(ACTATOMS))
    with pytest.raises(SystemExit):
        opt.setup_active_region_geometry()
