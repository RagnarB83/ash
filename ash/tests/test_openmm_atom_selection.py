from types import SimpleNamespace

import pytest

from ash.interfaces.interface_OpenMM import OpenMMTheory


def _atom(index, residue_name, element_symbol):
    return SimpleNamespace(
        index=index,
        residue=SimpleNamespace(name=residue_name),
        element=SimpleNamespace(symbol=element_symbol),
    )


def _theory_with_atoms(atoms):
    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.topology = SimpleNamespace(atoms=lambda: iter(atoms))
    theory.mm_elements = []
    return theory


def test_openmm_atom_indices_for_protein_heavy_atoms_and_solvent():
    theory = _theory_with_atoms([
        _atom(0, "ALA", "C"),
        _atom(1, "ALA", "H"),
        _atom(2, "ALA", "N"),
        _atom(3, "HOH", "O"),
        _atom(4, "HOH", "H"),
        _atom(5, "LIG", "C"),
        _atom(6, "SOL", "O"),
    ])

    assert theory.get_nonH_atomindices() == [0, 2, 3, 5, 6]
    assert theory.get_protein_heavy_atomindices() == [0, 2]
    assert theory.get_solvent_atomindices() == [3, 4, 6]


def test_openmm_atom_selection_accepts_custom_residue_names():
    theory = _theory_with_atoms([
        _atom(0, "PRT", "C"),
        _atom(1, "MYW", "O"),
    ])

    assert theory.get_protein_heavy_atomindices("prt") == [0]
    assert theory.get_solvent_atomindices(["myw"]) == [1]


def test_openmm_atom_selection_requires_topology_metadata():
    theory = OpenMMTheory.__new__(OpenMMTheory)

    with pytest.raises(ValueError, match="topology with residue and element information"):
        theory.get_protein_heavy_atomindices()
