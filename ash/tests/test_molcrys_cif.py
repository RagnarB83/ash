from ash.functions.functions_molcrys import fill_unitcell, read_ciffile


def test_read_ciffile_modern_symmetry_loop_with_wyckoff_symbols(tmp_path):
    cif_path = tmp_path / "orthorhombic.cif"
    cif_path.write_text(
        """data_orthorhombic
_cell_length_a 11.905(7)
_cell_length_b 11.221(20)
_cell_length_c 6.397(4)
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_cell_formula_units_Z 4

loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x,y,z'
2 '-x+1/2,-y,z+1/2'
3 '-x,y+1/2,-z'
4 'x+1/2,-y+1/2,-z+1/2'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
V1 V 4a 0.1 0.2 0.3
""",
        encoding="utf-8",
    )

    cell_lengths, cell_angles, labels, elements, coordinates, symops, cellunits = read_ciffile(cif_path)

    assert cell_lengths == [11.905, 11.221, 6.397]
    assert cell_angles == [90.0, 90.0, 90.0]
    assert labels == ["V1"]
    assert elements == ["V"]
    assert coordinates == [[0.1, 0.2, 0.3]]
    assert cellunits == 4
    assert symops == [
        "x,y,z",
        "-x+1/2,-y,z+1/2",
        "-x,y+1/2,-z",
        "x+1/2,-y+1/2,-z+1/2",
    ]

    expanded, expanded_elements = fill_unitcell(
        cell_lengths, cell_angles, labels, elements, coordinates, symops
    )
    assert len(expanded) == 4
    assert expanded_elements == ["V"] * 4
