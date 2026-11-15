"""Tests for PSF segment-name handling used by add_atoms_to_PSF (ASH issue #354).

add_atoms_to_PSF used a random 4-character dummy segment name ("AD" + letter + digit),
which can collide with a segment that already exists in the PSF-file and makes psfgen
fail. The name is now chosen deterministically as the first unused ADD1, ADD2, ...
"""
import os

import pytest

from ash.modules.module_coords import (find_free_segment_name,
                                       get_segment_names_from_PSF)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_PSF = os.path.join(HERE, "..", "..", "examples", "QM-MM-examples",
                           "QM-MM-CHARMM-example", "step3_pbcsetup.psf")


def write_psf(path, segnames):
    """Minimal X-PLOR PSF with one atom per requested segment name."""
    lines = ["PSF EXT CMAP CHEQ XPLOR", "", "         1 !NTITLE",
             "* TEST FILE", "", "    {:6d} !NATOM".format(len(segnames))]
    for i, segname in enumerate(segnames, start=1):
        lines.append("{:10d} {:8s} {:8d} {:8s} {:8s} {:8s} {:14.6f} {:14.5f} {:11d}".format(
            i, segname, 1, "SER", "N", "NH3", -0.3, 14.007, 0))
    lines += ["", "         0 !NBOND: bonds", ""]
    path.write_text("\n".join(lines))
    return str(path)


def test_segment_names_parsed_in_order(tmp_path):
    psffile = write_psf(tmp_path / "a.psf", ["PROA", "PROA", "SOLV", "POT"])
    assert get_segment_names_from_PSF(psffile) == ["PROA", "SOLV", "POT"]


def test_first_free_name_is_add1(tmp_path):
    psffile = write_psf(tmp_path / "b.psf", ["PROA", "SOLV"])
    assert find_free_segment_name(psffile) == "ADD1"


def test_existing_add_names_are_skipped(tmp_path):
    """The whole point of the fix: never hand back a name that is already in the PSF."""
    psffile = write_psf(tmp_path / "c.psf", ["PROA", "ADD1", "ADD2", "SOLV"])
    name = find_free_segment_name(psffile)
    assert name == "ADD3"
    assert name not in get_segment_names_from_PSF(psffile)


def test_gap_in_add_names_takes_lowest_free(tmp_path):
    psffile = write_psf(tmp_path / "d.psf", ["ADD2"])
    assert find_free_segment_name(psffile) == "ADD1"


def test_custom_prefix(tmp_path):
    psffile = write_psf(tmp_path / "e.psf", ["NEW1"])
    assert find_free_segment_name(psffile, prefix="NEW") == "NEW2"


def test_atom_block_is_not_read_past_natom(tmp_path):
    """Sections after !NATOM (bonds etc.) must not be mistaken for atom lines."""
    psffile = write_psf(tmp_path / "f.psf", ["PROA"])
    assert get_segment_names_from_PSF(psffile) == ["PROA"]


@pytest.mark.skipif(not os.path.isfile(EXAMPLE_PSF), reason="example PSF not present")
def test_real_charmm_gui_psf():
    segnames = get_segment_names_from_PSF(EXAMPLE_PSF)
    assert segnames == ["PROA", "SOLV", "POT", "CLA"]
    assert find_free_segment_name(EXAMPLE_PSF) == "ADD1"
