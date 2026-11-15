"""Tests for print_fragments_table (ASH issue #520-unrelated; issue #336 table formatting).

The table used hardcoded column widths (formula 15, label 25), so any fragment with a
longer formula or label pushed the Charge/Mult/Energy columns out of alignment with the
header. These tests pin the alignment for long entries and check that the separator
lines match the real table width.
"""
import pytest

from ash.modules.module_singlepoint import print_fragments_table


class FakeFragment:
    def __init__(self, formula, label, charge, mult):
        self.formula = formula
        self.label = label
        self.charge = charge
        self.mult = mult


def table_lines(capsys, fragments, energies, **kwargs):
    print_fragments_table(fragments, energies, **kwargs)
    out = capsys.readouterr().out
    return [line for line in out.split("\n") if line.strip() != ""]


def column_starts(line, ncols):
    """Start index of each whitespace-separated column, from the right edge inwards."""
    # Right-aligned columns are compared by their end positions
    ends = []
    index = 0
    for part in line.split():
        index = line.index(part, index)
        ends.append(index + len(part))
        index += len(part)
    assert len(ends) == ncols
    return ends


def test_long_label_stays_aligned(capsys):
    """A label longer than the old 25-char field must not shift the later columns."""
    fragments = [
        FakeFragment("H2O", "water", 0, 1),
        FakeFragment("C6H12O6", "glucose-alpha-D-pyranose-conformerA", 0, 1),
    ]
    lines = table_lines(capsys, fragments, [-76.4, -686.9])
    header = lines[3]
    rows = lines[5:7]
    # Charge, Mult and Energy are right-aligned: their right edges must line up with
    # the corresponding header right edges on every row
    header_ends = column_starts(header, 5)
    for row in rows:
        row_ends = column_starts(row, 5)
        assert row_ends[2:] == header_ends[2:], f"misaligned row: {row!r}"


def test_long_formula_stays_aligned(capsys):
    """Same for a formula longer than the old 15-char field."""
    fragments = [
        FakeFragment("H2O", "water", 0, 1),
        FakeFragment("Fe1C24H30N4O2S2Cl3", "complex", -2, 5),
    ]
    lines = table_lines(capsys, fragments, [-76.4, -1234.5])
    header_ends = column_starts(lines[3], 5)
    for row in lines[5:7]:
        assert column_starts(row, 5)[2:] == header_ends[2:], f"misaligned row: {row!r}"


def test_separators_match_table_width(capsys):
    """The ==== and ---- rules must be as wide as the widest content line, not a fixed 100."""
    fragments = [FakeFragment("H2O", "water", 0, 1)]
    lines = table_lines(capsys, fragments, [-76.4])
    rules = [line for line in lines if set(line) in ({"="}, {"-"})]
    assert len(rules) == 4, lines
    widths = {len(rule) for rule in rules}
    assert len(widths) == 1, f"rules of differing width: {widths}"
    content = [line for line in lines if set(line) not in ({"="}, {"-"})]
    # Title line excluded: it is free text, not a table row
    rows = [line for line in content if "Table of energies" not in line]
    assert widths.pop() == max(len(row) for row in rows)


def test_none_label_and_values_printed(capsys):
    """A fragment with label None prints 'None' and the energy to 10 decimals."""
    fragments = [FakeFragment("H2", None, 0, 1)]
    lines = table_lines(capsys, fragments, [-1.1175059999])
    row = lines[5]
    assert "None" in row
    assert "-1.1175059999" in row
    assert "H2" in row


def test_unit_appears_in_header(capsys):
    fragments = [FakeFragment("H2O", "water", 0, 1)]
    lines = table_lines(capsys, fragments, [0.0], unit="kcal/mol")
    assert "Energy(kcal/mol)" in lines[3]


def test_empty_fragment_list(capsys):
    """No fragments must not raise (max() over an empty sequence)."""
    lines = table_lines(capsys, [], [])
    assert "Formula" in lines[3]
