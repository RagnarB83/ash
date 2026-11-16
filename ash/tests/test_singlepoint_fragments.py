"""Singlepoint_fragments: result attributes, relative energies and input checks (#336).

Three things were wrong in Singlepoint_fragments:

1. ASH_Results.charge/mult were set from the loop variable left over after the fragment
   loop, so they reported whichever fragment happened to be last. That is exactly wrong
   for the use case in #336, a set of isomers or spin-states with different charge/mult.
   They are single-job attributes, so they are now only set when every fragment agrees.
2. An empty or missing fragment list raised UnboundLocalError from deep inside the
   function instead of an ASH error message.
3. An unrecognised unit raised KeyError instead of saying which units are valid.

result.labels is now always set, not only when relative_energies=True, so the energies
can be matched to the fragments in the default case too.

A stub theory is used so no QM program is needed.
"""
import numpy as np
import pytest

from ash import Fragment
from ash.modules.module_singlepoint import Singlepoint_fragments

# Energies handed out in order, lowest one in the middle
STUB_ENERGIES = [-10.0, -10.5, -9.8]

# Expected Eh -> unit factors, written out here on purpose rather than imported from the
# module under test, so the test checks the values instead of comparing them to themselves
EXPECTED_FACTORS = {'kcal/mol': 627.50946900, 'kcalpermol': 627.50946900,
                    'kJ/mol': 2625.499638, 'kJpermol': 2625.499638,
                    'eV': 27.211386245988, 'cm-1': 219474.6313702,
                    'Eh': 1.0, 'mEh': 1000, 'meV': 27211.386245988}


class StubTheory:
    """Returns a fixed energy per call, in order."""
    theorytype = "QM"
    numcores = 1
    label = "stub"
    filename = "stub"
    printlevel = 1

    def __init__(self, energies=None):
        self.energies = list(STUB_ENERGIES if energies is None else energies)
        self.callcount = 0

    def run(self, current_coords=None, elems=None, Grad=False, charge=None, mult=None,
            **kwargs):
        energy = self.energies[self.callcount]
        self.callcount += 1
        return energy

    def cleanup(self):
        pass


def make_fragment(label, dist, charge, mult):
    return Fragment(elems=["H", "H"], coords=[[0.0, 0.0, 0.0], [0.0, 0.0, dist]],
                    charge=charge, mult=mult, label=label, printlevel=0)


def mixed_fragments():
    """Fragments with different charge and mult, as in a spin-state/redox series."""
    return [make_fragment("neutral", 0.74, 0, 1),
            make_fragment("anion", 0.80, -1, 2),
            make_fragment("dication", 0.70, 2, 3)]


def same_chargemult_fragments():
    """Conformers: same charge and mult throughout."""
    return [make_fragment("conf1", 0.74, 0, 1),
            make_fragment("conf2", 0.80, 0, 1),
            make_fragment("conf3", 0.70, 0, 1)]


##############################################
# charge/mult on the results object
##############################################

def test_charge_mult_none_for_mixed_fragments():
    """Must not report the last fragment's charge/mult for a mixed set."""
    result = Singlepoint_fragments(theory=StubTheory(), fragments=mixed_fragments())

    assert result.charge is None
    assert result.mult is None


def test_charge_mult_kept_when_all_fragments_agree():
    """When unambiguous, charge/mult are still reported."""
    result = Singlepoint_fragments(theory=StubTheory(),
                                   fragments=same_chargemult_fragments())

    assert result.charge == 0
    assert result.mult == 1


def test_labels_always_set():
    """labels used to be set only when relative_energies was True."""
    result = Singlepoint_fragments(theory=StubTheory(), fragments=mixed_fragments())

    assert result.labels == ["neutral", "anion", "dication"]
    assert result.energies == STUB_ENERGIES


##############################################
# Relative energies
##############################################

def test_relative_energies_kcal_per_mol():
    """Relative to the lowest fragment, which must come out as exactly zero."""
    result = Singlepoint_fragments(theory=StubTheory(), fragments=mixed_fragments(),
                                   relative_energies=True, unit='kcal/mol')

    lowest = min(STUB_ENERGIES)
    expected = [(e - lowest) * EXPECTED_FACTORS["kcal/mol"]
                for e in STUB_ENERGIES]
    assert np.allclose(result.relative_energies, expected)
    assert min(result.relative_energies) == 0.0
    assert result.relative_energies[STUB_ENERGIES.index(lowest)] == 0.0


def test_relative_energies_in_Eh_are_plain_differences():
    result = Singlepoint_fragments(theory=StubTheory(), fragments=mixed_fragments(),
                                   relative_energies=True, unit='Eh')

    lowest = min(STUB_ENERGIES)
    assert np.allclose(result.relative_energies, [e - lowest for e in STUB_ENERGIES])


@pytest.mark.parametrize("unit", sorted(EXPECTED_FACTORS))
def test_every_documented_unit_is_accepted(unit):
    result = Singlepoint_fragments(theory=StubTheory(), fragments=mixed_fragments(),
                                   relative_energies=True, unit=unit)

    assert result.relative_energies is not None
    assert min(result.relative_energies) == 0.0


##############################################
# Input checks
##############################################

@pytest.mark.parametrize("fragments", [[], None])
def test_empty_or_missing_fragments_exits_cleanly(fragments):
    """Used to raise UnboundLocalError on the leftover charge variable."""
    with pytest.raises(SystemExit):
        Singlepoint_fragments(theory=StubTheory(), fragments=fragments)


def test_unknown_unit_exits_cleanly():
    """Used to raise KeyError from the conversion-factor lookup."""
    with pytest.raises(SystemExit):
        Singlepoint_fragments(theory=StubTheory(), fragments=mixed_fragments(),
                              relative_energies=True, unit='hartree')


def test_unknown_unit_not_checked_without_relative_energies():
    """unit is only used for relative energies, so it must not break the default path."""
    result = Singlepoint_fragments(theory=StubTheory(), fragments=mixed_fragments(),
                                   unit='hartree')

    assert result.energies == STUB_ENERGIES
