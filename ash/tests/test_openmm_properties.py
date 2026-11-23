"""OpenMMTheory must not modify the properties dict it is given.

The platform-properties dict was stored by reference, and OpenMMTheory then wrote
"Threads" into it, so the caller's own dict was silently changed and a dict reused for
a second theory object carried the first object's settings.
"""
import os

import pytest

openmm = pytest.importorskip("openmm")

from ash import Fragment, OpenMMTheory, ashpath

PDBFILE = os.path.join(ashpath, "tests", "pdbfiles", "1aki_solvated.pdb")


def make_theory(properties, numcores=4):
    return OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"], pdbfile=PDBFILE,
                        periodic=True, autoconstraints=None, rigidwater=False,
                        platform='CPU', numcores=numcores, properties=properties,
                        printlevel=0)


@pytest.mark.skipif(not os.path.isfile(PDBFILE), reason="test PDB-file not present")
def test_caller_dict_is_not_modified():
    caller_properties = {"DeterministicForces": "true"}
    before = dict(caller_properties)
    theory = make_theory(caller_properties)
    assert caller_properties == before, "OpenMMTheory modified the dict it was given"
    assert theory.properties is not caller_properties
    # The theory object itself still gets its Threads entry
    assert theory.properties["Threads"] == "4"
    assert theory.properties["DeterministicForces"] == "true"


@pytest.mark.skipif(not os.path.isfile(PDBFILE), reason="test PDB-file not present")
def test_same_dict_can_be_reused_for_two_theories():
    """A dict handed to two theory objects must not carry the first one's settings."""
    caller_properties = {"DeterministicForces": "true"}
    first = make_theory(caller_properties, numcores=4)
    second = make_theory(caller_properties, numcores=2)
    assert first.properties["Threads"] == "4"
    assert second.properties["Threads"] == "2"
    assert "Threads" not in caller_properties


@pytest.mark.skipif(not os.path.isfile(PDBFILE), reason="test PDB-file not present")
def test_properties_none_still_works():
    theory = make_theory(None)
    assert theory.properties["Threads"] == "4"
