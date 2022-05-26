from ash import *

def test_chargemult():
    #Testing charge/mult definitions

    fragcoords="""
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """
    HF_frag=Fragment(coordsstring=fragcoords)
    HF_frag2=Fragment(coordsstring=fragcoords, charge=0, mult=1)

    assert HF_frag.charge == None, "Charge is not None"
    assert HF_frag2.charge == 0, "Charge is not 0"
