from ash import *

def test_ORCA_SP():
    """
    Simple Singlepoint ORCA calculation with charge/mult in fragment
    """
    #ORCA
    orcasimpleinput="! BP86 def2-SVP tightscf"
    orcablocks="%scf maxiter 200 end"

    fragcoords="""
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    #Add coordinates to fragment
    HF_frag=Fragment(coordsstring=fragcoords, charge=0, mult=1)

    ORCASPcalculation = ORCATheory(fragment=HF_frag, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

    energy = Singlepoint(fragment=HF_frag, theory=ORCASPcalculation)

    #Clean up
    ORCASPcalculation.cleanup()

    #Reference energy
    ref=-100.350611851152
    threshold=1e-9
    assert abs(energy-ref) < threshold, "Energy-error above threshold"

def test_ORCA_SP2():
    """
    Simple Singlepoint ORCA calculation with charge/mult in Singlepoint function
    """
    #ORCA
    orcasimpleinput="! BP86 def2-SVP tightscf"
    orcablocks="%scf maxiter 200 end"

    fragcoords="""
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    #Add coordinates to fragment
    HF_frag=Fragment(coordsstring=fragcoords)

    ORCASPcalculation = ORCATheory(fragment=HF_frag, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

    energy = Singlepoint(fragment=HF_frag, theory=ORCASPcalculation, charge=0, mult=1)

    #Clean up
    ORCASPcalculation.cleanup()

    #Reference energy
    ref=-100.350611851152
    threshold=1e-9
    assert abs(energy-ref) < threshold, "Energy-error above threshold"

def test_ORCA_BS_SP():
    """
    Singlepoint Broken-symmetry ORCA calculation with charge/mult in fragment
    """
    #ORCA
    orcasimpleinput="! BP86 def2-SVP  tightscf"
    orcablocks="%scf maxiter 200 end"

    fragcoords="""
    Fe 0.0 0.0 0.0
    Fe 0.0 0.0 3.0
    """

    #Add coordinates to fragment
    Fe2_frag=Fragment(coordsstring=fragcoords, charge=6, mult=1)



    ORCABScalc = ORCATheory(orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                    brokensym=True, HSmult=11, atomstoflip=[1])

    #Simple Energy SP calc
    energy = Singlepoint(fragment=Fe2_frag, theory=ORCABScalc)

    #Clean up
    ORCABScalc.cleanup()

    #Reference energy
    ref=-2521.60831655367
    threshold=1e-9
    assert abs(energy-ref) < threshold, "Energy-error above threshold"
