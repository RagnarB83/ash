from ash import *
from math import isclose
import numpy as np

coords="""
O       -1.377626260      0.000000000     -1.740199718
H       -1.377626260      0.759337000     -1.144156718
H       -1.377626260     -0.759337000     -1.144156718
"""
#Defining fragment
H2O=Fragment(coordsstring=coords,charge=0,mult=1)

def test_xtb_load():
    global H2O
    #Default
    xtb_default = xTBTheory()
    assert xtb_default.xtbmethod=='GFN1'

    xtb_default.cleanup()
    result_inp = Singlepoint(theory=xtb_default,fragment=H2O)
    energy_inputfile = result_inp.energy
    result_inp2 = Singlepoint(theory=xtb_default,fragment=H2O, Grad=True)
    energy_inputfile = result_inp2.energy
    grad_inputfile = result_inp2.gradient

    #Specifying dummy xtbdir and changing xtbmethod
    xtb_with_dir_gfn2 = xTBTheory(xtbdir='/path/to/xtb',
        xtbmethod='GFN2', runmode='inputfile')
    assert xtb_with_dir_gfn2.xtbdir=='/path/to/xtb'
    assert xtb_with_dir_gfn2.xtbmethod=='GFN2'

    #Specifying library input
    xtb_library = xTBTheory(runmode='library')

    result_lib = Singlepoint(theory=xtb_library,fragment=H2O)
    energy_library = result_lib.energy

    result_lib2 = Singlepoint(theory=xtb_library,fragment=H2O, Grad=True)
    energy_library = result_lib2.energy
    grad_library = result_lib2.gradient
    refenergy=-5.768502689118895
    assert isclose(energy_library,energy_inputfile)
    assert isclose(energy_library,refenergy)

    #Comparing gradients
    assert np.allclose(grad_inputfile,grad_library,rtol=1e-4)

    xtb_default.cleanup()
    xtb_library.cleanup()
