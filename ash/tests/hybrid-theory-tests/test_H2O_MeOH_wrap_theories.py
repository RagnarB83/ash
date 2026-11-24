from functions import transvar_check, numgrad_check
from ash import *

tolerance=1e-5

#Wraptheory tests with xtb
# no covalent boundary

# H2O...MeOH fragment defined. Reading XYZ file
frag = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/h2o_MeOH.xyz", charge=0, mult=1)
region1=[3,4,5,6,7,8]
region2=[0,1,2]
hl = xTBTheory(xtbmethod="GFN2", accuracy=0.01)
ll = xTBTheory(xtbmethod="GFN1", accuracy=0.01)

# WrapTheory: HL on region1, LL on region 2 
def test_wrap_nocovalboundary():
    label="Wrap-nocovalboundary"
    theory = WrapTheory(theories=[ll,hl], theory1_atoms=region1, theory2_atoms=region2, theories_chargemult=[[0,1],[0,1]])

    result= Singlepoint(theory=theory, fragment=frag, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)
