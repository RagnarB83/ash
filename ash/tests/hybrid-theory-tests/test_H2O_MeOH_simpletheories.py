from functions import transvar_check, numgrad_check, MockPairTheory
from ash import *

tolerance=1e-5


# H2O...MeOH fragment defined. Reading XYZ file
frag = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/h2o_MeOH.xyz")
xtbtheory = xTBTheory(xtbmethod="GFN2", accuracy=0.1)
#mocktheory = MockPairTheory()
pyscf_hf = PySCFTheory(scf_type="RHF", basis="STO-3G")

# xtb
def test_xtb():
    label="xtb"
    result= Singlepoint(theory=xtbtheory, fragment=frag, charge=0, mult=1, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, pyscf_hf, TOL=tolerance)

# Pyscf
def test_hf():
    label="hf"
    result= Singlepoint(theory=pyscf_hf, fragment=frag, charge=0, mult=1, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, pyscf_hf, TOL=tolerance)

