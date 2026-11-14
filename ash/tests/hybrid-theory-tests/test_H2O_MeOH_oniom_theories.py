from functions import transvar_check, numgrad_check
from ash import *

tolerance=1e-5

#ONIOM tests with xtb
# no covalent boundary

# H2O...MeOH fragment defined. Reading XYZ file
frag = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/h2o_MeOH.xyz")
region1=[3,4,5,6,7,8]
region2=[0,1,2]
#hl = PySCFTheory(scf_type="RKS", functional="PBE", basis="def2-SVP", densityfit=False)
hl = xTBTheory(xtbmethod="GFN2")
ll = xTBTheory(xtbmethod="GFN1")

# Mech
def test_oniom_nocovalboundary_mech1():
    label="ONIOM-nocovalboundary_mech1"
    theory = ONIOMTheory(fragment=frag, theories_N=[hl,ll], regions_N=[region1,[]],
        fullregion_charge=0, fullregion_mult=1, regions_chargemult=[[0,1],[0,1]], embedding="mechanical")
    result= Singlepoint(theory=theory, fragment=frag, charge=0, mult=1, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)

# Mech: swap region
def test_oniom_nocovalboundary_mech2():
    label="ONIOM-nocovalboundary_mech2"
    theory = ONIOMTheory(fragment=frag, theories_N=[hl,ll], regions_N=[region2,[]],
        fullregion_charge=0, fullregion_mult=1, regions_chargemult=[[0,1],[0,1]], embedding="mechanical")
    result= Singlepoint(theory=theory, fragment=frag, charge=0, mult=1, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)

# Mech: Swap HL and LL
def test_oniom_nocovalboundary_mech3():
    label="ONIOM-nocovalboundary_mech3"
    theory = ONIOMTheory(fragment=frag, theories_N=[ll,hl], regions_N=[region1,[]],
        fullregion_charge=0, fullregion_mult=1, regions_chargemult=[[0,1],[0,1]], embedding="mechanical")
    result= Singlepoint(theory=theory, fragment=frag, charge=0, mult=1, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)

# Elstat
def test_oniom_nocovalboundary_elstat():
    label="ONIOM-nocovalboundary_elstat1"
    theory = ONIOMTheory(fragment=frag, theories_N=[hl,ll], regions_N=[region1,[]],
        fullregion_charge=0, fullregion_mult=1, regions_chargemult=[[0,1],[0,1]], embedding="electrostatic")
    result= Singlepoint(theory=theory, fragment=frag, charge=0, mult=1, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)

# Elstat: Swap HL and LL
def test_oniom_nocovalboundary_elstat2():
    label="ONIOM-nocovalboundary_elstat2"
    theory = ONIOMTheory(fragment=frag, theories_N=[ll,hl], regions_N=[region1,[]],
        fullregion_charge=0, fullregion_mult=1, regions_chargemult=[[0,1],[0,1]], embedding="electrostatic")
    result= Singlepoint(theory=theory, fragment=frag, charge=0, mult=1, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)
