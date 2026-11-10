from functions import transvar_check, numgrad_check
from ash import *

tolerance=1e-5

# Hybrid-theory tests with xtb.
# Testing WrapTheory with a covalent boundary

# educt system (from BHDIV10 in GMTKN55). Enol reactant for a dehydration reaction.
frag = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/educt.xyz", charge=0, mult=1)
pdbfile=f"{ashpath}/tests/pdbfiles/educt.pdb"
xmlfile=f"{ashpath}/tests/extra_files/gaff_educt.xml"

region1_atoms=[2,3,6,7,8,9,10,11,12,13]
region2_atoms=listdiff(frag.allatoms,region1_atoms)
hl = xTBTheory(xtbmethod="GFN2")
ll = xTBTheory(xtbmethod="GFN1")

# Mechanical embedding WrapTheory: looping over linkatom methods
def test_wrap_coval_mech():
    label="Wrap-covalboundary_mech"

    for lsetting in ["simple", "ratio"]:
        print("Linkatom method:", lsetting)
        label=str(lsetting)
        theory = WrapTheory(theories=[hl,ll], theory1_atoms=region1_atoms, theory2_atoms=region2_atoms,
                linkatom_method=lsetting)

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, TOL=tolerance)

# Mechanical embedding WrapTheory 2: looping over linkatom methods
# Swapping atom-regions
def test_wrap_coval_mech2():
    label="Wrap-covalboundary_mech2"

    for lsetting in ["simple", "ratio"]:
        print("Linkatom method:", lsetting)
        label=str(lsetting)
        theory = WrapTheory(theories=[hl,ll], theory1_atoms=region2_atoms, theory2_atoms=region1_atoms,
                linkatom_method=lsetting)

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, TOL=tolerance)

# Mechanical embedding WrapTheory 2: looping over linkatom methods
# Swapping theories
def test_wrap_coval_mech3():
    label="Wrap-covalboundary_mech3"

    for lsetting in ["simple", "ratio"]:
        print("Linkatom method:", lsetting)
        label=str(lsetting)
        theory = WrapTheory(theories=[ll,hl], theory1_atoms=region2_atoms, theory2_atoms=region1_atoms,
                linkatom_method=lsetting)

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, TOL=tolerance)
