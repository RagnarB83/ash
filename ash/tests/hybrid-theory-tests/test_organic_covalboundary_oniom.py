from functions import transvar_check, numgrad_check
from ash import *

tolerance=1e-5

# Hybrid-theory tests with xtb.
# Testing ONIOMTheory with a covalent boundary

# educt system (from BHDIV10 in GMTKN55). Enol reactant for a dehydration reaction.
frag = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/educt.xyz", charge=0, mult=1)
pdbfile=f"{ashpath}/tests/pdbfiles/educt.pdb"
xmlfile=f"{ashpath}/tests/extra_files/gaff_educt.xml"

region1_atoms=[2,3,6,7,8,9,10,11,12,13]
region2_atoms=listdiff(frag.allatoms,region1_atoms)
hl = xTBTheory(xtbmethod="GFN2")
ll = xTBTheory(xtbmethod="GFN1")

# Mechanical embedding ONIOM: looping over linkatom methods
def test_oniom_coval_mech():
    label="ONIOM-covalboundary_mech"

    for lsetting in ["simple", "ratio"]:
        print("Linkatom method:", lsetting)
        label="oniom_coval_mech_"+str(lsetting)
        theory = ONIOMTheory(fragment=frag, theories_N=[hl,ll], regions_N=[region1_atoms,region2_atoms],
                fullregion_charge=0, fullregion_mult=1, regions_chargemult=[[0,1],[0,1]], embedding="mech",
                linkatom_method=lsetting)

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, label=label, TOL=tolerance)

# Electrostatic embedding ONIOM: looping over linkatom methods
def test_oniom_coval_elstat():
    label="ONIOM-covalboundary_elstat"

    for lsetting in [("simple",False),("simple",True), ("ratio", False),("ratio", True)]:
        print("Linkatom method:", lsetting)
        label="oniom_coval_elstat_"+str(lsetting)
        theory = ONIOMTheory(fragment=frag, theories_N=[hl,ll], regions_N=[region1_atoms,region2_atoms],
                fullregion_charge=0, fullregion_mult=1, regions_chargemult=[[0,1],[0,1]], embedding="elstat",
                linkatom_method=lsetting[0], dipole_correction=lsetting[1])

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        #transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, label=label, TOL=tolerance)

# Electrostatic embedding with RCD ONIOM: looping over linkatom methods
def test_oniom_coval_elstat_rcd():
    label="ONIOM-covalboundary_elstat_rcd"

    for lsetting in ["simple", "ratio"]:
        print("Linkatom method:", lsetting)
        label="oniom_coval_elstat_rcd_"+str(lsetting)
        theory = ONIOMTheory(fragment=frag, theories_N=[hl,ll], regions_N=[region1_atoms,region2_atoms],
                fullregion_charge=0, fullregion_mult=1, regions_chargemult=[[0,1],[0,1]], embedding="elstat",
                linkatom_method=lsetting, chargeboundary_method="rcd")

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        #transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, label=label, TOL=tolerance)

#test_oniom_theory_swap()
