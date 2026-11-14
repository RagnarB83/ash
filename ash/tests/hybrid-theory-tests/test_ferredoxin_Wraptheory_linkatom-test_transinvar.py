from functions import transvar_check, numgrad_check
from ash import *

grad_tolerance=4e-4

# Testing translational invariance of gradient
# Test system: Ferredoxin protein with one Fe2S2 site in QM-region
# 
pdbfile=f"{ashpath}/tests/pdbfiles/ferredoxin_prep_warmedup.pdb"
frag = Fragment(pdbfile=pdbfile)
region1 = [2692, 2693, 2694, 2695, 558, 619, 651, 1057]
qm = xTBTheory(xtbmethod="GFN1")

def test_wrap_mech_linkatom_loop():
    # Looping over linkatom-method, L-forceproj and dipole-corr
    for loption in ["simple", "ratio"]:
        label="wrap_mech_"+str(loption)
        # Mechanical embedding Wrap
        omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
        pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None)
        # Removing constraints, bonded forces and internal QM-QM interactions
        omm.remove_constraints_for_atoms(region1)
        omm.modify_bonded_forces(region1)
        omm.addexceptions(region1)
        theory = WrapTheory(theories=[qm,omm], theory1_atoms=region1, theories_chargemult=[[-2,11],[0,1]], 
                            printlevel=2, linkatom_method=loption)

        result = Singlepoint(theory=theory, fragment=frag, Grad=True, charge=-2, mult=11)
        transvar_check(result.gradient, label=label, tolerance=grad_tolerance)

test_wrap_mech_linkatom_loop()
