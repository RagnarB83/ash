from functions import transvar_check, numgrad_check
from ash import *

grad_tolerance=1e-4
e_tolerance=1e-8 # Eh

# Test for making sure that QMMMTheory-mechanical and analogous WrapTheory definition give same energies
# This test the energy-definition of WrapTheory, the linkatom routines etc.

# Test system: Ferredoxin protein with one Fe2S2 site in QM-region
pdbfile=f"{ashpath}/tests/pdbfiles/ferredoxin_prep_warmedup.pdb"
frag = Fragment(pdbfile=pdbfile)
region1 = [2692, 2693, 2694, 2695, 558, 619, 651, 1057]
qm = xTBTheory(xtbmethod="GFN1")

def test_qm_mm_mech_vs_wrap():
    label="qm_mm_mech"
    # Mechanical embedding QM/MM
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
    pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None, do_energy_decomposition=True)
    theory = QMMMTheory(qm_theory=qm, mm_theory=omm, qmatoms=region1, embedding="mech", fragment=frag,
                    qm_charge=-2, qm_mult=11, unusualboundary=True, linkatom_method="ratio", printlevel=3)
    result1 = Singlepoint(theory=theory, fragment=frag, Grad=True, charge=-2, mult=11)
    e_qm_mm = result1.energy
    transvar_check(result1.gradient, label=label, tolerance=grad_tolerance)
    ##########################################
    # Wrap1: Defining 
    label="wrap1"
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
    pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None, do_energy_decomposition=True)
    # Constraints
    omm.remove_constraints_for_atoms(region1)
    # Remove possible bonded-forces
    omm.modify_bonded_forces(region1)
    # Removing internal QM-QM interactions
    omm.addexceptions(region1)
    wrap1 = WrapTheory(theories=[qm,omm], theory1_atoms=region1,
            theories_chargemult=[[-2,11],[0,1]], printlevel=2)
    result_wrap = Singlepoint(theory=wrap1, fragment=frag, Grad=True, charge=-2, mult=11)
    transvar_check(result_wrap.gradient, label=label, tolerance=grad_tolerance)
    e_wrap1 = result_wrap.energy
    ######################################
    # Wrap2: swapping 
    label="wrap2"
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
    pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None, do_energy_decomposition=True)
    # Constraints
    omm.remove_constraints_for_atoms(region1)
    # Remove possible bonded-forces
    omm.modify_bonded_forces(region1)
    # Removing internal QM-QM interactions
    omm.addexceptions(region1)
    wrap2 = WrapTheory(theories=[omm,qm], theory2_atoms=region1,
            theories_chargemult=[[0,1],[-2,11]], printlevel=2)
    result_wrap2 = Singlepoint(theory=wrap2, fragment=frag, Grad=True, charge=-2, mult=11)
    transvar_check(result_wrap2.gradient, label=label, tolerance=grad_tolerance)
    e_wrap2 = result_wrap2.energy

    print("e_qm_mm:", e_qm_mm)
    print("e_wrap1:", e_wrap1)
    print("e_wrap2:", e_wrap2)
    assert abs(e_qm_mm -e_wrap1 ) < e_tolerance, f"E_QMM={e_qm_mm} differs from E_Wrap={e_wrap1}!" 
    assert abs(e_wrap1 -e_wrap2 ) < e_tolerance, f"E_QMM={e_qm_mm} differs from E_Wrap={e_wrap1}!"

#test_qm_mm_mech_vs_wrap()
