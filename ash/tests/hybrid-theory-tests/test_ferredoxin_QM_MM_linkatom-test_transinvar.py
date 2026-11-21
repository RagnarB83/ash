from functions import transvar_check, numgrad_check
from ash import *

#grad_tolerance=4e-4
grad_tolerance=1e-4

# Testing translational invariance of gradient
# Test system: Ferredoxin protein with one Fe2S2 site in QM-region
# 
pdbfile=f"{ashpath}/tests/pdbfiles/ferredoxin_prep_warmedup.pdb"
frag = Fragment(pdbfile=pdbfile)
region1 = [2692, 2693, 2694, 2695, 558, 619, 651, 1057]
qm = xTBTheory(xtbmethod="GFN1")

# Mechanical embedding default
def test_qm_mm_mech_linkatom_default():
    label="qm_mm_mech_default"
    # Mechanical embedding QM/MM
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
    pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None)
    theory = QMMMTheory(qm_theory=qm, mm_theory=omm, qmatoms=region1, embedding="mech", fragment=frag,
                    qm_charge=-2, qm_mult=11, unusualboundary=True)
    result = Singlepoint(theory=theory, fragment=frag, Grad=True, charge=-2, mult=11)
    transvar_check(result.gradient, label=label, tolerance=grad_tolerance)

# Electrostatic embedding default
def test_qm_mm_elstat_linkatom_default():
    label="qm_mm_elstat_default"
    # Electrostatic embedding QM/MM
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
    pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None)
    theory = QMMMTheory(qm_theory=qm, mm_theory=omm, qmatoms=region1, embedding="elstat", fragment=frag,
                    qm_charge=-2, qm_mult=11, unusualboundary=True)
    result = Singlepoint(theory=theory, fragment=frag, Grad=True, charge=-2, mult=11)
    transvar_check(result.gradient, label=label, tolerance=grad_tolerance)

# For elstat charge-shifting: Looping over linkatom-method and dipole-corr
def test_qm_mm_elstat_chargeshift_linkatom_loop():
    for loption in [("simple",True),("simple",False),
                    ("ratio",True),("ratio",False)]:

        label="qm_mm_elstat_chargeshift"+str(loption)
        # Electrostatic embedding QM/MM
        omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
        pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None)
        theory = QMMMTheory(qm_theory=qm, mm_theory=omm, qmatoms=region1, embedding="elstat", fragment=frag,
                    qm_charge=-2, qm_mult=11, unusualboundary=True, 
                    linkatom_method=loption[0], dipole_correction=loption[1])
        result = Singlepoint(theory=theory, fragment=frag, Grad=True, charge=-2, mult=11)
        transvar_check(result.gradient, label=label, tolerance=grad_tolerance)

# For elstat RCD: Looping over linkatom-method
def test_qm_mm_elstat_RCD_linkatom_loop():
    # Looping over linkatom-method, L-forceproj and dipole-corr
    for loption in ["simple", "ratio"]:
        label="qm_mm_elstat_rcd"+str(loption)
        # Electrostatic embedding QM/MM
        omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
        pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None)
        theory = QMMMTheory(qm_theory=qm, mm_theory=omm, qmatoms=region1, embedding="elstat", fragment=frag,
                    qm_charge=-2, qm_mult=11, unusualboundary=True,
                    linkatom_method=loption,  chargeboundary_method="rcd")
        result = Singlepoint(theory=theory, fragment=frag, Grad=True, charge=-2, mult=11)
        transvar_check(result.gradient, label=label, tolerance=grad_tolerance)

def test_qm_mm_mech_linkatom_loop():
    # Looping over linkatom-method,and dipole-corr
    for loption in [("simple",True),("simple",False),
                    ("ratio",True),("ratio", False)]:
        label="qm_mm_mech_"+str(loption)
        # Mechanical embedding QM/MM
        omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", f"{ashpath}/tests/extra_files/ferredoxin.xml"],
        pdbfile=pdbfile, periodic=True, platform="cpu", rigidwater=False, autoconstraints=None)
        theory = QMMMTheory(qm_theory=qm, mm_theory=omm, qmatoms=region1, embedding="mech", fragment=frag,
                    qm_charge=-2, qm_mult=11, unusualboundary=True,
                    linkatom_method=loption[0], dipole_correction=loption[1])
        result = Singlepoint(theory=theory, fragment=frag, Grad=True, charge=-2, mult=11)
        transvar_check(result.gradient, label=label, tolerance=grad_tolerance)

test_qm_mm_mech_linkatom_default()
test_qm_mm_elstat_linkatom_default()
test_qm_mm_elstat_chargeshift_linkatom_loop()
test_qm_mm_elstat_RCD_linkatom_loop()
test_qm_mm_mech_linkatom_loop()
