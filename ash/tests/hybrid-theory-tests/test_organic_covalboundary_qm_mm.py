from functions import transvar_check, numgrad_check
from ash import *

tolerance=1e-5

# Hybrid-theory tests with xtb.
# Testing QMMMTheory with a covalent boundary

# educt system (from BHDIV10 in GMTKN55). Enol reactant for a dehydration reaction.
frag = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/educt.xyz", charge=0, mult=1)
pdbfile=f"{ashpath}/tests/pdbfiles/educt.pdb"
xmlfile=f"{ashpath}/tests/extra_files/gaff_educt.xml"

region1_atoms=[2,3,6,7,8,9,10,11,12,13]
hl = xTBTheory(xtbmethod="GFN2")
ll = xTBTheory(xtbmethod="GFN1")

# Mechanical embedding QM/MM: looping over linkatom methods
def test_qm_mm_coval_mech():
    label="QMMM-covalboundary_mech"

    for lsetting in ["simple", "ratio"]:
        print("Linkatom method:", lsetting)
        label=str(lsetting)
        omm = OpenMMTheory(pdbfile=pdbfile, xmlfiles=[xmlfile],  autoconstraints=None, rigidwater=False)
        theory = QMMMTheory(qm_theory=hl, mm_theory=omm, qmatoms=region1_atoms, fragment=frag, embedding="mech",
                            linkatom_method=lsetting)

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, TOL=tolerance)

# Electrostatic embedding QM/MM: looping over linkatom methods
def test_qm_mm_coval_elstat():
    label="QMMM-covalboundary_elstat"

    for lsetting in [("simple",True), ("simple",False), ("ratio",True), ("ratio",False)]:
        print("Linkatom method:", lsetting)
        label=str(lsetting)
        omm = OpenMMTheory(pdbfile=pdbfile, xmlfiles=[xmlfile],  autoconstraints=None, rigidwater=False)
        theory = QMMMTheory(qm_theory=hl, mm_theory=omm, qmatoms=region1_atoms, fragment=frag, embedding="elstat",
                            linkatom_method=lsetting[0], dipole_correction=lsetting[1])

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, TOL=tolerance)

# Electrostatic embedding QM/MM with RCD: looping over linkatom methods
def test_qm_mm_coval_elstat_rcd():
    label="QMMM-covalboundary_elstat_rcd"

    for lsetting in ["simple", "ratio"]:
        print("Linkatom method:", lsetting)
        label=str(lsetting)
        omm = OpenMMTheory(pdbfile=pdbfile, xmlfiles=[xmlfile],  autoconstraints=None, rigidwater=False)
        theory = QMMMTheory(qm_theory=hl, mm_theory=omm, qmatoms=region1_atoms, fragment=frag, embedding="elstat",
                            linkatom_method=lsetting, chargeboundary_method="rcd")

        result = Singlepoint(theory=theory, fragment=frag, Grad=True)
        transvar_check(result.gradient, label=label, tolerance=tolerance)
        numgrad_check(frag, theory, TOL=tolerance)
