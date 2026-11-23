from functions import transvar_check, numgrad_check
from ash import *

tolerance=1e-5

#QM/MM tests with xtb
# no covalent boundary

# H2O...MeOH fragment defined. Reading XYZ file
frag = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/h2o_MeOH.xyz")
region1=[3,4,5,6,7,8]
region2=[0,1,2]
hl = xTBTheory(xtbmethod="GFN2")
omm = OpenMMTheory(xmlfiles=[f"{ashpath}/tests/extra_files/MeOH_H2O-sigma.xml"], pdbfile=f"{ashpath}/tests/pdbfiles/h2o_MeOH.pdb", autoconstraints=None, rigidwater=False)

# Mech1
def test_qm_mm_nocovalboundary_mech1():
    label="QM/MM-nocovalboundary_mech1"
    theory = QMMMTheory(fragment=frag, qm_theory=hl, mm_theory=omm, qmatoms=region1,
                            embedding='mech', qm_charge=0, qm_mult=1)
    result= Singlepoint(theory=theory, fragment=frag, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)
# Mech2: swapping QM-region
def test_qm_mm_nocovalboundary_mech2():
    label="QM/MM-nocovalboundary_mech2"
    theory = QMMMTheory(fragment=frag, qm_theory=hl, mm_theory=omm, qmatoms=region2,
                            embedding='mech', qm_charge=0, qm_mult=1)
    result= Singlepoint(theory=theory, fragment=frag, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)
# Elstat
def test_qm_mm_nocovalboundary_elstat():
    label="QM/MM-nocovalboundary_elstat"
    theory = QMMMTheory(fragment=frag, qm_theory=hl, mm_theory=omm, qmatoms=region1,
                            embedding='elstat', qm_charge=0, qm_mult=1)
    result= Singlepoint(theory=theory, fragment=frag, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)
# Elstat2
def test_qm_mm_nocovalboundary_elstat2():
    label="QM/MM-nocovalboundary_elstat2"
    theory = QMMMTheory(fragment=frag, qm_theory=hl, mm_theory=omm, qmatoms=region2,
                            embedding='elstat', qm_charge=0, qm_mult=1)
    result= Singlepoint(theory=theory, fragment=frag, Grad=True)
    transvar_check(result.gradient, label=label, tolerance=tolerance)
    numgrad_check(frag, theory, TOL=tolerance)

def test_qm_mm_nocovalboundary_elstat_rcd():
    """RCD boundary correction should be a no-op without a covalent boundary."""
    theory = QMMMTheory(fragment=frag, qm_theory=hl, mm_theory=omm, qmatoms=region1,
                        embedding='elstat', qm_charge=0, qm_mult=1,
                        chargeboundary_method='rcd')
    result = Singlepoint(theory=theory, fragment=frag, Grad=True)
    assert result.gradient.shape == frag.coords.shape
