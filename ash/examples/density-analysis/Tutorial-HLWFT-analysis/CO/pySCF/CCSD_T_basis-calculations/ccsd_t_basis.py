from ash import *

numcores=10
actualcores=10

frag = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

for basis in ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z"]:
    pyscfcalc = PySCFTheory(basis=basis, numcores=actualcores, scf_type='RHF', CC=True, CCmethod="CCSD(T)",
        CC_density=True, conv_tol=1e-9,memory=10000, noautostart=True)
    result = Singlepoint(fragment=frag, theory=pyscfcalc)
