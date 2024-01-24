from ash import *

numcores=4
actualcores=4

frag = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

pyscfcalc = PySCFTheory(basis="cc-pVDZ", numcores=actualcores, scf_type='UHF', CC=True, CCmethod="CCSD(T)",
    CC_density=True, conv_tol=1e-9,memory=10000)
result = Singlepoint(fragment=frag, theory=pyscfcalc)
