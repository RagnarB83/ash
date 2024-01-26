from ash import *

numcores=4
#Define fragment
frag = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

cfouroptions = {
'CALC':'CCSD(T)',
'REF':'UHF',
'BASIS':'PVDZ',
'FROZEN_CORE':'ON',
'MEM_UNIT':'MB',
'PROP':'FIRST_ORDER',
'MEMORY':3100,
'SCF_CONV':8,
'LINEQ_CONV':10,
'SCF_MAXCYC':7000,
'SYMMETRY':'OFF',
}
cfourcalc = CFourTheory(cfouroptions=cfouroptions, numcores=numcores)

#Geometry optimization
Singlepoint(theory=cfourcalc, fragment=frag)

#Convert CFour Molden file,MOLDEN_NAT, to Multiwfn-compatible file
convert_CFour_Molden_file("MOLDEN_NAT")
