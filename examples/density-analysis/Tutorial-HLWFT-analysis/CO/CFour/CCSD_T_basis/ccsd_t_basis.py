from ash import *

numcores=10

frag = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

for basis in ["PVDZ", "PVTZ", "PVQZ", "PV5Z"]:
    cfouroptions = {
'CALC':'CCSD(T)',
'REF':'UHF',
'BASIS':basis,
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
    result = Singlepoint(fragment=frag, theory=cfourcalc)
    #Convert CFour Molden file,MOLDEN_NAT, to Multiwfn-compatible file
    convert_CFour_Molden_file("MOLDEN_NAT")
    os.rename("MOLDEN_NAT_new.molden", f"CCSD_T_{basis}_nat.molden")
    os.rename("cfourjob.out", f"CFOUR_CCSD_T_{basis}_cfourjob.out")
