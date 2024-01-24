from ash import *

numcores=4
actualcores=numcores

#Define fragment
frag = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)


#Defining MRCCTheory object
mrccinput="""
basis=cc-pVDZ
calc=CCSD
mem=30000MB
scftype=RHF
ccmaxit=150
cctol=5
dens=1
"""
MRCCcalc = MRCCTheory(mrccinput=mrccinput, numcores=actualcores)

#Run MRCC calculation
result=Singlepoint(theory=MRCCcalc,fragment=frag)

convert_MRCC_Molden_file(mrccoutputfile=f"{MRCCcalc.filename}.out", moldenfile="MOLDEN", mrccdensityfile="CCDENSITIES", printlevel=2)
