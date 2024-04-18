from ash import *

numcores=8

CO = Fragment(diatomic="CO", bondlength=1.1294, charge=0, mult=1)

HF = ORCATheory(orcasimpleinput="! HF cc-pVDZ verytightscf", numcores=numcores, filename="HF")
r2SCAN = ORCATheory(orcasimpleinput="! r2SCAN cc-pVDZ verytightscf", numcores=numcores, filename="r2SCAN")
BLYP = ORCATheory(orcasimpleinput="! BLYP cc-pVDZ verytightscf", numcores=numcores, filename="BLYP")
B3LYP = ORCATheory(orcasimpleinput="! B3LYP cc-pVDZ verytightscf", numcores=numcores, filename="B3LYP")
BHLYP = ORCATheory(orcasimpleinput="! BHLYP cc-pVDZ verytightscf", numcores=numcores, filename="BHLYP")
wB97XV = ORCATheory(orcasimpleinput="! wB97X-V cc-pVDZ verytightscf", numcores=numcores, filename="wB97XV")
for theory in [HF,r2SCAN,BLYP,B3LYP,BHLYP,wB97XV]:
    #
    Singlepoint(theory=theory, fragment=CO)

    # Make Moldenfile
    mfile = make_molden_file_ORCA(theory.filename+'.gbw')
