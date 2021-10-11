from ash import *

#
crestdir='/opt/crest'
orcadir='/opt/orca_4.2.1'
numcores=4

ethanol="""
C       -0.320953666     -0.445752596     -0.454992334
H       -1.209131378     -0.094637949     -0.996914852
H        0.567224046     -0.094637949     -0.996914852
C       -0.320953666      0.073221832      0.964592709
H       -0.320953666      1.161201707      0.972533201
H       -1.206061462     -0.282072664      1.494234215
H        0.564154130     -0.282072664      1.494234215
O       -0.320953666     -1.875567463     -0.397320556
H       -0.320953666     -2.171455619     -1.322139443
"""
frag=Fragment(coordsstring=ethanol, charge=0, mult=1)

#Defining MLTheory: DFT optimization
orcadir='/opt/orca_4.2.1'
MLsimpleinput="! B3LYP D3BJ def2-SVP TightSCF Grid5 Finalgrid6"
MLblockinput="""
%scf maxiter 200 end
"""
ML_B3LYP = ORCATheory(orcadir=orcadir, orcasimpleinput=MLsimpleinput, orcablocks=MLblockinput, numcores=numcores, charge=frag.charge, mult=frag.mult)
#Defining HLTheory: DLPNO-CCSD(T)/CBS
HLsimpleinput="! DLPNO-CCSD(T) Extrapolate(2/3,def2) def2-QZVPP/C TightSCF"
HLblockinput="""
%scf maxiter 200 end
"""
HL_CC = ORCATheory(orcadir=orcadir, orcasimpleinput=HLsimpleinput, orcablocks=HLblockinput, numcores=numcores, charge=frag.charge, mult=frag.mult)


confsampler_protocol(fragment=frag, crestdir=crestdir, xtbmethod='GFN2-xTB', MLtheory=ML_B3LYP, 
                         HLtheory=HL_CC, orcadir=orcadir, numcores=numcores, charge=frag.charge, mult=frag.mult)


