from ash import *
import sys
import interface_knarr


orcadir='/opt/orca_4.2.1'
orcasimpleinput="! HF-3c "
orcablocks="%scf maxiter 200 end"

reactstring="""
   C  -2.66064921   -0.44148342    0.02830018
   H  -2.26377685   -1.23173358    0.68710920
   H  -2.29485851   -0.62084858   -0.99570465
   H  -2.27350346    0.53131334    0.37379014
   F  -4.03235214   -0.44462811    0.05296388
"""
Reactant=Fragment(coordsstring=reactstring)

#Calculator object without frag
ORCAcalc = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
ORCAcalc.cleanup()
#Using geomeTRIC optimization
Optimizer(theory=ORCAcalc,fragment=Reactant)


#Numfreq
freqresult = NumFreq(Reactant, ORCAcalc, npoint=1, numcores=1, runmode='serial')

print("freqresult:", freqresult)

