#Interface to geomeTRIC program

from yggdrasill import *
from functions_optimization import *
import os
path="/Users/bjornssonsu/ownCloud/PyQMMM-project/Yggdrasill-testdir"
os.chdir(path)

#Define Yggdrasill frament and theory stuff
orcasimpleinput="! BP86 def2-SVP Grid5 Finalgrid6 tightscf"
orcablocks="%scf maxiter 200 end"
orcadir="/Applications/orca_4.2.1"
ORCASPcalculation = ORCATheory(orcadir=orcadir, charge=0, mult=1, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
#
xyzfile = "h2o_strained.xyz"
#xyzfile = "h2o.xyz"
molfrag = Fragment(xyzfile=xyzfile)

#Calling geomeTRICOptimizer, a wrapper function around geomeTRIC code
geomeTRICOptimizer(ORCASPcalculation,molfrag)