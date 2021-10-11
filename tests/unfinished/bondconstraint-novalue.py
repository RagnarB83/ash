from ash import *
import sys
coordstring="""
Fe      -0.654024588      0.000000000     -1.593315220
S       -0.320054586      0.000000000     -3.249249816
S       -0.361800836      0.000000000     -0.173942710

"""
HF_frag=Fragment(coordsstring=coordstring)
#ORCA
orcadir='/opt/orca_4.2.1'
orcasimpleinput="! BP86 def2-SVP Grid5 Finalgrid6 tightscf"
orcablocks="%scf maxiter 200 end"
ORCAcalc = ORCATheory(orcadir=orcadir, charge=-1, mult=6,
                    orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

#Geometry optimization of the ORCA using geomeTRIC optimizer
geoconstraints = { 'bond' : [[0,1]]}
geomeTRICOptimizer(fragment=HF_frag, theory=ORCAcalc, coordsystem='tric', constraints=geoconstraints, convergence_setting='ORCA')

ORCAcalc.cleanup()
