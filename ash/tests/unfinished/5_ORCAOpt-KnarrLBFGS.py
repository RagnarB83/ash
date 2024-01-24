from ash import *
import sys
#Define global system settings ( scale, tol and conndepth keywords for connectivity)

#ORCA
orcadir='/opt/orca_4.2.1'
orcasimpleinput="! BP86 def2-SVP Grid5 Finalgrid6 tightscf"
orcablocks="%scf maxiter 200 end"

fragcoords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""

#Add coordinates to fragment
HF_frag=Fragment(coordsstring=fragcoords)

ORCAcalc = ORCATheory(orcadir=orcadir, fragment=HF_frag, charge=0, mult=1, 
                                orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

#Basic Cartesian optimization with KNARR-LBFGS
SimpleOpt(fragment=HF_frag, theory=ORCAcalc, optimizer='KNARR-LBFGS')

ORCAcalc.cleanup()
sys.exit(0)
