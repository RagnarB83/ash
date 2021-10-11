from ash import *

#
numcores=4

N2=Fragment(xyzfile="/Users/bjornsson/ownCloud/ASH-tests/testsuite/n2.xyz", charge=0, mult=1)
H2=Fragment(xyzfile="/Users/bjornsson/ownCloud/ASH-tests/testsuite/h2.xyz", charge=0, mult=1)
NH3=Fragment(xyzfile="/Users/bjornsson/ownCloud/ASH-tests/testsuite/nh3.xyz", charge=0, mult=1)

##Defining reaction##
# List of species from reactant to product
specieslist=[N2, H2, NH3] #Use same order as stoichiometry

#Equation stoichiometry : negative integer for reactant, positive integer for product
# Example: N2 + 3H2 -> 2NH3  reaction should be:  [1,3,-2]
stoichiometry=[-1, -3, 2] #Use same order as specieslist
##

#Defining theory for Opt+Freq step in thermochemprotocol
simpleinput="! B3LYP D3BJ def2-TZVP TightSCF Grid5 Finalgrid6"
blockinput="""
%scf maxiter 200 end
"""
ORCAobject = ORCATheory(orcasimpleinput=simpleinput, orcablocks=blockinput, numcores=numcores)

#Thermochemistry protocol
thermochemprotocol_reaction(Opt_theory=ORCAobject , SP_theory=module_highlevel_workflows.DLPNO_W1theory, fraglist=specieslist, stoichiometry=stoichiometry, orcadir=orcadir, numcores=numcores)
