from ash import *

#Function to perform all single-basis, F12-basis and extrapolation calculations at the CCSD(T) level using def2 and cc family.

#Define molecular fragments from XYZ-files or other
N2=Fragment(xyzfile='n2.xyz', charge=0, mult=1, label='N2')
N=Fragment(atom='N', charge=0, mult=4, label='N')

#Create a list of fragments and define the stoichiometry
specieslist=[N2, N]
stoichiometry=[-1,2]
reactionlabel='N2_BDE'

# Call Reaction_Highlevel_Analysis
Reaction_Highlevel_Analysis(fraglist=specieslist, stoichiometry=stoichiometry, numcores=1, memory=7000, reactionlabel=reactionlabel,
                                def2_family=True, cc_family=True, F12_family=True, extrapolation=True, highest_cardinal=5 )
