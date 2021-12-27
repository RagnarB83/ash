from ash import *

#Script to calculate Bond dissociation energy of a simple molecule
#CCSD(T)-CBS with multiple extrapolations of cc family

#Define molecular fragment from XYZ-file
N2=Fragment(xyzfile='n2.xyz', readchargemult=True)
#Define atomic species
N=Fragment(elems=['N'], coords=[[0.0,0.0,0.0]], charge=0, mult=4)

#Create a list of fragments and define the stoichiometry
specieslist=[N2, N]
stoichiometry=[-1,2]

#Dictionary to store the results
resultdict={}

#For loop that iterates over something: here the different cardinals in the basis-set extrapolation
for cardinals in [(2,3),(3,4),(4,5),(5,6)]:
    #Define theory level
    cc = CC_CBS_Theory(elements=["N"], cardinals = cardinals, basisfamily="cc", numcores=1)
    #Set charge/mult of theory for N2 and run singlepoint
    cc.charge=N2.charge; cc.mult=N2.mult
    e_n2 = Singlepoint(theory=cc, fragment=N2)
    #Set charge/mult of theory for N and run singlepoint
    cc.charge=N.charge; cc.mult=N.mult
    e_n = Singlepoint(theory=cc, fragment=N)

    #Store final energies of fragment in dictionary
    resultdict['cc'+str(cardinals)]=[e_n2,e_n]
    #Print BDE for each cardinal-set
    ReactionEnergy(stoichiometry=stoichiometry, label="cc_{}".format(cardinals), list_of_fragments=specieslist, list_of_energies=[e_n2,e_n])

print("\n\n")
#Print final results
for basisfamily,energies in resultdict.items():
    ReactionEnergy(stoichiometry=stoichiometry, label=basisfamily, list_of_fragments=specieslist, list_of_energies=energies)
