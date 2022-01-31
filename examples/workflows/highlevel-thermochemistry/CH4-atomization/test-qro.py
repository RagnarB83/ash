from ash import *

numcores=1

#Define fragments
methane=Fragment(xyzfile="methane.xyz", charge=0, mult=1)
C=Fragment(atom="C", charge=0, mult=3)
H=Fragment(atom="H", charge=0, mult=2)
fragments=[methane,C,H] #Combining into a list
stoichiometry=[-1,1,4] #Defining stoichiometry of reaction, here atomization reaction

#Define Theories
DFTopt=ORCATheory(orcasimpleinput="!r2scan-3c", numcores=numcores)
HL=CC_CBS_Theory(elements=["C","H"], DLPNO=False, basisfamily="cc", cardinals=[3,4], CVSR=True, numcores=numcores, Openshellreference="QRO", atomicSOcorrection=True)

#RUn thermochemistry protocol: Opt+Freq using DFTOpt, final energy using HL theory
thermochemdict = thermochemprotocol_reaction(fraglist=fragments, stoichiometry=stoichiometry, Opt_theory=DFTopt, SP_theory=HL, numcores=numcores, memory=5000,
                   analyticHessian=True, temp=298.15, pressure=1.0)
print("thermochemdict:", thermochemdict)

#Grabbing atomization energy at 0K (with ZPVE) or 298 K.
TAE_0K=thermochemdict['deltaE_0']
print("TAE_0K:", TAE_0K)
TAE_298K=thermochemdict['deltaH']

#Calculate Enthalpy of formation from atomization energy
deltaH_form_0K = FormationEnthalpy(TAE_0K, fragments, stoichiometry, RT=False)
deltaH_form_298K = FormationEnthalpy(TAE_298K, fragments, stoichiometry, RT=True)


print("\n FINAL RESULTS")
print("="*50)
print("\n\nCalculated deltaH_form_0K:", deltaH_form_0K)
print("Calculated deltaH_form_298K:", deltaH_form_298K)
print("-"*50)
print("Experimental deltaH_form(0K): -15.908 kcal/mol")
print("Experimental deltaH_form(298K): -17.812 kcal/mol")

#Methane from AtCT
#deltaHf (0K) = -15.907504780114722 kcal/mol
#deltaHf(298.15K)= -17.812141491395792 kcal/mol

#TAE from Karton 2007 paper
#TAE_electronic = 420.26 kcal/mol
#ZPVE = 27.74 kcal/mol
#TAE at 0K (with ZPVE) = 392.52 kcal/mol
