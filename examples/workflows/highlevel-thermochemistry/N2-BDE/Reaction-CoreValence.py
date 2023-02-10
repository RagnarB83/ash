from ash import *

#Script to calculate reaction energy of a simple molecule with CCSD(T) and multiple basis sets, CCSD(T)-F12 and CCSD(T)/CBS extrapolations
# With Core-Valence-Scalar-Relativistic Correction

#Define molecular fragments from XYZ-files or other
N2=Fragment(xyzfile='n2.xyz', charge=0, mult=1, label='N2')
N=Fragment(atom='N', charge=0, mult=4, label='N')

#Create a list of fragments and define the stoichiometry
specieslist=[N2, N]
stoichiometry=[-1,2]
reactionlabel='N2_BDE'

elements_involved=['N']

#Define theory
cc = CC_CBS_Theory(elements=elements_involved, cardinals = [5,6], basisfamily="cc", CVSR=True, CVbasis="W1-mtsmall", FCI=True)

#Store results
energies=[]
corecorrenergies=[]
fcicorrenergies=[]
ccsdcorrenergies=[]
ccsdtenergies=[]
for species in specieslist:
    result = Singlepoint(theory=cc, fragment=species)
    e = result.energy
    #Grab energy components
    energycomponents=cc.energy_components
    core_corr_energy=energycomponents["E_corecorr_and_SR"]
    fcicorr_energy=energycomponents["E_FCIcorrection"]
    corrccsd_energy=energycomponents["E_corrCCSD_CBS"]
    ccsdt_cbs_energy=energycomponents["E_CC_CBS"]
    energies.append(e)
    ccsdtenergies.append(ccsdt_cbs_energy)
    corecorrenergies.append(core_corr_energy)
    fcicorrenergies.append(fcicorr_energy)
    ccsdcorrenergies.append(corrccsd_energy)

#Calculate reaction energies of both total energies and separate contributions
reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit='kcal/mol', label='Total')
reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=ccsdtenergies, unit='kcal/mol', label='CCSD(T)/CBS')
reaction_energy_CVSR, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=corecorrenergies, unit='kcal/mol', label='CVSR')
reaction_energy_CVSR, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=fcicorrenergies, unit='kcal/mol', label='FCIcorr')
reaction_energy_CVSR, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=ccsdcorrenergies, unit='kcal/mol', label='CCSDcorr')
