from ash import *

#Script to calculate Bond dissociation energy of a simple molecule
#using CCSD(T) with cc and def2 basis sets, CCSD(T)-F12 and CCSD(T)/CBS extrapolations
#Slightly verbose

#Define molecular fragments from XYZ-files or other
N2=Fragment(xyzfile='n2.xyz', charge=0, mult=1, label='N2')
N=Fragment(atom='N', charge=0, mult=4, label='N')

#Combined list of all elements involved in the species of the reaction
elements_involved=['N']

#Create a list of fragments and define the stoichiometry
specieslist=[N2, N]
stoichiometry=[-1,2]
reactionlabel='BDE'


###################################################
# Do single-basis CCSD(T) calculations: def2 family 
###################################################
print("Now running through single-basis CCSD(T) calculations with def2 family")

#Storing the results in a ProjectResults object
CCSDT_def2_bases_proj = ProjectResults('CCSDT_def2_bases')
CCSDT_def2_bases_proj.energy_dict={}
CCSDT_def2_bases_proj.reaction_energy_dict={}
CCSDT_def2_bases_proj.reaction_energy_list=[]

#Dict to store energies of species. Uses fragment label that has to be defined above for each
CCSDT_def2_bases_proj.species_energies_dict={}
for species in specieslist:
    CCSDT_def2_bases_proj.species_energies_dict[species.label]=[]


#Define basis-sets and labels
method='CCSD(T)'
#CCSDT_def2_bases_proj.basis_sets=['def2-SVP','def2-TZVPP','def2-QZVPP']
CCSDT_def2_bases_proj.cardinals=[2,3,4]
CCSDT_def2_bases_proj.labels=[]

#For loop that iterates over basis sets
for cardinal in CCSDT_def2_bases_proj.cardinals:
    label=cardinal
    CCSDT_def2_bases_proj.labels.append(label)
    #Define theory level
    #cc = ORCATheory(orcasimpleinput="! {} {} verytightscf".format(method,basis), orcablocks=blocks)
    cc = CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="def2")
    #Single-point calcs on all fragments
    energies=[]
    for species in specieslist:
        result = Singlepoint(theory=cc, fragment=species)
        e = result.energy
        cc.cleanup()
        energies.append(e)
        #Store energy for each species
        CCSDT_def2_bases_proj.species_energies_dict[species.label].append(e)

    #Storing total energies of all species in dict
    CCSDT_def2_bases_proj.energy_dict[label]=energies

    #Store reaction energy in dict
    reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit='kcal/mol', label=reactionlabel, silent=False)
    CCSDT_def2_bases_proj.reaction_energy_dict[label]=reaction_energy
    CCSDT_def2_bases_proj.reaction_energy_list.append(reaction_energy)
    #Resetting energies
    energies=[]

#Print results
CCSDT_def2_bases_proj.printall()

###################################################
# Do single-basis CCSD(T) calculations: cc family
###################################################
print("Now running through single-basis CCSD(T) calculations with cc family")

#Storing the results in a ProjectResults object
CCSDT_cc_bases_proj = ProjectResults('CCSDT_cc_bases')
CCSDT_cc_bases_proj.energy_dict={}
CCSDT_cc_bases_proj.reaction_energy_dict={}
CCSDT_cc_bases_proj.reaction_energy_list=[]

#Dict to store energies of species. Uses fragment label that has to be defined above for each
CCSDT_cc_bases_proj.species_energies_dict={}
for species in specieslist:
    CCSDT_cc_bases_proj.species_energies_dict[species.label]=[]


#Define basis-sets and labels
method='CCSD(T)'
#CCSDT_cc_bases_proj.basis_sets=['cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ', 'cc-pV5Z']
CCSDT_cc_bases_proj.cardinals=[2,3,4,5,6]
CCSDT_cc_bases_proj.labels=[]

#For loop that iterates over basis sets
for cardinal in CCSDT_cc_bases_proj.cardinals:
    label=cardinal
    CCSDT_cc_bases_proj.labels.append(label)
    #Define theory level
    #cc = ORCATheory(orcasimpleinput="! {} {} verytightscf".format(method,basis), orcablocks=blocks)
    cc = CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="cc")
    #Single-point calcs on all fragments
    energies=[]
    for species in specieslist:
        result = Singlepoint(theory=cc, fragment=species)
        e = result.energy
        cc.cleanup()
        energies.append(e)
        #Store energy for each species
        CCSDT_cc_bases_proj.species_energies_dict[species.label].append(e)

    #Storing total energies of all species in dict
    CCSDT_cc_bases_proj.energy_dict[label]=energies

    #Store reaction energy in dict
    reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit='kcal/mol', label=reactionlabel, silent=False)
    CCSDT_cc_bases_proj.reaction_energy_dict[label]=reaction_energy
    CCSDT_cc_bases_proj.reaction_energy_list.append(reaction_energy)
    #Resetting energies
    energies=[]

#Print results
CCSDT_cc_bases_proj.printall()


###########################################
# Do single-basis CCSD(T)-F12 calculations
###########################################
print("Now running through single-basis CCSD(T)-F12 calculations")

#Storing the results in a ProjectResults object
CCSDTF12_bases_proj = ProjectResults('CCSDTF12_bases')
CCSDTF12_bases_proj.energy_dict={}
CCSDTF12_bases_proj.reaction_energy_dict={}
CCSDTF12_bases_proj.reaction_energy_list=[]

#Dict to store energies of species. Uses fragment label that has to be defined above for each
CCSDTF12_bases_proj.species_energies_dict={}
for species in specieslist:
    CCSDTF12_bases_proj.species_energies_dict[species.label]=[]


#Define basis-sets and labels
method='CCSD(T)-F12'
CCSDTF12_bases_proj.cardinals=[2,3,4]
CCSDTF12_bases_proj.labels=[]

#For loop that iterates over basis sets
for cardinal in CCSDTF12_bases_proj.cardinals:
    label=str(cardinal)
    CCSDTF12_bases_proj.labels.append(label)
    #Define theory level
    cc = CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="cc-f12", F12=True)
    #Single-point calcs on all fragments
    energies=[]
    for species in specieslist:
        result = Singlepoint(theory=cc, fragment=species)
        e = result.energy
        cc.cleanup()
        energies.append(e)
        #Store energy for each species
        CCSDTF12_bases_proj.species_energies_dict[species.label].append(e)

    #Storing total energies of all species in dict
    CCSDTF12_bases_proj.energy_dict[label]=energies

    #Store reaction energy in dict
    reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit='kcal/mol', label=reactionlabel, silent=False)
    #CCSDT_cc_reaction_energy_dict[label]=reaction_energy
    CCSDTF12_bases_proj.reaction_energy_dict[label]=reaction_energy
    CCSDTF12_bases_proj.reaction_energy_list.append(reaction_energy)
    #Resetting energies
    energies=[]

print()
#Print results
CCSDTF12_bases_proj.printall()

#################################################
# Do CCSD(T)/CBS extrapolations with cc family
#################################################
print("Now running through extrapolation CCSD(T) calculations")

#Storing the results in a ProjectResults object
CCSDTextrap_proj = ProjectResults('CCSDT_extrap')
CCSDTextrap_proj.energy_dict={}
CCSDTextrap_proj.reaction_energy_dict={}
CCSDTextrap_proj.reaction_energy_list=[]

#Dict to store energies of species. Uses fragment label that has to be defined above for each
CCSDTextrap_proj.species_energies_dict={}
for species in specieslist:
    CCSDTextrap_proj.species_energies_dict[species.label]=[]


#Define basis-sets and labels
CCSDTextrap_proj.cardinals=[[2,3],[3,4],[4,5],[5,6]]
CCSDTextrap_proj.labels=[]

#For loop that iterates over basis sets
for cardinals in CCSDTextrap_proj.cardinals:
    label=str(cardinal)
    CCSDTextrap_proj.labels.append(label)
    #Define theory level
    cc = CC_CBS_Theory(elements=elements_involved, cardinals = cardinals, basisfamily="cc")
    #Single-point calcs on all fragments
    energies=[]
    for species in specieslist:
        result = Singlepoint(theory=cc, fragment=species)
        e = result.energy
        cc.cleanup()
        energies.append(e)
        #Store energy for each species
        CCSDTextrap_proj.species_energies_dict[species.label].append(e)

    #Storing total energies of all species in dict
    CCSDTextrap_proj.energy_dict[label]=energies

    #Store reaction energy in dict
    reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit='kcal/mol', label=reactionlabel, silent=False)
    #CCSDT_cc_reaction_energy_dict[label]=reaction_energy
    CCSDTextrap_proj.reaction_energy_dict[label]=reaction_energy
    CCSDTextrap_proj.reaction_energy_list.append(reaction_energy)
    #Resetting energies
    energies=[]

print()
#Print results
CCSDTextrap_proj.printall()


###################################################
# Do CCSD(T)/CBS extrapolations with def2 family
###################################################
print("Now running through extrapolation CCSD(T) calculations")

#Storing the results in a ProjectResults object
CCSDTextrapdef2_proj = ProjectResults('CCSDT_extrapdef2')
CCSDTextrapdef2_proj.energy_dict={}
CCSDTextrapdef2_proj.reaction_energy_dict={}
CCSDTextrapdef2_proj.reaction_energy_list=[]

#Dict to store energies of species. Uses fragment label that has to be defined above for each
CCSDTextrapdef2_proj.species_energies_dict={}
for species in specieslist:
    CCSDTextrapdef2_proj.species_energies_dict[species.label]=[]


#Define basis-sets and labels
CCSDTextrapdef2_proj.cardinals=[[2,3],[3,4]]
CCSDTextrapdef2_proj.labels=[]

#For loop that iterates over basis sets
for cardinals in CCSDTextrapdef2_proj.cardinals:
    label=str(cardinal)
    CCSDTextrapdef2_proj.labels.append(label)
    #Define theory level
    cc = CC_CBS_Theory(elements=elements_involved, cardinals = cardinals, basisfamily="def2")
    #Single-point calcs on all fragments
    energies=[]
    for species in specieslist:
        result = Singlepoint(theory=cc, fragment=species)
        e = result.energy
        cc.cleanup()
        energies.append(e)
        #Store energy for each species
        CCSDTextrapdef2_proj.species_energies_dict[species.label].append(e)

    #Storing total energies of all species in dict
    CCSDTextrapdef2_proj.energy_dict[label]=energies

    #Store reaction energy in dict
    reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit='kcal/mol', label=reactionlabel, silent=False)
    #CCSDT_cc_reaction_energy_dict[label]=reaction_energy
    CCSDTextrapdef2_proj.reaction_energy_dict[label]=reaction_energy
    CCSDTextrapdef2_proj.reaction_energy_list.append(reaction_energy)
    #Resetting energies
    energies=[]

print()
#Print results
CCSDTextrapdef2_proj.printall()



################
#Plotting
###############

#Energy plot for each species:
for species in specieslist:
    specieslabel=species.label
    eplot = ASH_plot('{} energy plot'.format(specieslabel), num_subplots=1, x_axislabel="Cardinal", y_axislabel='Energy (Eh)', title='{} Energy'.format(specieslabel))
    eplot.addseries(0, x_list=CCSDT_cc_bases_proj.cardinals, y_list=CCSDT_cc_bases_proj.species_energies_dict[specieslabel], label='cc-pVnZ', color='blue')
    eplot.addseries(0, x_list=CCSDT_def2_bases_proj.cardinals, y_list=CCSDT_def2_bases_proj.species_energies_dict[specieslabel], label='def2-nVP', color='red')
    eplot.addseries(0, x_list=CCSDTF12_bases_proj.cardinals, y_list=CCSDTF12_bases_proj.species_energies_dict[specieslabel], label='cc-pVnZ-F12', color='purple')
    eplot.addseries(0, x_list=[2.5], y_list=CCSDTextrap_proj.species_energies_dict[specieslabel][0], label='CBS-cc-23', line=False,  marker='x', color='gray')
    eplot.addseries(0, x_list=[3.5], y_list=CCSDTextrap_proj.species_energies_dict[specieslabel][1], label='CBS-cc-34', line=False, marker='x', color='green')
    eplot.addseries(0, x_list=[4.5], y_list=CCSDTextrap_proj.species_energies_dict[specieslabel][2], label='CBS-cc-45', line=False, marker='x', color='black')
    eplot.addseries(0, x_list=[5.5], y_list=CCSDTextrap_proj.species_energies_dict[specieslabel][3], label='CBS-cc-56', line=False, marker='x', color='orange')
    eplot.addseries(0, x_list=[2.5], y_list=CCSDTextrapdef2_proj.species_energies_dict[specieslabel][0], label='CBS-def2-23', line=False,  marker='x', color='cyan')
    eplot.addseries(0, x_list=[3.5], y_list=CCSDTextrapdef2_proj.species_energies_dict[specieslabel][1], label='CBS-def2-34', line=False, marker='x', color='pink')
    eplot.savefig('{}_Energy'.format(specieslabel))

#Reaction energy plot
reactionenergyplot = ASH_plot('{}'.format(reactionlabel), num_subplots=1, x_axislabel="Cardinal", y_axislabel='Energy ({})'.format(reactionlabel), title='{}'.format(reactionlabel))
reactionenergyplot.addseries(0, x_list = CCSDT_cc_bases_proj.cardinals, y_list=CCSDT_cc_bases_proj.reaction_energy_list, label='cc-pVnZ', color='blue')
reactionenergyplot.addseries(0, x_list = CCSDT_def2_bases_proj.cardinals, y_list=CCSDT_def2_bases_proj.reaction_energy_list, label='def2-nVP', color='red')
reactionenergyplot.addseries(0, x_list = CCSDTF12_bases_proj.cardinals, y_list=CCSDTF12_bases_proj.reaction_energy_list, label='cc-pVnZ-F12', color='purple')
reactionenergyplot.addseries(0, x_list=[2.5], y_list=CCSDTextrap_proj.reaction_energy_list[0], label='CBS-cc-23', line=False,  marker='x', color='gray')
reactionenergyplot.addseries(0, x_list=[3.5], y_list=CCSDTextrap_proj.reaction_energy_list[1], label='CBS-cc-34', line=False, marker='x', color='green')
reactionenergyplot.addseries(0, x_list=[4.5], y_list=CCSDTextrap_proj.reaction_energy_list[2], label='CBS-cc-45', line=False, marker='x', color='black')
reactionenergyplot.addseries(0, x_list=[5.5], y_list=CCSDTextrap_proj.reaction_energy_list[3], label='CBS-cc-56', line=False, marker='x', color='orange')
reactionenergyplot.addseries(0, x_list=[2.5], y_list=CCSDTextrapdef2_proj.reaction_energy_list[0], label='CBS-def2-23', line=False, marker='x', color='cyan')
reactionenergyplot.addseries(0, x_list=[3.5], y_list=CCSDTextrapdef2_proj.reaction_energy_list[1], label='CBS-def2-34', line=False, marker='x', color='pink')
reactionenergyplot.savefig('Reaction energy')
