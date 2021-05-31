"""
Contains functions defining multi-step workflows

"""
import os
import subprocess as sp
import shutil
import math
import time
import constants
import ash
import dictionaries_lists
import module_coords
import interface_geometric
import interface_crest
from functions_general import BC,print_time_rel,print_line_with_mainheader,pygrep
import ash_header
import settings_ash
import module_highlevel_workflows
import functions_elstructure


def ReactionEnergy(stoichiometry=None, list_of_fragments=None, list_of_energies=None, unit='kcal/mol', label=None, reference=None):
    """Calculate reaction energy from list of energies (or energies from list of fragments) and stoichiometry

    Args:
        stoichiometry (list, optional): A list of integers, e.g. [-1,-1,1,1]. Defaults to None.
        list_of_fragments (list, optional): A list of ASH fragments . Defaults to None.
        list_of_energies ([type], optional): A list of total energies in hartrees. Defaults to None.
        unit (str, optional): Unit for relative energy. Defaults to 'kcal/mol'.
        label (string, optional): Optional label for energy. Defaults to None.
        reference (float, optional): Optional shift-parameter of energy Defaults to None.

    Returns:
        tuple : energy and error in chosen unit
    """
    conversionfactor = { 'kcal/mol' : 627.50946900, 'kcalpermol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
    if label is None:
        label=''
    #print(BC.OKBLUE,BC.BOLD, "ReactionEnergy function. Unit:", unit, BC.END)
    reactant_energy=0.0 #hartree
    product_energy=0.0 #hartree
    if stoichiometry is None:
        print("stoichiometry list is required")
        exit(1)



    #List of energies option
    if list_of_energies is not None:

        if len(list_of_energies) != len(stoichiometry):
            print("Number of energies not equal to number of stoichiometry values")
            print("Check ")

        #print("List of total energies provided (Eh units assumed).")
        #print("list_of_energies:", list_of_energies)
        #print("stoichiometry:", stoichiometry)
        for i,stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy=reactant_energy+list_of_energies[i]*abs(stoich)
            if stoich > 0:
                product_energy=product_energy+list_of_energies[i]*abs(stoich)
        reaction_energy=(product_energy-reactant_energy)*conversionfactor[unit]
        if reference is None:
            error=None
            print(BC.BOLD, "Reaction_energy({}): {} {}".format(label,BC.OKGREEN,reaction_energy, unit), BC.END)
        else:
            error=reaction_energy-reference
            print(BC.BOLD, "Reaction_energy({}): {} {} {} (Error: {}) {}".format(label,BC.OKGREEN,reaction_energy, unit, error, BC.END))
    else:
        print("No list of total energies provided. Using internal energy of each fragment instead.")
        print("")
        for i,stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy=reactant_energy+list_of_fragments[i].energy*abs(stoich)
            if stoich > 0:
                product_energy=product_energy+list_of_fragments[i].energy*abs(stoich)
        reaction_energy=(product_energy-reactant_energy)*conversionfactor[unit]
        if reference is None:
            error=None
            print(BC.BOLD, "Reaction_energy({}): {} {}".format(label,BC.OKGREEN,reaction_energy, unit), BC.END)
        else:
            error=reaction_energy-reference
            print(BC.BOLD, "Reaction_energy({}): {} {} {} (Error: {})".format(label,BC.OKGREEN,reaction_energy, unit, error, BC.END))
    return reaction_energy, error





#Provide crest/xtb info, MLtheory object (e.g. ORCA), HLtheory object (e.g. ORCA)
def confsampler_protocol(fragment=None, crestdir=None, xtbmethod='GFN2-xTB', MLtheory=None, 
                         HLtheory=None, orcadir=None, numcores=1, charge=None, mult=None):
    """[summary]

    Args:
        fragment (ASH fragment, optional): An ASH fragment. Defaults to None.
        crestdir (str, optional): Path to Crest. Defaults to None.
        xtbmethod (str, optional): The xTB method string. Defaults to 'GFN2-xTB'.
        MLtheory (ASH theory object, optional): Theoryobject for medium-level theory. Defaults to None.
        HLtheory (ASH theory object, optional): Theoryobject for high-level theory. Defaults to None.
        orcadir (str, optional): Path to ORCA. Defaults to None.
        numcores (int, optional): Number of cores. Defaults to 1.
        charge (int, optional): Charge. Defaults to None.
        mult (in, optional): Spin multiplicity. Defaults to None.
    """
    module_init_time=time.time()
    print("="*50)
    print("CONFSAMPLER FUNCTION")
    print("="*50)
    
    #1. Calling crest
    #call_crest(fragment=molecule, xtbmethod='GFN2-xTB', crestdir=crestdir, charge=charge, mult=mult, solvent='H2O', energywindow=6 )
    interface_crest.call_crest(fragment=fragment, xtbmethod=xtbmethod, crestdir=crestdir, charge=charge, mult=mult, numcores=numcores)

    #2. Grab low-lying conformers from crest_conformers.xyz as list of ASH fragments.
    list_conformer_frags, xtb_energies = interface_crest.get_crest_conformers()

    print("list_conformer_frags:", list_conformer_frags)
    print("")
    print("Crest Conformer Searches done. Found {} conformers".format(len(xtb_energies)))
    print("xTB energies: ", xtb_energies)

    #3. Run ML (e.g. DFT) geometry optimizations for each crest-conformer

    ML_energies=[]
    print("")
    for index,conformer in enumerate(list_conformer_frags):
        print("")
        print("Performing ML Geometry Optimization for Conformer ", index)
        interface_geometric.geomeTRICOptimizer(fragment=conformer, theory=MLtheory, coordsystem='tric')
        ML_energies.append(conformer.energy)
        #Saving ASH fragment and XYZ file for each ML-optimized conformer
        os.rename('Fragment-optimized.ygg', 'Conformer{}_ML.ygg'.format(index))
        os.rename('Fragment-optimized.xyz', 'Conformer{}_ML.xyz'.format(index))

    print("")
    print("ML Geometry Optimization done")
    print("ML_energies: ", ML_energies)

    #4.Run high-level thery. Provide HLtheory object (typically ORCATheory)
    HL_energies=[]
    for index,conformer in enumerate(list_conformer_frags):
        print("")
        print("Performing High-level calculation for ML-optimized Conformer ", index)
        HLenergy = ash.Singlepoint(theory=HLtheory, fragment=conformer)
        HL_energies.append(HLenergy)


    print("")
    print("=================")
    print("FINAL RESULTS")
    print("=================")

    #Printing total energies
    print("")
    print(" Conformer   xTB-energy    ML-energy    HL-energy (Eh)")
    print("----------------------------------------------------------------")

    min_xtbenergy=min(xtb_energies)
    min_MLenergy=min(ML_energies)
    min_HLenergy=min(HL_energies)

    for index,(xtb_en,ML_en,HL_en) in enumerate(zip(xtb_energies,ML_energies, HL_energies)):
        print("{:10} {:13.10f} {:13.10f} {:13.10f}".format(index,xtb_en, ML_en, HL_en))

    print("")
    #Printing relative energies
    min_xtbenergy=min(xtb_energies)
    min_MLenergy=min(ML_energies)
    min_HLenergy=min(HL_energies)
    harkcal = 627.50946900
    print(" Conformer   xTB-energy    ML-energy    HL-energy (kcal/mol)")
    print("----------------------------------------------------------------")
    for index,(xtb_en,ML_en,HL_en) in enumerate(zip(xtb_energies,ML_energies, HL_energies)):
        rel_xtb=(xtb_en-min_xtbenergy)*harkcal
        rel_ML=(ML_en-min_MLenergy)*harkcal
        rel_HL=(HL_en-min_HLenergy)*harkcal
        print("{:10} {:13.10f} {:13.10f} {:13.10f}".format(index,rel_xtb, rel_ML, rel_HL))

    print("")
    print("Confsamplerprotocol done!")
    print_time_rel(module_init_time, modulename='Confsamplerprotocol', moduleindex=0)
    

# opt+freq+HL protocol for single species
def thermochemprotocol_single(fragment=None, Opt_theory=None, SP_theory=None, orcadir=None, numcores=None, memory=5000,
                       workflow_args=None, analyticHessian=True, temp=298.15, pressure=1.0):
    module_init_time=time.time()
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL (single-species)-------------", BC.END)
    if fragment.charge == None:
        print("1st. Fragment: {}".format(fragment.__dict__))
        print("No charge/mult information present in fragment. Each fragment in provided fraglist must have charge/mult information defined.")
        print("Example:")
        print("fragment.charge= 0; fragment.mult=1")
        print("Exiting...")
        exit()
    #DFT Opt+Freq  and Single-point High-level workflow
    #Only Opt+Freq for molecules, not atoms
    print("-------------------------------------------------------------------------")
    print("THERMOCHEM PROTOCOL-single: Step 1. Geometry optimization")
    print("-------------------------------------------------------------------------")
    if fragment.numatoms != 1:
        #DFT-opt
        #Adding charge and mult to theory object, taken from each fragment object
        Opt_theory.charge = fragment.charge
        Opt_theory.mult = fragment.mult
        interface_geometric.geomeTRICOptimizer(theory=Opt_theory,fragment=fragment)
        print("-------------------------------------------------------------------------")
        print("THERMOCHEM PROTOCOL-single: Step 2. Frequency calculation")
        print("-------------------------------------------------------------------------")
        #DFT-FREQ
        if analyticHessian == True:
            thermochem = ash.AnFreq(fragment=fragment, theory=Opt_theory, numcores=numcores)                
        else:
            thermochem = ash.NumFreq(fragment=fragment, theory=Opt_theory, npoint=2, runmode='serial')
    else:
        #Setting thermoproperties for atom
        thermochem = thermochemcalc([],atoms,fragment, fragment.mult, temp=temp,pressure=pressure)
        
    print("-------------------------------------------------------------------------")
    print("THERMOCHEM PROTOCOL-single: Step 3. High-level single-point calculation")
    print("-------------------------------------------------------------------------")
    #Workflow (callable function) or ORCATheory object
    #print("SP_theory:", SP_theory)
    #print("callable(SP_theory):", callable(SP_theory))
    #print("callable(SP_theory):", callable(module_highlevel_workflows.SP_theory))
    #print("callable(SP_theory):", callable(SP_theory))
    if callable(SP_theory) is True:
        FinalE, componentsdict = SP_theory(fragment=fragment, charge=fragment.charge,
                    mult=fragment.mult, orcadir=orcadir, numcores=numcores, memory=memory, workflow_args=workflow_args)
    elif SP_theory.__class__.__name__ == "ORCATheory":
        #Adding charge and mult to theory object, taken from each fragment object
        SP_theory.charge = fragment.charge
        SP_theory.mult = fragment.mult
        FinalE = ash.Singlepoint(fragment=fragment, theory=SP_theory)
        SP_theory.cleanup()
        #TODO: Add SCF-energy and corr-energy to dict here. Need to grab. Can we make general?
        componentsdict={}
        #componentsdict = {'E_SCF_CBS' : scf_energy, 'E_corr_CBS' : corr_energy}
    else:
        print("Unknown Singlepoint protocol")
        exit()
    print_time_rel(module_init_time, modulename='thermochemprotocol_single', moduleindex=0)
    return FinalE, componentsdict, thermochem


#Thermochemistry protocol. Take list of fragments, stoichiometry, and 2 theory levels
#Requires orcadir, and Opt_theory level (typically an ORCATheory object), SP_theory (either ORCATTheory or workflow.
def thermochemprotocol_reaction(Opt_theory=None, SP_theory=None, fraglist=None, stoichiometry=None, orcadir=None, numcores=1, memory=5000,
                       workflow_args=None, analyticHessian=True, temp=298.15, pressure=1.0):
    """[summary]

    Args:
        Opt_theory (ASH theory, optional): ASH theory for optimizations. Defaults to None.
        SP_theory (ASH theory, optional): ASH theory for Single-points. Defaults to None.
        fraglist (list, optional): List of ASH fragments. Defaults to None.
        stoichiometry (list, optional): list of integers defining stoichiometry. Defaults to None.
        orcadir (str, optional): Path to ORCA. Defaults to None.
        numcores (int, optional): Number of cores. Defaults to 1.
        memory (int, optional): Memory in MB (ORCA). Defaults to 5000.
        workflow_args ([type], optional): dictionary for workflow arguments. Defaults to None.
        analyticHessian (bool, optional): Analytical Hessian or not. Defaults to True.
        temp (float, optional): Temperature in Kelvin. Defaults to 298.15.
        pressure (float, optional): Pressure in atm. Defaults to 1.0.
    """
    module_init_time=time.time()
    print("")
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL (reaction)-------------", BC.END)
    print("")
    print("Running thermochemprotocol function for fragment list:")
    for i,frag in enumerate(fraglist):
        print("Fragment {} Formula: {}  Label: {}".format(i,frag.prettyformula,frag.label))
    print("Stoichiometry:", stoichiometry)
    print("")
    FinalEnergies_el = []; FinalEnergies_zpve = []; FinalEnthalpies = []; FinalFreeEnergies = []; list_of_dicts = []; ZPVE_Energies=[]
    Hcorr_Energies = []; Gcorr_Energies = []
    
    #Looping over species in fraglist
    for species in fraglist:
        #Get energy and components for species
        FinalE, componentsdict, thermochem = thermochemprotocol_single(fragment=species, Opt_theory=Opt_theory, SP_theory=SP_theory, orcadir=orcadir, numcores=numcores, memory=memory,
                       workflow_args=workflow_args, analyticHessian=analyticHessian, temp=temp, pressure=pressure)
        
        ZPVE=thermochem['ZPVE']
        Hcorr=thermochem['Hcorr']
        Gcorr=thermochem['Gcorr']
        
        FinalEnergies_el.append(FinalE)
        FinalEnergies_zpve.append(FinalE+ZPVE)
        FinalEnthalpies.append(FinalE+Hcorr)
        FinalFreeEnergies.append(FinalE+Gcorr)
        list_of_dicts.append(componentsdict)
        ZPVE_Energies.append(ZPVE)
        Hcorr_Energies.append(Hcorr)
        Gcorr_Energies.append(Gcorr)
        
    print("")
    print("")
    print("FINAL REACTION ENERGY:")
    print("Enthalpy and Gibbs Energies for  T={} and P={}".format(temp,pressure))
    print("----------------------------------------------")
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnergies_el, unit='kcalpermol', label='Total ΔE_el')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnergies_zpve, unit='kcalpermol', label='Total Δ(E+ZPVE)')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnthalpies, unit='kcalpermol', label='Total ΔH')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalFreeEnergies, unit='kcalpermol', label='Total ΔG')
    print("----------------------------------------------")
    print("Individual contributions")
    #Print individual contributions if available
    #ZPVE, Hcorr, gcorr
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ZPVE_Energies, unit='kcalpermol', label='ΔZPVE')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=Hcorr_Energies, unit='kcalpermol', label='ΔHcorr')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=Gcorr_Energies, unit='kcalpermol', label='ΔGcorr')
    #Contributions to CCSD(T) energies
    if 'E_SCF_CBS' in componentsdict:
        scf_parts=[dict['E_SCF_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=scf_parts, unit='kcalpermol', label='ΔSCF')
    if 'E_CCSDcorr_CBS' in componentsdict:
        ccsd_parts=[dict['E_CCSDcorr_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ccsd_parts, unit='kcalpermol', label='ΔCCSD')
    if 'E_triplescorr_CBS' in componentsdict:
        triples_parts=[dict['E_triplescorr_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=triples_parts, unit='kcalpermol', label='Δ(T)')
    if 'E_corr_CBS' in componentsdict:
        valencecorr_parts=[dict['E_corr_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=valencecorr_parts, unit='kcalpermol', label='ΔCCSD+Δ(T) corr')
    if 'E_SO' in componentsdict:
        SO_parts=[dict['E_SO'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=SO_parts, unit='kcalpermol', label='ΔSO')
    if 'E_corecorr_and_SR' in componentsdict:
        CV_SR_parts=[dict['E_corecorr_and_SR'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=CV_SR_parts, unit='kcalpermol', label='ΔCV+SR')
    
    print("")
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL END-------------", BC.END)
    #print_time_rel(ash_header.init_time,modulename='Entire thermochemprotocol')
    print_time_rel(module_init_time, modulename='thermochemprotocol_reaction', moduleindex=0)



#Thermochemistry protocol. Take list of fragments, stoichiometry, and 2 theory levels
#Requires orcadir, and Opt_theory level (typically an ORCATheory object), SP_theory (either ORCATTheory or workflow.
#Old non-modularized code
#TODO: DELETE, deprecated
def old_thermochemprotocol(Opt_theory=None, SP_theory=None, fraglist=None, stoichiometry=None, orcadir=None, numcores=None, memory=5000,
                       workflow_args=None, analyticHessian=True, temp=298.15, pressure=1.0):
    print("")
    print("inactive, to be deleted....")
    exit()
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL-------------", BC.END)
    print("")
    if fraglist[0].charge == None:
        print("1st. Fragment: {}".format(fraglist[0].__dict__))
        print("No charge/mult information present in fragment. Each fragment in provided fraglist must have charge/mult information defined.")
        print("Example:")
        print("fragment.charge= 0; fragment.mult=1")
        print("Exiting...")
        exit()
    #DFT Opt+Freq  and Single-point High-level workflow
    FinalEnergies = []; FinalEnthalpies = []; FinalFreeEnergies = []; list_of_dicts = []; ZPVE_Energies=[]
    Hcorr_Energies = []; Gcorr_Energies = []
    for species in fraglist:
        #Only Opt+Freq for molecules, not atoms
        if species.numatoms != 1:
            #DFT-opt
            #TODO: Check if this works in general. At least for ORCA.
            
            #Adding charge and mult to theory object, taken from each fragment object
            Opt_theory.charge = species.charge
            Opt_theory.mult = species.mult
            interface_geometric.geomeTRICOptimizer(theory=Opt_theory,fragment=species)
            
            #DFT-FREQ
            if analyticHessian == True:
                thermochem = ash.AnFreq(fragment=species, theory=Opt_theory, numcores=numcores)                
            else:
                thermochem = ash.NumFreq(fragment=species, theory=Opt_theory, npoint=2, runmode='serial')
            ZPVE = thermochem['ZPVE']
            Hcorr = thermochem['Hcorr']
            Gcorr = thermochem['Gcorr']
        else:
            #Setting thermoproperties for atom
            thermochem = thermochemcalc([],atoms,species, species.mult, temp=temp,pressure=pressure)
            ZPVE = thermochem['ZPVE']
            Hcorr = thermochem['Hcorr']
            Gcorr = thermochem['Gcorr']
            
        
        #Workflow (callable function) or ORCATheory object
        if callable(SP_theory) is True:
            FinalE, componentsdict = DLPNO_CC_CBS(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=memory, workflow_args=workflow_args)
        elif SP_theory.__class__.__name__ == "ORCATheory":
            #Adding charge and mult to theory object, taken from each fragment object
            SP_theory.charge = species.charge
            SP_theory.mult = species.mult
            FinalE = ash.Singlepoint(fragment=species, theory=SP_theory)
            SP_theory.cleanup()
            #TODO: Add SCF-energy and corr-energy to dict here. Need to grab. Can we make general?
            componentsdict={}
            #componentsdict = {'E_SCF_CBS' : scf_energy, 'E_corr_CBS' : corr_energy}
        else:
            print("Unknown Singlepoint protocol")
            exit()
        FinalEnergies.append(FinalE+ZPVE)
        FinalEnthalpies.append(FinalE+Hcorr)
        FinalFreeEnergies.append(FinalE+Gcorr)
        list_of_dicts.append(componentsdict)
        ZPVE_Energies.append(ZPVE)
        Hcorr_Energies.append(Hcorr)
        Gcorr_Energies.append(Gcorr)
    print("")
    print("")
    print("FINAL REACTION ENERGY:")
    print("Enthalpy and Gibbs Energies for  T={} and P={}".format(temp,pressure))
    print("----------------------------------------------")
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnergies, unit='kcalpermol', label='Total ΔE')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnthalpies, unit='kcalpermol', label='Total ΔH')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalFreeEnergies, unit='kcalpermol', label='Total ΔG')
    print("----------------------------------------------")
    print("Individual contributions")
    #Print individual contributions if available
    #ZPVE
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ZPVE_Energies, unit='kcalpermol', label='ΔZPVE')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=Hcorr_Energies, unit='kcalpermol', label='ΔHcorr')
    ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=Gcorr_Energies, unit='kcalpermol', label='ΔGcorr')
    if 'E_SCF_CBS' in componentsdict:
        scf_parts=[dict['E_SCF_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=scf_parts, unit='kcalpermol', label='ΔSCF')
    if 'E_CCSDcorr_CBS' in componentsdict:
        ccsd_parts=[dict['E_CCSDcorr_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ccsd_parts, unit='kcalpermol', label='ΔCCSD')
    if 'E_triplescorr_CBS' in componentsdict:
        triples_parts=[dict['E_triplescorr_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=triples_parts, unit='kcalpermol', label='Δ(T)')
    if 'E_corr_CBS' in componentsdict:
        valencecorr_parts=[dict['E_corr_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=valencecorr_parts, unit='kcalpermol', label='ΔCCSD+Δ(T) corr')
    if 'E_SO' in componentsdict:
        SO_parts=[dict['E_SO'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=SO_parts, unit='kcalpermol', label='ΔSO')
    if 'E_corecorr_and_SR' in componentsdict:
        CV_SR_parts=[dict['E_corecorr_and_SR'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=CV_SR_parts, unit='kcalpermol', label='ΔCV+SR')
    
    print("")
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL END-------------", BC.END)
    #ash.print_time_rel(ash_header.init_time,modulename='Entire thermochemprotocol')
    #print_time_rel(module_init_time, modulename='thermochemprotocol')

def auto_active_space(fragment=None, orcadir=None, basis="def2-SVP", scalar_rel=None, charge=None, mult=None, 
    initial_orbitals='MP2', functional='TPSS', smeartemp=5000, tgen=1e-1, selection_thresholds=[1.999,0.001],
    numcores=1):
    print_line_with_mainheader("auto_active_space function")
    print("Will do N-step orbital selection scheme")
    print("basis:", basis)
    print("scalar_rel:", scalar_rel)
    print("")
    print("1. Initial Orbital Step")
    print("initial_orbitals:", initial_orbitals)
    print("2. ICE-CI Orbital Step")
    print("ICE-CI tgen:", tgen)
    print("Numcores:", numcores)
    #1. Converge an RI-MP2 natural orbital calculation
    if scalar_rel == None:
        scalar_rel_keyword=""
    else:
        scalar_rel_keyword=scalar_rel

    print("")
    #NOTE: Ideas: Insted of UHF-MP2 step. DFT-SCF and then MP2 natural orbitals from unrelaxed density on top?
    if initial_orbitals == 'MP2':
        steplabel='MP2natorbs'
        orcasimpleinput="! RI-MP2 autoaux tightscf {} {} ".format(basis, scalar_rel_keyword)
        orcablocks="""
        %scf
        maxiter 800
        end
        %mp2
        density unrelaxed
        natorbs true
        end
        """
        ORCAcalc_1 = ash.ORCATheory(orcadir=orcadir, charge=charge, mult=mult, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                    nprocs=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment)
        init_orbitals=ORCAcalc_1.filename+'.mp2nat'

        step1occupations=ash.interface_ORCA.MP2_natocc_grab(ORCAcalc_1.filename+'.out')
        print("MP2natoccupations:", step1occupations)
    elif initial_orbitals == 'FOD':
        steplabel='FODorbs'
        print("Initial orbitals: FOD-DFT")
        print("Functional used:", functional)
        print("Smear temperature: {} K".format(smeartemp))
        #Enforcing UKS so that we get QROs even for S=0
        orcasimpleinput="! UKS  {} {} {} tightscf  slowconv".format(functional,basis, scalar_rel_keyword)
        orcablocks="""
        %scf
        maxiter 800
        Smeartemp {}
        end
        """.format(smeartemp)
        ORCAcalc_1 = ash.ORCATheory(orcadir=orcadir, charge=charge, mult=mult, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                    nprocs=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment)
        step1occupations=ash.interface_ORCA.SCF_FODocc_grab(ORCAcalc_1.filename+'.out')
        print("FOD occupations:", step1occupations)
        #FOD occupations are unrestricted.
        #Need to change 1.0 to 2.0 to get fake double-occupations
        print("Warning: replacing unrestricted set (1.0 occupations) with doubly-occupied (2.0 occupations)")
        step1occupations =  [i if i != 1.0 else 2.0 for i in step1occupations]
        print("New FOD occupations:", step1occupations)
        shutil.copy(ORCAcalc_1.filename+'.gbw', ORCAcalc_1.filename+'_fod.gbw')
        init_orbitals=ORCAcalc_1.filename+'_fod.gbw'
    elif initial_orbitals == 'QRO' or initial_orbitals == 'DFT':
        print("not active")
        #
        exit()
        steplabel='DFTQROorbs'
        print("Initial orbitals: DFT-QRO")
        print("Functional used:", functional)
        #Enforcing UKS so that we get QROs even for S=0
        orcasimpleinput="! UKS {} {} {} tightscf UNO ".format(functional,basis, scalar_rel_keyword)
        orcablocks="""
        %scf
        maxiter 800
        end
        """
        ORCAcalc_1 = ash.ORCATheory(orcadir=orcadir, charge=charge, mult=mult, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                    nprocs=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment)
        step1occupations,qroenergies=ash.interface_ORCA.QRO_occ_energies_grab(ORCAcalc_1.filename+'.out')
        print("occupations:", step1occupations)
        print("qroenergies:", qroenergies)
        init_orbitals=ORCAcalc_1.filename+'.qro'
        #if mult != 1:
        #    init_orbitals=ORCAcalc.filename+'.qro'
        #else:
        #    #Use canonical if not open-shell
        #    init_orbitals=ORCAcalc.filename+'.gbw'

    print("Initial orbitals file:", init_orbitals)

    #2a. Count size of ICE-CI CASSCF active space to test
    print("")
    upper_threshold=selection_thresholds[0]
    lower_threshold=selection_thresholds[1]
    print("Selecting size of active-space for ICE-CI step")
    print("Using orbital tresholds:", upper_threshold,lower_threshold )
    numelectrons,numorbitals=functions_elstructure.select_space_from_occupations(step1occupations, selection_thresholds=[upper_threshold,lower_threshold])
    print("Will use CAS size of CAS({},{}) for ICE-CI step".format(numelectrons,numorbitals))

    #2b. Read orbitals into ICE-CI calculation
    orcasimpleinput="! CASSCF  {} {} MOREAD ".format(basis, scalar_rel_keyword)
    orcablocks="""
    %maxcore 5000
    %moinp \"{}\"
    %casscf
    gtol 99999
    nel {}
    norb {}
    cistep ice
    ci
    tgen {}
    maxiter 200
    end
    end
    """.format(init_orbitals,numelectrons,numorbitals,tgen)
    ORCAcalc_2 = ash.ORCATheory(orcadir=orcadir, charge=charge, mult=mult, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                nprocs=numcores)
    ash.Singlepoint(theory=ORCAcalc_2,fragment=fragment)

    ICEnatoccupations=ash.interface_ORCA.CASSCF_natocc_grab(ORCAcalc_2.filename+'.out')

    finalICE_Gen_CFGs,finalICE_SD_CFGs=ash.interface_ORCA.ICE_WF_size(ORCAcalc_2.filename+'.out')

    Tvar=float(pygrep("ICE TVar                          ...", ORCAcalc_2.filename+'.out')[-1])
    print("ICE-CI step done")
    print("Note: New natural orbitals from ICE-CI density matrix formed!")
    print("")

    print("Wavefunction size:")
    print("Tgen:", tgen)
    print("Tvar:", Tvar)
    print("Orbital space of CAS({},{}) used for ICE-CI step".format(numelectrons,numorbitals))
    print("Num generator CFGs:", finalICE_Gen_CFGs)
    print("Num CFGS after S+D:", finalICE_SD_CFGs)
    print("")

    #3. Do something clever with occupations

    print("Table of natural occupation numbers")
    print("")
    print("{:<9} {:6} {:6}".format("Orbital", steplabel, "ICE-nat-occ"))
    print("----------------------------------------")
    for index,(step1occ,iceocc) in enumerate(zip(step1occupations,ICEnatoccupations)):
        print("{:<9} {:9.4f} {:9.4f}".format(index,step1occ,iceocc))


    minimal_CAS=functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.95,0.05])
    medium_CAS=functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.98,0.02])
    large_CAS=functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.995,0.005])
    print("")
    print("Recommended active spaces based on ICE-CI natural occupations:")
    print("Minimal (1.95,0.05): CAS({},{})".format(minimal_CAS[0],minimal_CAS[1]))
    print("Medium (1.98,0.02): CAS({},{})".format(medium_CAS[0],medium_CAS[1]))
    print("Large (1.995,0.005): CAS({},{})".format(large_CAS[0],large_CAS[1]))

    print("Orbital file to use for future calculations:", ORCAcalc_2.filename+'.gbw')
    print("Note: orbitals are new natural orbitals formed from the ICE-CI density matrix")

    #Returning list of active spaces
    return [[minimal_CAS[0],minimal_CAS[1]],[medium_CAS[0],medium_CAS[1]],[large_CAS[0],large_CAS[1]]]