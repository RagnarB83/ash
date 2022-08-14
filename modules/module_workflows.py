"""
Contains functions defining multi-step workflows

"""
import os
import subprocess as sp
import shutil
import time
import copy

#import ash
#from ash import Singlepoint
import ash.interfaces.interface_geometric
import ash.interfaces.interface_crest
from ash.functions.functions_general import BC,print_time_rel,print_line_with_mainheader,pygrep, ashexit,n_max_values
#from ash.modules.module_singlepoint import Singlepoint_fragments
from ash.modules.module_highlevel_workflows import ORCA_CC_CBS_Theory
from ash.modules.module_coords import read_xyzfiles
import ash.functions.functions_elstructure
from ash.modules.module_plotting import ASH_plot
from ash.modules.module_singlepoint import ReactionEnergy
from ash.modules.module_coords import check_charge_mult
from ash.modules.module_freq import thermochemcalc

#Simple class to keep track of results. To be extended
class ProjectResults():
    def __init__(self,name=None):
        self.name=name
        print("Creating ProjectResults object.")

    def printall(self):
        print("Printing all defined attributes of object:", self.name)
        for i,j in self.__dict__.items():
            print("{} : {}".format(i,j))






#Provide crest/xtb info, MLtheory object (e.g. ORCA), HLtheory object (e.g. ORCA)
def confsampler_protocol(fragment=None, crestdir=None, xtbmethod='GFN2-xTB', MLtheory=None, 
                         HLtheory=None, numcores=1, charge=None, mult=None):
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

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, MLtheory.theorytype, fragment, "confsampler_protocol", theory=MLtheory)

    #1. Calling crest
    #call_crest(fragment=molecule, xtbmethod='GFN2-xTB', crestdir=crestdir, charge=charge, mult=mult, solvent='H2O', energywindow=6 )
    ash.interfaces.interface_crest.call_crest(fragment=fragment, xtbmethod=xtbmethod, crestdir=crestdir, charge=charge, mult=mult, numcores=numcores)

    #2. Grab low-lying conformers from crest_conformers.xyz as list of ASH fragments.
    list_conformer_frags, xtb_energies = ash.interfaces.interface_crest.get_crest_conformers(charge=charge, mult=mult)

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
        ash.interfaces.interface_geometric.geomeTRICOptimizer(fragment=conformer, theory=MLtheory, coordsystem='tric', charge=charge, mult=mult)
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
        HLenergy = ash.Singlepoint(theory=HLtheory, fragment=conformer, charge=charge, mult=mult)

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
def thermochemprotocol_single(fragment=None, Opt_theory=None, SP_theory=None, numcores=None, memory=5000,
                       analyticHessian=True, temp=298.15, pressure=1.0, charge=None, mult=None):
    module_init_time=time.time()
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL (single-species)-------------", BC.END)
    #Check charge/mult
    #charge,mult = check_charge_mult(charge, mult, Opt_theory.theorytype, fragment, "thermochemprotocol_single", theory=Opt_theory)
    
    #DFT Opt+Freq  and Single-point High-level workflow
    #Only Opt+Freq for molecules, not atoms
    print("-------------------------------------------------------------------------")
    print("THERMOCHEM PROTOCOL-single: Step 1. Geometry optimization")
    print("-------------------------------------------------------------------------")
    if fragment.numatoms != 1:
        if Opt_theory != None:
            #DFT-opt
            ash.interfaces.interface_geometric.geomeTRICOptimizer(theory=Opt_theory,fragment=fragment, charge=charge, mult=mult)
            print("-------------------------------------------------------------------------")
            print("THERMOCHEM PROTOCOL-single: Step 2. Frequency calculation")
            print("-------------------------------------------------------------------------")
            
            #Checking analyticHessian and Theory
            if analyticHessian is True and isinstance(Opt_theory,ash.ORCATheory) is False: 
                analyticHessian=False

            #DFT-FREQ
            if analyticHessian == True:
                thermochem = ash.AnFreq(fragment=fragment, theory=Opt_theory, numcores=numcores, charge=charge, mult=mult)                
            else:
                thermochem = ash.NumFreq(fragment=fragment, theory=Opt_theory, npoint=2, runmode='serial', charge=charge, mult=mult)
        else:
            print("Opt_theory is set to None. Skipping optimization and frequency calculation.\n")
            #If Opt_theory == None => No Opt, no freq
            thermochem={'ZPVE':0.0,'Gcorr':0.0,'Hcorr':0.0}
    else:
        #Setting thermoproperties for atom
        thermochem = thermochemcalc([],[0],fragment, fragment.mult, temp=temp,pressure=pressure)
        
    print("-------------------------------------------------------------------------")
    print("THERMOCHEM PROTOCOL-single: Step 3. High-level single-point calculation")
    print("-------------------------------------------------------------------------")

    FinalE = ash.Singlepoint(fragment=fragment, theory=SP_theory, charge=charge, mult=mult)
    #Get energy components
    if isinstance(SP_theory,ORCA_CC_CBS_Theory):
        componentsdict=SP_theory.energy_components
    else:
        componentsdict={}
    SP_theory.cleanup()

    print_time_rel(module_init_time, modulename='thermochemprotocol_single', moduleindex=0)
    return FinalE, componentsdict, thermochem


#Thermochemistry protocol. Take list of fragments, stoichiometry, and 2 theory levels
#Requires orcadir, and Opt_theory level (typically an ORCATheory object), SP_theory (either ORCATTheory or workflow.
def thermochemprotocol_reaction(Opt_theory=None, SP_theory=None, reaction=None, fraglist=None, stoichiometry=None, numcores=1, memory=5000,
                       analyticHessian=True, temp=298.15, pressure=1.0, unit='kcal/mol'):
    """[summary]

    Args:
        Opt_theory (ASH theory, optional): ASH theory for optimizations. Defaults to None.
        SP_theory (ASH theory, optional): ASH theory for Single-points. Defaults to None.
        reaction (ASH reaction): ASH reaction object. Defaults to None.
        fraglist (list, optional): List of ASH fragments. Defaults to None.
        stoichiometry (list, optional): list of integers defining stoichiometry. Defaults to None.
        numcores (int, optional): Number of cores. Defaults to 1.
        memory (int, optional): Memory in MB (ORCA). Defaults to 5000.
        analyticHessian (bool, optional): Analytical Hessian or not. Defaults to True.
        temp (float, optional): Temperature in Kelvin. Defaults to 298.15.
        pressure (float, optional): Pressure in atm. Defaults to 1.0.
    """
    module_init_time=time.time()
    print("")
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL (reaction)-------------", BC.END)
    print("")
    # Reaction input
    if reaction != None:
        fraglist = reaction.fragments
        stoichiometry = reaction.stoichiometry

    print("Running thermochemprotocol function for fragments:")
    for i,frag in enumerate(fraglist):
        print(f"Fragment {i} Formula: {frag.prettyformula}  Label: {frag.label} Charge: {frag.charge} Mult: {frag.mult}")
    print("Stoichiometry:", stoichiometry)
    print("")
    FinalEnergies_el = []; FinalEnergies_zpve = []; FinalEnthalpies = []; FinalFreeEnergies = []; list_of_dicts = []; ZPVE_Energies=[]
    Hcorr_Energies = []; Gcorr_Energies = []
    
    #Looping over species in fraglist
    for species in fraglist:
        #Get energy and components for species
        FinalE, componentsdict, thermochem = thermochemprotocol_single(fragment=species, Opt_theory=Opt_theory, SP_theory=SP_theory, numcores=numcores, memory=memory,
                       analyticHessian=analyticHessian, temp=temp, pressure=pressure, charge=species.charge, mult=species.mult)
        
        print("FinalE:", FinalE)
        print("componentsdict", componentsdict)
        print("thermochem", thermochem)
        ZPVE=thermochem['ZPVE']
        Hcorr=thermochem['Hcorr']
        Gcorr=thermochem['Gcorr']
        print("ZPVE:", ZPVE)
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
    #Final reaction energy and components dictionary
    finaldict={}
    print("FINAL REACTION ENERGY:")
    print("Enthalpy and Gibbs Energies for  T={} and P={}".format(temp,pressure))
    print("-"*80)
    deltaE=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnergies_el, unit=unit, label='Total ΔE_el')[0]
    deltaE_0=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnergies_zpve, unit=unit, label='Total Δ(E+ZPVE)')[0]
    deltaH=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnthalpies, unit=unit, label='Total ΔH(T={}'.format(temp))[0]
    deltaG=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalFreeEnergies, unit=unit, label='Total ΔG(T={}'.format(temp))[0]
    print("-"*80)
    print("Individual contributions")
    #Print individual contributions if available
    #ZPVE, Hcorr, gcorr
    deltaZPVE=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ZPVE_Energies, unit=unit, label='ΔZPVE')[0]
    deltaHcorr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=Hcorr_Energies, unit=unit, label='ΔHcorr')[0]
    deltaGcorr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=Gcorr_Energies, unit=unit, label='ΔGcorr')[0]
    print("-"*80)
    finaldict={'deltaE':deltaE, 'deltaE_0':deltaE_0, 'deltaH':deltaH, 'deltaG':deltaG, 'deltaZPVE':deltaZPVE, 'deltaHcorr':deltaHcorr, 'deltaGcorr':deltaGcorr}
    #Contributions to CCSD(T) energies
    if 'E_SCF_CBS' in componentsdict:
        scf_parts=[dict['E_SCF_CBS'] for dict in list_of_dicts]
        deltaSCF=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=scf_parts, unit=unit, label='ΔSCF')[0]
        finaldict['deltaSCF']=deltaSCF
    if 'E_corrCCSD_CBS' in componentsdict:
        ccsd_parts=[dict['E_corrCCSD_CBS'] for dict in list_of_dicts]
        delta_CCSDcorr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ccsd_parts, unit=unit, label='ΔCCSD')[0]
        finaldict['delta_CCSDcorr']=delta_CCSDcorr
    if 'E_corrCCT_CBS' in componentsdict:
        triples_parts=[dict['E_corrCCT_CBS'] for dict in list_of_dicts]
        delta_Tcorr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=triples_parts, unit=unit, label='Δ(T)')[0]
        finaldict['delta_Tcorr']=delta_Tcorr
    if 'E_corr_CBS' in componentsdict:
        valencecorr_parts=[dict['E_corr_CBS'] for dict in list_of_dicts]
        delta_CC_corr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=valencecorr_parts, unit=unit, label='ΔCCSD+Δ(T) corr')[0]
        finaldict['delta_CC_corr']=delta_CC_corr
    if 'E_SO' in componentsdict:
        SO_parts=[dict['E_SO'] for dict in list_of_dicts]
        delta_SO_corr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=SO_parts, unit=unit, label='ΔSO')[0]
        finaldict['delta_SO_corr']=delta_SO_corr
    if 'E_corecorr_and_SR' in componentsdict:
        CV_SR_parts=[dict['E_corecorr_and_SR'] for dict in list_of_dicts]
        delta_CVSR_corr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=CV_SR_parts, unit=unit, label='ΔCV+SR')[0]
        finaldict['delta_CVSR_corr']=delta_CVSR_corr
    if 'T1energycorr' in componentsdict:
        T1corr_parts=[dict['T1energycorr'] for dict in list_of_dicts]
        delta_T1_corr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=T1corr_parts, unit=unit, label='ΔΔT1corr')[0]
        finaldict['delta_T1_corr']=delta_T1_corr
    if 'E_FCIcorrection' in componentsdict:
        fcicorr_parts=[dict['E_FCIcorrection'] for dict in list_of_dicts]
        delta_FCI_corr=ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=fcicorr_parts, unit=unit, label='ΔFCIcorr')[0]
        finaldict['delta_FCI_corr']=delta_FCI_corr
    print("-"*80)
    print("")
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL END-------------", BC.END)
    print_time_rel(module_init_time, modulename='thermochemprotocol_reaction', moduleindex=0)

    return finaldict

def auto_active_space(fragment=None, orcadir=None, basis="def2-SVP", scalar_rel=None, charge=None, mult=None, 
    initial_orbitals='MP2', functional='TPSS', smeartemp=5000, tgen=1e-1, selection_thresholds=[1.999,0.001],
    numcores=1, memory=9000, extrablocks=None):

    print_line_with_mainheader("auto_active_space function")

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "auto_active_space", theory=None)

    print("Will do N-step orbital selection scheme")
    print("basis:", basis)
    print("scalar_rel:", scalar_rel)
    print("")
    print("1. Initial Orbital Step")
    print("initial_orbitals:", initial_orbitals)
    print("2. ICE-CI Orbital Step")
    print("ICE-CI tgen:", tgen)
    print("Numcores:", numcores)
    print("Extra blocks:", extrablocks)
    #1. Converge an RI-MP2 natural orbital calculation
    if scalar_rel == None:
        scalar_rel_keyword=""
    else:
        scalar_rel_keyword=scalar_rel

    #extrablocks
    if extrablocks == None:
        extrablocks=""

    print("")
    #NOTE: Ideas: Insted of UHF-MP2 step. DFT-SCF and then MP2 natural orbitals from unrelaxed/relaxed density on top?
    #FULL double-hybrid natural orbitals??
    if initial_orbitals == 'MP2':
        print("Initial orbitals: MP2")
        steplabel='MP2natorbs'
        orcasimpleinput="! RI-MP2 autoaux tightscf {} {} ".format(basis, scalar_rel_keyword)
        orcablocks="""
        %scf
        maxiter 800
        end
        %mp2
        density relaxed
        natorbs true
        end
        {}
        """.format(extrablocks)
        ORCAcalc_1 = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                    numcores=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment, charge=charge, mult=mult)
        shutil.copy(ORCAcalc_1.filename+'.out', ORCAcalc_1.filename+'_MP2natorbs.out')
        init_orbitals=ORCAcalc_1.filename+'.mp2nat'

        step1occupations=ash.interfaces.interface_ORCA.MP2_natocc_grab(ORCAcalc_1.filename+'.out')
        print("MP2natoccupations:", step1occupations)
    elif initial_orbitals == 'FOD':
        print("Initial orbitals: FOD-DFT")
        steplabel='FODorbs'
        print("Functional used:", functional)
        print("Smear temperature: {} K".format(smeartemp))
        #Enforcing UKS so that we get QROs even for S=0
        orcasimpleinput="! UKS  {} {} {} tightscf  slowconv".format(functional,basis, scalar_rel_keyword)
        orcablocks="""
        %scf
        maxiter 800
        Smeartemp {}
        end
        {}
        """.format(smeartemp, extrablocks)
        ORCAcalc_1 = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                    numcores=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment, charge=charge, mult=mult)
        shutil.copy(ORCAcalc_1.filename+'.out', ORCAcalc_1.filename+'_FOD_DFTorbs.out')
        step1occupations=ash.interfaces.interface_ORCA.SCF_FODocc_grab(ORCAcalc_1.filename+'.out')
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
        ashexit()
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
        ORCAcalc_1 = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                    numcores=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment, charge=charge, mult=mult)
        step1occupations,qroenergies=ash.interfaces.interface_ORCA.QRO_occ_energies_grab(ORCAcalc_1.filename+'.out')
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
    numelectrons,numorbitals=ash.functions.functions_elstructure.select_space_from_occupations(step1occupations, selection_thresholds=[upper_threshold,lower_threshold])
    print("Will use CAS size of CAS({},{}) for ICE-CI step".format(numelectrons,numorbitals))

    #2b. Read orbitals into ICE-CI calculation
    orcasimpleinput="! CASSCF  {} {} MOREAD ".format(basis, scalar_rel_keyword)
    orcablocks="""
    %maxcore {}
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
    {}
    """.format(memory,init_orbitals,numelectrons,numorbitals,tgen, extrablocks)
    ORCAcalc_2 = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks,
                                numcores=numcores)
    ash.Singlepoint(theory=ORCAcalc_2,fragment=fragment, charge=charge, mult=mult)

    ICEnatoccupations=ash.interfaces.interface_ORCA.CASSCF_natocc_grab(ORCAcalc_2.filename+'.out')

    finalICE_Gen_CFGs,finalICE_SD_CFGs=ash.interfaces.interface_ORCA.ICE_WF_size(ORCAcalc_2.filename+'.out')

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


    minimal_CAS=ash.functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.95,0.05])
    medium1_CAS=ash.functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.98,0.02])
    medium2_CAS=ash.functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.985,0.015])
    medium3_CAS=ash.functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.99,0.01])
    medium4_CAS=ash.functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.992,0.008])
    large_CAS=ash.functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.995,0.005])

    spaces_dict={"minimal_CAS":minimal_CAS,"medium1_CAS":medium1_CAS,"medium2_CAS":medium2_CAS, "medium3_CAS":medium3_CAS, "medium4_CAS":medium4_CAS, "large_CAS":large_CAS  }

    print("")
    print("Recommended active spaces based on ICE-CI natural occupations:")
    print("Minimal (1.95,0.05): CAS({},{})".format(minimal_CAS[0],minimal_CAS[1]))
    print("Medium1 (1.98,0.02): CAS({},{})".format(medium1_CAS[0],medium1_CAS[1]))
    print("Medium2 (1.985,0.015): CAS({},{})".format(medium2_CAS[0],medium2_CAS[1]))
    print("Medium3 (1.99,0.01): CAS({},{})".format(medium3_CAS[0],medium3_CAS[1]))
    print("Medium4 (1.992,0.008): CAS({},{})".format(medium4_CAS[0],medium4_CAS[1]))
    print("Large (1.995,0.005): CAS({},{})".format(large_CAS[0],large_CAS[1]))

    print("Orbital file to use for future calculations:", ORCAcalc_2.filename+'.gbw')
    print("Note: orbitals are new natural orbitals formed from the ICE-CI density matrix")

    #Returning dict of active spaces
    return spaces_dict



#Simple function to run calculations (SP or OPT) on collection of XYZ-files
#Assuming XYZ-files have charge,mult info in header, or if single global charge,mult, apply that
#NOTE: Maybe change to opt_theory and SP_theory. If both are set: then first Opt using opt_theory then HL-SP using SP_theory.
#If only opt_theory set then second SP step skipped, if only SP_theory set then we skip the first Opt-step
# TODO: Add parallelization. Requires some rewriting 
def calc_xyzfiles(xyzdir=None, theory=None, HL_theory=None, Opt=False, Freq=False, charge=None, mult=None, xtb_preopt=False ):
    print_line_with_mainheader("calc_xyzfiles function")

    #Checkf if xyz-directory exists
    if os.path.isdir(xyzdir) is False:
        print("XYZ directory does not exist. Check that dirname is complete, has been copied to scratch. You may have to use full path to dir")
        ashexit()
    #Whether we have an extra high-level single-point step after Optimization
    if HL_theory != None:
        Highlevel=True
    else:
        Highlevel=False


    print("XYZ directory:", xyzdir)
    print("Theory:", theory.__class__.__name__)
    print("Optimization:", Opt)
    if Opt is True:
        print("Highlevel Theory SP after Opt:", HL_theory.__class__.__name__)
    print("Global charge/mult options:", charge, mult)
    print("xTB preoptimization", xtb_preopt)

    if charge == None or mult == None:
        print("Charge/mult options are None. This means that XYZ-files must have charge/mult information in their header\n\n")
        readchargemult=True
    else:
        readchargemult=False

    energies=[]
    hlenergies=[]
    optenergies=[]
    finalxyzdir="optimized_xyzfiles"
    #Remove old dir if present
    try:
        shutil.rmtree(finalxyzdir)
    except:
        pass
    #Create new directory with optimized geometries
    os.mkdir(finalxyzdir)

    #Reading XYZ-files to get list of fragments
    fragments =read_xyzfiles(xyzdir,readchargemult=readchargemult, label_from_filename=True)

    #List of original filenames
    filenames=[frag.label for frag in fragments]

    #Now looping over fragments
    for fragment in fragments:
        filename=fragment.label
        print("filename:", filename)
        #Global charge/mult keywords (rare)
        if charge != None and mult != None:
            fragment.charge=charge; fragment.mult=mult
        #Do Optimization or Singlepoint
        if Opt is True:
            if xtb_preopt is True:
                print("xTB Pre-optimization is active. Will first call xTB directly to do pre-optimization before actual optimization!")
                #Defining temporary xtBtheory
                xtbcalc=ash.xTBTheory(numcores=theory.numcores)
                #Run direct xtb optimization. This will update fragment.
                xtbcalc.Opt(fragment=fragment, charge=fragment.charge, mult=fragment.mult)
                xtbcalc.cleanup()
            
            #Now doing actual OPT
            optenergy = ash.interfaces.interface_geometric.geomeTRICOptimizer(theory=theory, fragment=fragment, coordsystem='tric', charge=fragment.charge, mult=fragment.mult)
            theory.cleanup()
            energy=optenergy
            optenergies.append(optenergy)
            #Rename optimized XYZ-file
            filenamestring_suffix="" #nothing for now
            os.rename("Fragment-optimized.xyz",os.path.splitext(fragment.label)[0]+filenamestring_suffix+".xyz")
            shutil.copy(os.path.splitext(fragment.label)[0]+filenamestring_suffix+".xyz",finalxyzdir)

            #Freq job after OPt
            if Freq is True:
                print("Performing Numerical Frequency job on Optimized fragment.")
                thermochem = ash.NumFreq(fragment=fragment, theory=theory, npoint=2, displacement=0.005, numcores=theory.numcores, runmode='serial', charge=fragment.charge, mult=fragment.mult)
                theory.cleanup()
            if Highlevel is True:
                print("Performing Highlevel on Optimized fragment.")
                hlenergy = ash.Singlepoint(theory=HL_theory, fragment=fragment, charge=fragment.charge, mult=fragment.mult)
                energy=hlenergy
                hlenergies.append(hlenergy)
                HL_theory.cleanup()
        else:
            energy = ash.Singlepoint(theory=theory, fragment=fragment, charge=fragment.charge, mult=fragment.mult)
            theory.cleanup()
        
        energies.append(energy)
        print("Energy of file {} : {} Eh".format(fragment.label, energy))
        print("")

    #TODO: Collect things in dictionary before printing table
    # Then we can sort the items in an intelligent way before printing
    #TODO: if Freq is True, print E+ZPE, H, G, ZPE, Hcorr, Gcorr
    print("optenergies:", optenergies)
    print("hlenergies:", hlenergies)
    print("Final energies", energies)
    print("\n{:30} {:>7} {:>7} {:>20}".format("XYZ-file","Charge","Mult", "Energy(Eh)"))
    print("-"*70)
    for xyzfile, frag, e in zip(filenames, fragments,energies):
        print("{:30} {:>7} {:>7} {:>20.10f}".format(xyzfile,frag.charge, frag.mult, e))
    
    if Opt is True:
        print("\n\nXYZ-files with optimized coordinates can be found in:", finalxyzdir)

    return energies


def Reaction_Highlevel_Analysis(fraglist=None, stoichiometry=None, numcores=1, memory=7000, reactionlabel='Reactionlabel', energy_unit='kcal/mol',
                                def2_family=True, cc_family=True, aug_cc_family=False, F12_family=True, DLPNO=False, extrapolation=True, highest_cardinal=6,
                                plot=True ):
    """Function to perform high-level CCSD(T) calculations for a reaction with associated plots.
       Performs CCSD(T) with cc and def2 basis sets, CCSD(T)-F12 and CCSD(T)/CBS extrapolations

    Args:
        fragment ([type], optional): [description]. Defaults to None.
        fraglist ([type], optional): [description]. Defaults to None.
        stoichiometry ([type], optional): [description]. Defaults to None.
        numcores (int, optional): [description]. Defaults to 1.
        memory (int, optional): [description]. Defaults to 7000.
        reactionlabel (str, optional): [description]. Defaults to 'Reactionlabel'.
        energy_unit (str): Energy unit for ReactionEnergy. Options: 'kcal/mol', 'kJ/mol', 'eV', 'cm-1'. Default: 'kcal/mol'
        def2_family (bool, optional): [description]. Defaults to True.
        cc_family (bool, optional): [description]. Defaults to True.
        F12_family (bool, optional): [description]. Defaults to True.
        highest_cardinal (int, optional): [description]. Defaults to 5.
        plot (Boolean): whether to plot the results or not (requires Matplotlib). Defaults to True. 
    """
    elements_involved=[]
    for frag in fraglist:
        if frag.charge ==None or frag.mult ==None or frag.label == None:
            print("All fragments provided must have charge, mult defined and a label.")
            print("Example: N2=Fragment(xyzfile='n2.xyz', charge=0, mult=1, label='N2'")
            ashexit()
        elements_involved=elements_involved+frag.elems

    #Combined list of all elements involved in the species of the reaction
    elements_involved=list(set(elements_involved))
    specieslist=fraglist
    ###################################################
    # Do single-basis CCSD(T) calculations: def2 family
    ###################################################
    if def2_family is True:
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
        CCSDT_def2_bases_proj.cardinals=[2,3,4]
        CCSDT_def2_bases_proj.labels=[]

        #For loop that iterates over basis sets
        for cardinal in CCSDT_def2_bases_proj.cardinals:
            label=cardinal
            CCSDT_def2_bases_proj.labels.append(label)
            #Define theory level
            cc = ORCA_CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="def2", DLPNO=DLPNO, numcores=numcores, memory=memory)
            #Single-point calcs on all fragments
            energies=ash.Singlepoint_fragments(fragments=specieslist, theory=cc)
            for species,e in zip(specieslist,energies):
                CCSDT_def2_bases_proj.species_energies_dict[species.label].append(e)

            #Storing total energies of all species in dict
            CCSDT_def2_bases_proj.energy_dict[label]=energies

            #Store reaction energy in dict
            reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit=energy_unit, label=reactionlabel, silent=False)
            CCSDT_def2_bases_proj.reaction_energy_dict[label]=reaction_energy
            CCSDT_def2_bases_proj.reaction_energy_list.append(reaction_energy)

        #Print results
        CCSDT_def2_bases_proj.printall()

    ###################################################
    # Do single-basis CCSD(T) calculations: cc family
    ###################################################
    if cc_family is True:
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
        CCSDT_cc_bases_proj.cardinals=list(range(2,highest_cardinal+1))
        CCSDT_cc_bases_proj.labels=[]

        #For loop that iterates over basis sets
        for cardinal in CCSDT_cc_bases_proj.cardinals:
            label=cardinal
            CCSDT_cc_bases_proj.labels.append(label)
            #Define theory level
            cc = ORCA_CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="cc", DLPNO=DLPNO, numcores=numcores, memory=memory)
            #Single-point calcs on all fragments
            energies=ash.Singlepoint_fragments(fragments=specieslist, theory=cc)
            for species,e in zip(specieslist,energies):
                CCSDT_cc_bases_proj.species_energies_dict[species.label].append(e)

            #Storing total energies of all species in dict
            CCSDT_cc_bases_proj.energy_dict[label]=energies

            #Store reaction energy in dict
            reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit='kcal/mol', label=reactionlabel, silent=False)
            CCSDT_cc_bases_proj.reaction_energy_dict[label]=reaction_energy
            CCSDT_cc_bases_proj.reaction_energy_list.append(reaction_energy)

        #Print results
        CCSDT_cc_bases_proj.printall()

    ###################################################
    # Do single-basis CCSD(T) calculations: aug_cc family
    ###################################################
    if aug_cc_family is True:
        print("Now running through single-basis CCSD(T) calculations with aug_cc family")

        #Storing the results in a ProjectResults object
        CCSDT_aug_cc_bases_proj = ProjectResults('CCSDT_aug_cc_bases')
        CCSDT_aug_cc_bases_proj.energy_dict={}
        CCSDT_aug_cc_bases_proj.reaction_energy_dict={}
        CCSDT_aug_cc_bases_proj.reaction_energy_list=[]

        #Dict to store energies of species. Uses fragment label that has to be defined above for each
        CCSDT_aug_cc_bases_proj.species_energies_dict={}
        for species in specieslist:
            CCSDT_aug_cc_bases_proj.species_energies_dict[species.label]=[]

        #Define basis-sets and labels
        CCSDT_aug_cc_bases_proj.cardinals=list(range(2,highest_cardinal+1))
        CCSDT_aug_cc_bases_proj.labels=[]

        #For loop that iterates over basis sets
        for cardinal in CCSDT_aug_cc_bases_proj.cardinals:
            label=cardinal
            CCSDT_aug_cc_bases_proj.labels.append(label)
            #Define theory level
            cc = ORCA_CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="aug-cc", DLPNO=DLPNO, numcores=numcores, memory=memory)
            #Single-point calcs on all fragments
            energies=ash.Singlepoint_fragments(fragments=specieslist, theory=cc)
            for species,e in zip(specieslist,energies):
                CCSDT_aug_cc_bases_proj.species_energies_dict[species.label].append(e)

            #Storing total energies of all species in dict
            CCSDT_aug_cc_bases_proj.energy_dict[label]=energies

            #Store reaction energy in dict
            reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit=energy_unit, label=reactionlabel, silent=False)
            CCSDT_aug_cc_bases_proj.reaction_energy_dict[label]=reaction_energy
            CCSDT_aug_cc_bases_proj.reaction_energy_list.append(reaction_energy)

        #Print results
        CCSDT_aug_cc_bases_proj.printall()




    ###########################################
    # Do single-basis CCSD(T)-F12 calculations
    ###########################################
    if F12_family is True:
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
        CCSDTF12_bases_proj.cardinals=[2,3,4]
        CCSDTF12_bases_proj.labels=[]

        #For loop that iterates over basis sets
        for cardinal in CCSDTF12_bases_proj.cardinals:
            label=str(cardinal)
            CCSDTF12_bases_proj.labels.append(label)
            #Define theory level
            cc = ORCA_CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="cc-f12", F12=True, DLPNO=DLPNO, numcores=numcores, memory=memory)
            #Single-point calcs on all fragments
            energies=ash.Singlepoint_fragments(fragments=specieslist, theory=cc)
            for species,e in zip(specieslist,energies):
                CCSDTF12_bases_proj.species_energies_dict[species.label].append(e)

            #Storing total energies of all species in dict
            CCSDTF12_bases_proj.energy_dict[label]=energies

            #Store reaction energy in dict
            reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit=energy_unit, label=reactionlabel, silent=False)
            CCSDTF12_bases_proj.reaction_energy_dict[label]=reaction_energy
            CCSDTF12_bases_proj.reaction_energy_list.append(reaction_energy)

        print()
        #Print results
        CCSDTF12_bases_proj.printall()

    #################################################
    # Do CCSD(T)/CBS extrapolations with cc family
    #################################################
    if extrapolation is True:
        print("Now running through extrapolation CCSD(T) cc calculations")

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
        if highest_cardinal == 6:
            CCSDTextrap_proj.cardinals=[[2,3],[3,4],[4,5],[5,6]]
        elif highest_cardinal == 5:
            CCSDTextrap_proj.cardinals=[[2,3],[3,4],[4,5]]
        else:
            CCSDTextrap_proj.cardinals=[[2,3],[3,4]]
        CCSDTextrap_proj.labels=[]

        #For loop that iterates over basis sets
        for cardinals in CCSDTextrap_proj.cardinals:
            label=str(cardinal)
            CCSDTextrap_proj.labels.append(label)
            #Define theory level
            cc = ORCA_CC_CBS_Theory(elements=elements_involved, cardinals = cardinals, basisfamily="cc", DLPNO=DLPNO, numcores=numcores, memory=memory)
            #Single-point calcs on all fragments
            energies=ash.Singlepoint_fragments(fragments=specieslist, theory=cc)
            for species,e in zip(specieslist,energies):
                CCSDTextrap_proj.species_energies_dict[species.label].append(e)

            #Storing total energies of all species in dict
            CCSDTextrap_proj.energy_dict[label]=energies

            #Store reaction energy in dict
            reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit=energy_unit, label=reactionlabel, silent=False)
            CCSDTextrap_proj.reaction_energy_dict[label]=reaction_energy
            CCSDTextrap_proj.reaction_energy_list.append(reaction_energy)

        print()
        #Print results
        CCSDTextrap_proj.printall()


    #################################################
    # Do CCSD(T)/CBS extrapolations with aug_cc family
    #################################################
    if extrapolation is True:
        if aug_cc_family is True:
            print("Now running through extrapolation CCSD(T) aug_cc calculations")

            #Storing the results in a ProjectResults object
            CCSDTextrapaugcc_proj = ProjectResults('CCSDT_extrap_augcc')
            CCSDTextrapaugcc_proj.energy_dict={}
            CCSDTextrapaugcc_proj.reaction_energy_dict={}
            CCSDTextrapaugcc_proj.reaction_energy_list=[]

            #Dict to store energies of species. Uses fragment label that has to be defined above for each
            CCSDTextrapaugcc_proj.species_energies_dict={}
            for species in specieslist:
                CCSDTextrapaugcc_proj.species_energies_dict[species.label]=[]

            #Define basis-sets and labels
            if highest_cardinal == 6:
                CCSDTextrapaugcc_proj.cardinals=[[2,3],[3,4],[4,5],[5,6]]
            elif highest_cardinal == 5:
                CCSDTextrapaugcc_proj.cardinals=[[2,3],[3,4],[4,5]]
            else:
                CCSDTextrapaugcc_proj.cardinals=[[2,3],[3,4]]
            CCSDTextrapaugcc_proj.labels=[]

            #For loop that iterates over basis sets
            for cardinals in CCSDTextrapaugcc_proj.cardinals:
                label=str(cardinal)
                CCSDTextrapaugcc_proj.labels.append(label)
                #Define theory level
                cc = ORCA_CC_CBS_Theory(elements=elements_involved, cardinals = cardinals, basisfamily="aug-cc", DLPNO=DLPNO, numcores=numcores, memory=memory)
                #Single-point calcs on all fragments
                energies=ash.Singlepoint_fragments(fragments=specieslist, theory=cc)
                for species,e in zip(specieslist,energies):
                    CCSDTextrapaugcc_proj.species_energies_dict[species.label].append(e)

                #Storing total energies of all species in dict
                CCSDTextrapaugcc_proj.energy_dict[label]=energies

                #Store reaction energy in dict
                reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit=energy_unit, label=reactionlabel, silent=False)
                CCSDTextrapaugcc_proj.reaction_energy_dict[label]=reaction_energy
                CCSDTextrapaugcc_proj.reaction_energy_list.append(reaction_energy)

            print()
            #Print results
            CCSDTextrapaugcc_proj.printall()




    ###################################################
    # Do CCSD(T)/CBS extrapolations with def2 family
    ###################################################
    if extrapolation is True:
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
            cc = ORCA_CC_CBS_Theory(elements=elements_involved, cardinals = cardinals, basisfamily="def2", DLPNO=DLPNO, numcores=numcores, memory=memory)
            #Single-point calcs on all fragments
            energies=ash.Singlepoint_fragments(fragments=specieslist, theory=cc)
            for species,e in zip(specieslist,energies):
                CCSDTextrapdef2_proj.species_energies_dict[species.label].append(e)

            #Storing total energies of all species in dict
            CCSDTextrapdef2_proj.energy_dict[label]=energies

            #Store reaction energy in dict
            reaction_energy, unused = ReactionEnergy(stoichiometry=stoichiometry, list_of_energies=energies, unit=energy_unit, label=reactionlabel, silent=False)
            CCSDTextrapdef2_proj.reaction_energy_dict[label]=reaction_energy
            CCSDTextrapdef2_proj.reaction_energy_list.append(reaction_energy)

        print()
        #Print results
        CCSDTextrapdef2_proj.printall()



    ################
    #Plotting
    ###############
    if plot is True:
        #Energy plot for each species:
        for species in specieslist:
            specieslabel=species.label
            eplot = ASH_plot('{} energy plot'.format(specieslabel), num_subplots=1, x_axislabel="Cardinal", y_axislabel='Energy (Eh)', title='{} Energy'.format(specieslabel))
            if cc_family is True:
                eplot.addseries(0, x_list=CCSDT_cc_bases_proj.cardinals, y_list=CCSDT_cc_bases_proj.species_energies_dict[specieslabel], label='cc-pVnZ', color='blue')
            if aug_cc_family is True:
                eplot.addseries(0, x_list=CCSDT_aug_cc_bases_proj.cardinals, y_list=CCSDT_aug_cc_bases_proj.species_energies_dict[specieslabel], label='aug-cc-pVnZ', color='lightblue')
            if def2_family is True:
                eplot.addseries(0, x_list=CCSDT_def2_bases_proj.cardinals, y_list=CCSDT_def2_bases_proj.species_energies_dict[specieslabel], label='def2-nVP', color='red')
            if F12_family is True:
                eplot.addseries(0, x_list=CCSDTF12_bases_proj.cardinals, y_list=CCSDTF12_bases_proj.species_energies_dict[specieslabel], label='cc-pVnZ-F12', color='purple')
            if extrapolation is True:
                eplot.addseries(0, x_list=[2.5], y_list=CCSDTextrap_proj.species_energies_dict[specieslabel][0], label='CBS-cc-23', line=False,  marker='x', color='gray')
                eplot.addseries(0, x_list=[3.5], y_list=CCSDTextrap_proj.species_energies_dict[specieslabel][1], label='CBS-cc-34', line=False, marker='x', color='green')
                if aug_cc_family is True:
                    eplot.addseries(0, x_list=[2.5], y_list=CCSDTextrapaugcc_proj.species_energies_dict[specieslabel][0], label='CBS-aug-cc-23', line=False,  marker='x', color='brown')
                    eplot.addseries(0, x_list=[3.5], y_list=CCSDTextrapaugcc_proj.species_energies_dict[specieslabel][1], label='CBS-aug-cc-34', line=False, marker='x', color='lightgreen') 
                if highest_cardinal > 4:
                    eplot.addseries(0, x_list=[4.5], y_list=CCSDTextrap_proj.species_energies_dict[specieslabel][2], label='CBS-cc-45', line=False, marker='x', color='black')
                    if aug_cc_family is True:
                        eplot.addseries(0, x_list=[4.5], y_list=CCSDTextrapaugcc_proj.species_energies_dict[specieslabel][2], label='CBS-aug-cc-45', line=False, marker='x', color='silver')
                    if highest_cardinal > 5:
                        eplot.addseries(0, x_list=[5.5], y_list=CCSDTextrap_proj.species_energies_dict[specieslabel][3], label='CBS-cc-56', line=False, marker='x', color='orange')
                        if aug_cc_family is True:
                            eplot.addseries(0, x_list=[5.5], y_list=CCSDTextrapaugcc_proj.species_energies_dict[specieslabel][3], label='CBS-aug-cc-56', line=False, marker='x', color='maroon')
                eplot.addseries(0, x_list=[2.5], y_list=CCSDTextrapdef2_proj.species_energies_dict[specieslabel][0], label='CBS-def2-23', line=False,  marker='x', color='cyan')
                eplot.addseries(0, x_list=[3.5], y_list=CCSDTextrapdef2_proj.species_energies_dict[specieslabel][1], label='CBS-def2-34', line=False, marker='x', color='pink')
            eplot.savefig('{}_Energy'.format(specieslabel))

        #Reaction energy plot
        reactionenergyplot = ASH_plot('{}'.format(reactionlabel), num_subplots=1, x_axislabel="Cardinal", y_axislabel='Energy ({})'.format(energy_unit), title='{}'.format(reactionlabel))
        if cc_family is True:
            reactionenergyplot.addseries(0, x_list = CCSDT_cc_bases_proj.cardinals, y_list=CCSDT_cc_bases_proj.reaction_energy_list, label='cc-pVnZ', color='blue')
        if aug_cc_family is True:
            reactionenergyplot.addseries(0, x_list=CCSDT_aug_cc_bases_proj.cardinals, y_list=CCSDT_aug_cc_bases_proj.reaction_energy_list, label='aug-cc-pVnZ', color='lightblue')
        if def2_family is True:
            reactionenergyplot.addseries(0, x_list = CCSDT_def2_bases_proj.cardinals, y_list=CCSDT_def2_bases_proj.reaction_energy_list, label='def2-nVP', color='red')
        if F12_family is True:
            reactionenergyplot.addseries(0, x_list = CCSDTF12_bases_proj.cardinals, y_list=CCSDTF12_bases_proj.reaction_energy_list, label='cc-pVnZ-F12', color='purple')
        if extrapolation is True:
            reactionenergyplot.addseries(0, x_list=[2.5], y_list=CCSDTextrap_proj.reaction_energy_list[0], label='CBS-cc-23', line=False,  marker='x', color='gray')
            reactionenergyplot.addseries(0, x_list=[3.5], y_list=CCSDTextrap_proj.reaction_energy_list[1], label='CBS-cc-34', line=False, marker='x', color='green')
            if aug_cc_family is True:
                reactionenergyplot.addseries(0, x_list=[2.5], y_list=CCSDTextrapaugcc_proj.reaction_energy_list[0], label='CBS-aug-cc-23', line=False,  marker='x', color='brown')
                reactionenergyplot.addseries(0, x_list=[3.5], y_list=CCSDTextrapaugcc_proj.reaction_energy_list[1], label='CBS-aug-cc-34', line=False, marker='x', color='lightgreen')
            if highest_cardinal > 4:
                reactionenergyplot.addseries(0, x_list=[4.5], y_list=CCSDTextrap_proj.reaction_energy_list[2], label='CBS-cc-45', line=False, marker='x', color='black')
                if aug_cc_family is True:
                    reactionenergyplot.addseries(0, x_list=[4.5], y_list=CCSDTextrapaugcc_proj.reaction_energy_list[2], label='CBS-aug-cc-45', line=False, marker='x', color='silver')
                if highest_cardinal > 5:
                    reactionenergyplot.addseries(0, x_list=[5.5], y_list=CCSDTextrap_proj.reaction_energy_list[3], label='CBS-cc-56', line=False, marker='x', color='orange')
                    if aug_cc_family is True:
                        reactionenergyplot.addseries(0, x_list=[5.5], y_list=CCSDTextrapaugcc_proj.reaction_energy_list[3], label='CBS-aug-cc-56', line=False, marker='x', color='maroon')
            reactionenergyplot.addseries(0, x_list=[2.5], y_list=CCSDTextrapdef2_proj.reaction_energy_list[0], label='CBS-def2-23', line=False, marker='x', color='cyan')
            reactionenergyplot.addseries(0, x_list=[3.5], y_list=CCSDTextrapdef2_proj.reaction_energy_list[1], label='CBS-def2-34', line=False, marker='x', color='pink')
        reactionenergyplot.savefig('Reaction energy')


#NOTE: not ready
def BrokenSymmetryCalculator(theory=None, fragment=None, Opt=False, flip_atoms=None, BS_flip_options=None, charge=None, mult=None):

    if theory == None or fragment == None or flip_atoms == None or BS_flip_options==None:
        print("Please set theory, fragment, flip_atoms and BS_flip_options keywords")
        exit()

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "BrokenSymmetryCalculator", theory=theory)

    #Getting full-system atom numbers for each BS-flip
    atomstoflip=[flip_atoms[i-1] for i in BSflip]
    orcaobject = ORCATheory(orcadir=orcadir, orcasimpleinput=ORCAinpline, orcablocks=ORCAblocklines,
                        brokensym=brokensym, HSmult=HSmult, atomstoflip=atomstoflip, nprocs=numcores, extrabasisatoms=extrabasisatoms,
                        extrabasis="ZORA-def2-TZVP")


    #Looping over BS-flips
    for BSflip in BS_flip_options:
        calclabel=f'mult{mult}_BSflip {"_".join(map(str,BSflip))}'

        #Making a copy of ORCAobject (otherwise BS-flip won't work)
        orcacalc = copy.copy(orcaobject)

        # QM/MM OBJECT
        #qmmmobject_trunc = QMMMTheory(qm_theory=orcacalc, mm_theory=openmmobject, fragment=frag, embedding="Elstat", qmatoms=qmatoms, printlevel=2,
        #    TruncatedPC=True, TruncPCRadius=55, TruncatedPC_recalc_iter=50)

        # QM/MM with no TruncPC approximation. No BS, will read previous orbital file
        #qmmmobject_notrunc = QMMMTheory(qm_theory=orcacalc, mm_theory=openmmobject, fragment=frag, embedding="Elstat", qmatoms=qmatoms, printlevel=2)


        #OPT with TruncPC approximation
        geomeTRICOptimizer(theory=theory, fragment=frag, ActiveRegion=True, actatoms=actatoms, maxiter=500, coordsystem='hdlc', charge=charge, mult=mult)

        #Preserve geometry
        os.rename('Fragment-optimized.xyz', f'Fragment_{calclabel}_truncopt.xyz')

        #Opt without TruncPC approximation
        geomeTRICOptimizer(theory=qmmmobject_notrunc, fragment=frag, ActiveRegion=True, actatoms=actatoms, maxiter=500, coordsystem='hdlc', charge=charge, mult=mult)

        #Preserve geometry and ORCA output
        os.rename('Fragment-optimized.xyz', f'Fragment_{calclabel}_notrunc.xyz')
        os.rename('orca.out', f'orca_{calclabel}_notrunc.out')
        os.rename('orca.gbw', f'orca_{calclabel}_notrunc.gbw')

        #create final pdb file
        write_pdbfile(frag, outputname=f'Fragment_BSflip_{calclabel}_notrunc',openmmobject=openmmobject)


#From total atomization energy (either 0 K or 298 K) calculate enthalpy of formation
def FormationEnthalpy(TAE, fragments, stoichiometry, RT=False, deltaHF_atoms_dict=None ):
    #From ATCT
    deltaHF_atoms_exp_0K={'H':51.6333652, 'C':170.02820267686425, 'N':112.4710803, 'O':58.99713193, 'F':18.46510516, 'Si':107.67925430210326, 'Cl':28.59010516, 'Br':28.18283939}
    deltaHF_atoms_exp_298K={'H':52.10277247, 'C':171.33914913957935, 'N':112.916348, 'O':59.56716061, 'F':18.96845124, 'Si':108.71414913957935, 'Cl':28.97418738, 'Br':26.73398662}
    print("\nFormationEnthalpy function")
    print("RT is:", RT)
    
    if RT is True:
        print("Assuming T=298.15 K. Using atomic experimental enthalpies of formation at 298.15 K.")
        print("deltaHF_atoms_exp_298K:", deltaHF_atoms_exp_298K)
        deltaHF_atoms_exp=deltaHF_atoms_exp_298K
    else:
        print("Assuming T=0 K. Using atomic experimental enthalpies of formation at 0 K.")
        print("deltaHF_atoms_exp_0K:", deltaHF_atoms_exp_0K)
        deltaHF_atoms_exp=deltaHF_atoms_exp_0K

    #Replace deltaHF_atoms_dict entries with user-values:
    if deltaHF_atoms_dict != None:
        print("Additional deltaHF dictionary provided. Replacing.")
        for i in deltaHF_atoms_dict.items():
            deltaHF_atoms_exp[i[0]]=i[1]


    print("deltaHF_atoms_exp:", deltaHF_atoms_exp)

    #Looping over fragments and stoichiometry lists
    sum_of_exp_Hf_atoms=0.0
    for frag,stoich in zip(fragments,stoichiometry):
        if len(frag.elems) == 1:
            sum_of_exp_Hf_atoms+=(deltaHF_atoms_exp[frag.elems[0]])*stoich
    print("sum_of_exp_Hf_atoms:", sum_of_exp_Hf_atoms)
    print()
    deltaH_form=sum_of_exp_Hf_atoms-TAE
    if RT is True:
        print("DeltaH_form (298 K):", deltaH_form)
    else:
        print("DeltaH_form (0 K):", deltaH_form)
    return deltaH_form







#Finding non-Aufbau configurations via STEP levelshifting. Using either orbital occupation ordering for finding states or TDDFT
#TODO: extra_multiplicities.
def AutoNonAufbau(fragment=None, theory=None, num_occ_orbs=1, num_virt_orbs=3, spinset=[0], stability_analysis_GS=False, TDDFT=False, epsilon=0.1, maxiter=500, 
    manual_levelshift=None):

    print_line_with_mainheader("AutoNonAufbau")

    if isinstance(theory,ash.ORCATheory) is False:
        print("AutoNonAufbau only works with ORCATheory objects")
        exit()

    #NOTE: Skipping multiplicities for now as too complicated and inconsistent
    #if multiplicities not set
    #if multiplicities == None:
    #    print("Keyword argument multiplicities not set.")
    #    print("Looking for spin multiplicity in fragment")
    #    if fragment.mult == None:
    #        print("No mult in fragment either")
    #        ashexit()
    #    print("Found spin multiplicity: ", fragment.mult)

    #Keeping option for multiplicities for now
    multiplicities=[fragment.mult]
    #print("Spin multiplicities to calculate", multiplicities)
    print("Spin orbital sets to choose:", spinset)

    #The number of states
    print("Number of occupied orbitals allowed in MO swap:", num_occ_orbs)
    print("Number of virtual orbitals allowed in MO swap:", num_virt_orbs)
    totalnumstates=num_occ_orbs*num_virt_orbs*len(spinset)
    print("Total number of states:", totalnumstates)
    print("TDDFT:", TDDFT)
    print("stability_analysis_GS:", stability_analysis_GS)
    print("Epsilon:", epsilon)
    print("Manual levelshift:", manual_levelshift)


    #epsilon is shift parameter from Herbert paper. 0.1 was recommended

    #Removing old orbital files
    theory.cleanup()

    #First the regular calculation
    print("Now doing initial state SCF calculation")
    originalblocks=copy.copy(theory.orcablocks)
    if TDDFT is True:
        print("Doing TDDFT on top as well")
        blocks_with_ttddft=theory.orcablocks+f"%tddft nroots {totalnumstates} end"
        theory.orcablocks=blocks_with_ttddft
    
    #If stability_analysis_GS is True then do on GroundState
    if stability_analysis_GS == True:
        theory.orcablocks=theory.orcablocks+"%scf stabperform true end"
    E_GS=ash.Singlepoint(fragment=fragment, theory=theory)

    #Keep copy of GS file
    GS_GBW=theory.filename+'_GS.gbw'
    shutil.copy(theory.filename+'.gbw', GS_GBW)
    shutil.copy(theory.filename+'.out', theory.filename+'_GS.out')

    #Now removing TDDFT or stability input from blocks if done
    theory.orcablocks=originalblocks

    #Read TDDFT output if requested
    if TDDFT is True:
        #TDDFT transition energies
        transition_energies = ash.interfaces.interface_ORCA.tddftgrab(theory.filename+".out")
        print("TDDFT transition_energies (eV):", transition_energies)
        #Making sure to take GS energy not from FINAL SINGLEPOINT ENERGY since ORCA annoyingly adds the transition energy
        E_GS = float(pygrep("E(SCF)  =",theory.filename+'.out')[2])
        tddft_pairs = ash.interfaces.interface_ORCA.tddft_orbitalpairs_grab(theory.filename+".out")
        print("TDDFT orbital pairs per state:", tddft_pairs)

    #Read MO-energies
    mo_dict = ash.interfaces.interface_ORCA.MolecularOrbitalGrab(theory.filename+".out")
    print("mo_dict:", mo_dict)

    #MO numbers of the HOMO and the first LUMO for alpha and beta sets
    homo_number_alpha=len(mo_dict["occ_alpha"])-1
    lumo_number_alpha=len(mo_dict["occ_alpha"])
    homo_number_beta=len(mo_dict["occ_beta"])-1
    lumo_number_beta=len(mo_dict["occ_beta"])

    #############################
    # Defining excited states
    #############################

    # Class to define SCF configuration based on reference SCF configuration
    class NonAufbauState:
        def __init__(self,charge,mult,spinset,original_homo_index, occ_index, virt_index, mos_occ_energies, mos_unocc_energies):

            self.charge=charge
            self.mult=mult
            self.spinset=spinset

            #All MO energies combined
            self.all_mo_energies=mos_occ_energies+mos_unocc_energies

            #These are the original HOMO/LUMO numbers of the ground-state CFG
            self.original_homo_index=original_homo_index

            #Occ-orbital index to remove electron from
            self.occ_orb_index=occ_index
            #Virtual-orbital index to add electron to
            self.virt_orb_index=virt_index

            #Chosen Occ-orb MO energy
            self.occ_orb_energy=self.all_mo_energies[self.occ_orb_index]
            #Chosen Virt-orb MOenergy
            self.virt_orb_energy=self.all_mo_energies[self.virt_orb_index]
            #Chosen Occ-Virt Gap
            self.gap=self.occ_orb_energy-self.virt_orb_energy

            #Storing converged SCF energy later
            self.energy=None

    #States to calculate
    calculated_states=[]



    #Looping over num occupied orbitals chosen
    s_ind=1
    for o in range(0,num_occ_orbs):
        #Looping over num occupied orbitals chosen
        for v in range(0,num_virt_orbs):
            #Looping over alpha/bets sets chosen
            for s in spinset:
                if s == 0:
                    occ_index=homo_number_alpha-o
                    virt_index=lumo_number_alpha+v
                    cfg=NonAufbauState(fragment.charge, fragment.mult, s, homo_number_alpha, occ_index, virt_index, mo_dict["occ_alpha"], mo_dict["unocc_alpha"])
                elif s == 1:
                    occ_index=homo_number_beta-o
                    virt_index=lumo_number_beta+v
                    cfg=NonAufbauState(fragment.charge, fragment.mult, s, homo_number_beta, occ_index, virt_index, mo_dict["occ_beta"], mo_dict["unocc_beta"])
                calculated_states.append((s_ind,cfg))
                s_ind+=1       

    print("Calculated_states (index,spinset):", calculated_states)

    states_dict={}
    print()

    #Looping over how many states we want
    for state in calculated_states:
        #State: (index,mult)
        state_index=state[0]
        cfg=state[1]
        print("="*90)
        print(f"Now running excited state SCF no. {state_index} with multiplicity: {cfg.mult} and spinset {cfg.spinset}")
        print("="*90)
        #Choosing what orbital to rotate. Using TDDFT or just looping over orbitals
        if TDDFT is True:
            print("TDDFT selection scheme active!")
            orbpairs=tddft_pairs[state_index+1]
            print("TDDFT excitation orbital pairs:", orbpairs)
            weights=[i[2] for i in orbpairs]
            print("weights:", weights)
            largest_indices=n_max_values(weights,num=4)
            print("largest_indices:", largest_indices)
            maxweight=weights[largest_indices[0]]
            maxweight_2nd=weights[largest_indices[1]]
            maxweight_index=largest_indices[0]
            maxweight_index_2nd=largest_indices[1]
            print("maxweight:", maxweight)
            print("maxweight_2nd:", maxweight_2nd)
            #If one orbital pair has majority then we only use that one
            if maxweight > 0.5:
                multiple_pairs=False
                orbpairA=orbpairs[maxweight_index]
                print("Maximum weight TDDFT excitation orbital pair:", orbpairA)
            #Otherwise we take 2 that are probably a joint alpha-beta excitation pair
            #If more contribute then it's probably too complicated to find this way anyway
            else:
                multiple_pairs=True
                orbpairA=orbpairs[maxweight_index]
                orbpairB=orbpairs[maxweight_index_2nd]
                print("Large weight TDDFT excitation orbital pair:", orbpairA)
                print("Large weight TDDFT excitation orbital pair:", orbpairB)

            #Setting spin based on label in TDDFT excitation
            #And getting orbital number
            if 'a' in orbpairA[0]:
                spinvar_A=0
                orbpair_occ_A=int(orbpairA[0].replace("a",""))
                orbpair_unocc_A=int(orbpairA[1].replace("a",""))
            elif 'b' in orbpairA[0]:
                spinvar_A=1
                orbpair_occ_A=int(orbpairA[0].replace("b",""))
                orbpair_unocc_A=int(orbpairA[1].replace("b",""))
            #Defining MO rotation line for first pair
            print(f"Will rotate occupied orbitals: {orbpair_occ_A} and virtual orbital: {orbpair_unocc_A} in spin manifold {spinvar_A}")
            rotatelineA=f"rotate {{{orbpair_occ_A},{orbpair_unocc_A},90,{spinvar_A},{spinvar_A}}} end"
            
            #List of of occorbs
            occ_orb_list=[orbpair_occ_A]
            unocc_orb_list=[orbpair_unocc_A]
            spinvar_list=[spinvar_A]
            #If multiple pairs then define the second one, otherewise set to empty string
            if multiple_pairs is True:
                if 'a' in orbpairB[0]:
                    spinvar_B=0
                    orbpair_occ_B=int(orbpairB[0].replace("a",""))
                    orbpair_unocc_B=int(orbpairB[1].replace("a",""))
                elif 'b' in orbpairB[0]:
                    spinvar_B=1
                    orbpair_occ_B=int(orbpairB[0].replace("b",""))
                    orbpair_unocc_B=int(orbpairB[1].replace("b",""))
                print(f"Will rotate occupied orbitals: {orbpair_occ_B} and virtual orbital: {orbpair_unocc_B} in spin manifold {spinvar_B}")
                rotatelineB=f"rotate {{{orbpair_occ_B},{orbpair_unocc_B},90,{spinvar_B},{spinvar_B}}} end"
                #Adding to list
                occ_orb_list.append(orbpair_occ_B)
                unocc_orb_list.append(orbpair_unocc_B)
                spinvar_list.append(spinvar_B)
            else:
                rotatelineB=""
            

            #NOTE: Need to define HOMO-LUMO gap here for levelshift
            print("NOT READY")
            ashexit()



        else:
            print("Simple MO selection scheme")

            #Getting HOMO-LUMO gap for orbitals involved
            homo_lumo_gap=cfg.gap/27.211399
            print("homo_lumo_gap:", homo_lumo_gap)
            print(f"Will rotate orbital {cfg.occ_orb_index} (HOMO) and orbital {cfg.virt_orb_index}")

            #Defining rotatelines
            rotatelineA=f"rotate {{{cfg.occ_orb_index},{cfg.virt_orb_index},90,{cfg.spinset},{cfg.spinset}}} end"
            #Possible 2nd rotation line. Not used for simple MO selecting scheme.
            #TODO: Use for double excitation later
            rotatelineB=""
            #List of of occorb/unoccorb lists
            occ_orb_list=[cfg.occ_orb_index]
            unocc_orb_list=[cfg.virt_orb_index]
            spinvar_list=[cfg.spinset]
        #Lshift
        if manual_levelshift != None:
            print("Manual levelshift option chosen.")
            print("Using levelshift: {} Eh".format(manual_levelshift))
            lshift=manual_levelshift
        else:
            lshift=abs(homo_lumo_gap)+epsilon
        print("Levelshift:", lshift)

        #Rotating orb
        theory.extraline=f"""!Normalprint  nodamp
    %scf
    maxiter {maxiter}
    {rotatelineA}
    {rotatelineB}
    cnvshift true
    lshift {lshift}
    shifterr 0.0005
    end
        """
        if 'notrah' not in theory.orcasimpleinput:
            theory.extraline=theory.extraline+"\n!notrah"
        #Now new SP with previous orbitals, SCF rotation and levelshift
        theory.moreadfile=GS_GBW
        print("Now doing SCF calculation with rotated MOs and levelshift")
        E_ES=ash.Singlepoint(fragment=fragment, theory=theory, charge=cfg.charge, mult=cfg.mult)
        print("GS/ES state energy gap: {:3.2f} eV".format((E_ES-E_GS)*27.211399))
        if abs((E_ES-E_GS)*27.211399) > 0.04:
            print("Found something different than ground state")
            if (E_ES-E_GS)*27.211399 > 0.04:
                print("Converged SCF energy higher than ground-state SCF. Found new excited state SCF solution !")
            elif (E_ES-E_GS)*27.211399 < 0.04:
                print((E_ES-E_GS)*27.211399)
                print("Converged SCF energy lower than initial state SCF. Looks like we found a new groundstate!")
        else:
            print("GS/ES state energy gap smaller than 0.04 eV. Presumably found the original SCF again.")
        #Keeping copy of outputfile and GBW file
        shutil.copy(theory.filename+'.out', theory.filename+f'ES_SCF{state_index}_mult{cfg.mult}_spinset{cfg.spinset}.out')
        shutil.copy(theory.filename+'.gbw', theory.filename+f'ES_SCF{state_index}_mult{cfg.mult}_spinset{cfg.spinset}.gbw')

        #excited_state_energies.append(E_ES)
        #Adding all info to dict
        states_dict[state_index] = [E_ES,occ_orb_list,unocc_orb_list,spinvar_list,homo_lumo_gap,lshift, cfg.mult, theory.filename+f'ES_SCF{state_index}_mult{cfg.mult}_spinset{cfg.spinset}.gbw' ] 

    #Collecting energies
    print()
    print("states_dict:", states_dict)
    print()
    print(f"Ground-state SCF energy: {E_GS} Eh")
    if TDDFT is True:
        print("TDDFT transition energies:", transition_energies)

    print("-"*5)
    final_state_dict={}
    uniquestatecount=1
    for state in calculated_states:
        state_index=state[0]
        cfg=state[1]
        print("Excited state index:", state_index)
        print("Spin multiplicity:", cfg.mult)
        print(f"Excited-state SCF energy {state_index}: {states_dict[state_index][0]} Eh")
        print(f"Excited-state SCF transition energy: {(states_dict[state_index][0]-E_GS)*27.211399:3.2f} eV")
        print(f"Excited-state SCF orbital rotation: Occ:{states_dict[state_index][1]} Virt:{states_dict[state_index][2]}")
        print(f"Rotation in spin manifold: ", states_dict[state_index][3])
        print(f"Excited-state SCF HOMO-LUMO gap: {states_dict[state_index][4]} Eh ({states_dict[state_index][4]*27.211399}) eV: ")
        print(f"Excited-state SCF Levelshift chosen: {states_dict[state_index][5]}")
        print("-"*5)

        #Only keep states that did not fall back to GS. 
        # Currently keeping states with identical energy as they could be a member of a degenerate state.
        if (states_dict[state_index][0]-E_GS)*27.211399 > 0.04:
            final_state_dict[uniquestatecount] = states_dict[state_index]
            uniquestatecount+=1

    #Adding ground-state to state-dictionary with key 0
    states_dict[0]=[E_GS, [0], [0], [0], 0.0, 0.0, fragment.mult, GS_GBW]
    final_state_dict[0]=[E_GS, [0], [0], [0], 0.0, 0.0, fragment.mult, GS_GBW]

    print()
    print("Final list of SCF states (excluding calculations that fell back to Groundstate):")
    print()
    print(final_state_dict)

    return final_state_dict

#Geometry Optimizer for excited-state SCF solutions with ORCA
#NOTE: Requires dictionary result from AutoNonAufbau
#TODO: Future, wr
def ExcitedStateSCFOptimizer(theory=None, fragment=None, autononaufbaudict=None, state=1, charge=None, mult=None, maxiter=500, Freq=False, extrashift=0.0):

    print_line_with_mainheader("ExcitedStateOptimizer")

    if isinstance(theory,ash.ORCATheory) is False:
        print("ExcitedStateOptimizer only works with ORCATheory objects")
        exit()
    print("AutoNonAufbau dictionary:", autononaufbaudict)
    print("State chosen:", state)
    #Get GBW-file for chosen state
    gbwfile=autononaufbaudict[state][7]
    path=os.getcwd()
    theory.moreadfile=path+'/'+gbwfile
    print("theory.moreadfile:", theory.moreadfile)
    #Modify theory level shift for chosen state
    lshift=autononaufbaudict[state][5]+extrashift

    theory.extraline=f"""!Normalprint  nodamp
    %scf
    maxiter {maxiter}
    cnvshift true
    lshift {lshift}
    shifterr 0.0005
    end
        """.format(maxiter, lshift )
    print("Will now run optimization on excited state no:", state)

    optenergy = ash.geomeTRICOptimizer(theory=theory, fragment=fragment, charge=charge, mult=mult)

    #Excited state frequencies
    if Freq:
        print("Freq true. Doing excited state numerical frequencies")
        thermochem = ash.NumFreq(fragment=fragment, theory=theory, npoint=2, runmode='serial', charge=charge, mult=mult)
