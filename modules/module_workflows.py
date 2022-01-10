"""
Contains functions defining multi-step workflows

"""
import os
import subprocess as sp
import shutil
import time
import ash
#from ash import Singlepoint
import interfaces.interface_geometric
import interfaces.interface_crest
from functions.functions_general import BC,print_time_rel,print_line_with_mainheader,pygrep
#from modules.module_singlepoint import Singlepoint_fragments
from modules.module_highlevel_workflows import CC_CBS_Theory
from modules.module_coords import read_xyzfiles
import functions.functions_elstructure
from modules.module_plotting import ASH_plot
from modules.module_singlepoint import ReactionEnergy

#Simple class to keep track of results
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
    interfaces.interface_crest.call_crest(fragment=fragment, xtbmethod=xtbmethod, crestdir=crestdir, charge=charge, mult=mult, numcores=numcores)

    #2. Grab low-lying conformers from crest_conformers.xyz as list of ASH fragments.
    list_conformer_frags, xtb_energies = interfaces.interface_crest.get_crest_conformers()

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
        interfaces.interface_geometric.geomeTRICOptimizer(fragment=conformer, theory=MLtheory, coordsystem='tric')
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
                       analyticHessian=True, temp=298.15, pressure=1.0):
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
        interfaces.interface_geometric.geomeTRICOptimizer(theory=Opt_theory,fragment=fragment)
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

    SP_theory.charge = fragment.charge
    SP_theory.mult = fragment.mult
    FinalE = ash.Singlepoint(fragment=fragment, theory=SP_theory)
    #Get energy components
    if isinstance(SP_theory,CC_CBS_Theory):
        componentsdict=SP_theory.energy_components

    SP_theory.cleanup()

    print_time_rel(module_init_time, modulename='thermochemprotocol_single', moduleindex=0)
    return FinalE, componentsdict, thermochem


#Thermochemistry protocol. Take list of fragments, stoichiometry, and 2 theory levels
#Requires orcadir, and Opt_theory level (typically an ORCATheory object), SP_theory (either ORCATTheory or workflow.
def thermochemprotocol_reaction(Opt_theory=None, SP_theory=None, fraglist=None, stoichiometry=None, orcadir=None, numcores=1, memory=5000,
                       analyticHessian=True, temp=298.15, pressure=1.0):
    """[summary]

    Args:
        Opt_theory (ASH theory, optional): ASH theory for optimizations. Defaults to None.
        SP_theory (ASH theory, optional): ASH theory for Single-points. Defaults to None.
        fraglist (list, optional): List of ASH fragments. Defaults to None.
        stoichiometry (list, optional): list of integers defining stoichiometry. Defaults to None.
        orcadir (str, optional): Path to ORCA. Defaults to None.
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
                       analyticHessian=analyticHessian, temp=temp, pressure=pressure)
        
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
    if 'E_corrCCSD_CBS' in componentsdict:
        ccsd_parts=[dict['E_corrCCSD_CBS'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ccsd_parts, unit='kcalpermol', label='ΔCCSD')
    if 'E_corrCCT_CBS' in componentsdict:
        triples_parts=[dict['E_corrCCT_CBS'] for dict in list_of_dicts]
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
    if 'E_FCIcorrection' in componentsdict:
        fcicorr_parts=[dict['E_FCIcorrection'] for dict in list_of_dicts]
        ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=fcicorr_parts, unit='kcalpermol', label='ΔFCIcorr')
    print("")
    print(BC.WARNING, BC.BOLD, "------------THERMOCHEM PROTOCOL END-------------", BC.END)
    print_time_rel(module_init_time, modulename='thermochemprotocol_reaction', moduleindex=0)



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
                                    numcores=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment)
        init_orbitals=ORCAcalc_1.filename+'.mp2nat'

        step1occupations=ash.interfaces.interface_ORCA.MP2_natocc_grab(ORCAcalc_1.filename+'.out')
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
                                    numcores=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment)
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
                                    numcores=numcores)
        ash.Singlepoint(theory=ORCAcalc_1,fragment=fragment)
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
    numelectrons,numorbitals=functions.functions_elstructure.select_space_from_occupations(step1occupations, selection_thresholds=[upper_threshold,lower_threshold])
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
                                numcores=numcores)
    ash.Singlepoint(theory=ORCAcalc_2,fragment=fragment)

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


    minimal_CAS=functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.95,0.05])
    medium1_CAS=functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.98,0.02])
    medium2_CAS=functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.985,0.015])
    medium3_CAS=functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.99,0.01])
    medium4_CAS=functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.992,0.008])
    large_CAS=functions.functions_elstructure.select_space_from_occupations(ICEnatoccupations, selection_thresholds=[1.995,0.005])

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
        exit()
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

    #Looping through XYZ-files to get fragments
    fragments =read_xyzfiles(xyzdir,readchargemult=readchargemult, label_from_filename=True)

    #List of original filenames
    filenames=[frag.label for frag in fragments]

    #Now looping over fragments
    for fragment in fragments:
        filename=fragment.label
        print("filename:", filename)
        #Charge/mult from fragment
        if charge == None and mult ==None:
    	    theory.charge=fragment.charge; theory.mult=fragment.mult
        #Global charge/mult keywords (rare)
        else:
            theory.charge=charge; theory.mult=mult; fragment.charge=charge; fragment.mult=mult
        #Do Optimization or Singlepoint
        if Opt is True:
            if xtb_preopt is True:
                print("xTB Pre-optimization is active. Will first call xTB directly to do pre-optimization before actual optimization!")
                #Defining temporary xtBtheory
                xtbcalc=ash.xTBTheory(charge=theory.charge,mult=theory.mult, numcores=theory.numcores)
                #Run direct xtb optimization. This will update fragment.
                xtbcalc.Opt(fragment=fragment)
                xtbcalc.cleanup()
            
            #Now doing actual OPT
            optenergy = interfaces.interface_geometric.geomeTRICOptimizer(theory=theory, fragment=fragment, coordsystem='tric')
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
                thermochem = ash.NumFreq(fragment=fragment, theory=theory, npoint=2, displacement=0.005, numcores=theory.numcores, runmode='serial')
                theory.cleanup()
            if Highlevel is True:
                print("Performing Highlevel on Optimized fragment.")
                hlenergy = ash.Singlepoint(theory=HL_theory, fragment=fragment)
                energy=hlenergy
                hlenergies.append(hlenergy)
                HL_theory.cleanup()
        else:
            energy = ash.Singlepoint(theory=theory, fragment=fragment)
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
            exit()
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
            cc = CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="def2", DLPNO=DLPNO, numcores=numcores, memory=memory)
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
            cc = CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="cc", DLPNO=DLPNO, numcores=numcores, memory=memory)
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
            cc = CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="aug-cc", DLPNO=DLPNO, numcores=numcores, memory=memory)
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
            cc = CC_CBS_Theory(elements=elements_involved, cardinals = [cardinal], basisfamily="cc-f12", F12=True, DLPNO=DLPNO, numcores=numcores, memory=memory)
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
            cc = CC_CBS_Theory(elements=elements_involved, cardinals = cardinals, basisfamily="cc", DLPNO=DLPNO, numcores=numcores, memory=memory)
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
                cc = CC_CBS_Theory(elements=elements_involved, cardinals = cardinals, basisfamily="aug-cc", DLPNO=DLPNO, numcores=numcores, memory=memory)
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
            cc = CC_CBS_Theory(elements=elements_involved, cardinals = cardinals, basisfamily="def2", DLPNO=DLPNO, numcores=numcores, memory=memory)
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