import os
import shutil
import subprocess as sp

from ash.modules.module_coords import check_charge_mult,xyz2coord
from ash.interfaces.interface_geometric import geomeTRICOptimizer
from ash.modules.module_freq import NumFreq
from ash.modules.module_singlepoint import Singlepoint
from ash.functions.functions_general import print_line_with_subheader1,BC,ashexit,writestringtofile,isodd
import ash.settings_ash
#Various electron chemistry
#DEA/DI

#Get lowest multiplicity 
def lowest_mult(mult):
    if isodd(mult):
        return 1
    else:
        return 2

def ElectronImpact(fragment=None, theory=None, HLtheory=None, charge=None, mult=None, qcxmsdir=None, xtbmethod=None, numcores=1):

    #TODO: THIS IS NOT YET READY
    print_line_with_subheader1("ElectronImpact")

    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Electron impact", theory=theory)
    
    print()
    #######################
    # Initial species
    #######################
    print("-"*50)
    print(f"INITIAL MOLECULAR SPECIES charge: {charge} mult: {mult}")
    print("-"*50)
    print("Now starting optimization")
    geomeTRICOptimizer(theory=theory, fragment=fragment, maxiter=500, coordsystem='hdlc', charge=charge, mult=mult)
    print("Now starting Numerical frequencies")
    thermochem_neutral = NumFreq(fragment=fragment, theory=theory, npoint=2, runmode='serial', charge=charge, mult=mult)

    ZPVE=thermochem_neutral["ZPVE"]
    vibenergycorr=thermochem_neutral['vibenergycorr']
    E_vib=thermochem_neutral['E_vib']
    print("Now starting High-level singlepoint")
    HL_neut = Singlepoint(fragment=fragment, theory=HLtheory)
    HL_energy_neut = HL_neut.energy
    #######################
    #Direct Cation species
    #######################
    cation_charge=charge-1
    cation_mult=lowest_mult(mult+1) #Here choosing lowest spin multiplicity
    print("-"*50)
    print(f"CATION MOLECULAR SPECIES charge: {charge} mult:{mult}")
    print("-"*50)
    print("Now starting optimization")
    geomeTRICOptimizer(fragment=fragment, theory=theory, maxiter=500, coordsystem='hdlc', charge=cation_charge, mult=cation_mult)
    #print("Now starting Numerical frequencies")
    HL_cation = Singlepoint(fragment=fragment, theory=HLtheory)
    HL_energy_cation = HL_cation.energy
    #thermochem = NumFreq(fragment=fragment, theory=theory, npoint=2, runmode='serial', charge=cation_charge, mult=cation_mult)
    #ZPVE_cation=thermochem["ZPVE"]
    #vibenergycorr=thermochem['vibenergycorr']
    #E_vib=thermochem['E_vib']


    #######################
    #FRAGMENTATIONS
    #######################

    #QCEIMS to get fragments
    print("\nNow calling QCxMS")
    #TODO: THIS IS NOT YET READY
    unique_fragments_reactions = call_QCxMS(fragment=fragment, qcxmsdir=qcxmsdir, xtbmethod=xtbmethod, charge=charge, mult=mult, numcores=numcores, ntraj=300, tmax=10, tinit=500, ieeatm=0.6)

    print("Now reading unique_fragments_reactions:", unique_fragments_reactions)

    #unique_fragments_reactions should contain unique fragmentation reactions from QCxMS
    #TODO: THIS IS NOT YET READY
    #Looping over reactions and then fragments found by MD. Opt+Freq+HL
    molfrag_e_dict={}
    for reaction in unique_fragments_reactions:
        print("This is reaction:", reaction)
        for molfrag in reaction:
            label=molfrag.label #Label should contain formula, charge, spin and something extra
            geomeTRICOptimizer(theory=theory, fragment=molfrag, maxiter=500, coordsystem='hdlc', charge=molfrag.charge, mult=molfrag.mult)
            thermochem = NumFreq(fragment=molfrag, theory=theory, npoint=2, runmode='serial', charge=molfrag.charge, mult=molfrag.mult)

            HL_molfrag = Singlepoint(fragment=fragment, theory=HLtheory)
            HL_energy_molfrag = HL_molfrag.energy
            molfrag_e_dict[label] = HL_energy_molfrag



    #######################
    #FINAL RESULTS
    #######################

    #Collect optimized geometries together in a directory

    #Final table



#Interface to Grimme QCxMS program

#Very simple QCxMS interface
def call_QCxMS(fragment=None, qcxmsdir=None, xtbmethod=None, charge=None, mult=None, numcores=1, ntraj=300, tmax=10, tinit=500, ieeatm=0.6):
    print_line_with_subheader1("call_QCxMS")
    if qcxmsdir == None:
        print(BC.WARNING, "No qcxmsdir argument passed to call_crest. Attempting to find qcxmsdir variable inside settings_ash", BC.END)
        try:
            print("ash.settings_ash.settings_dict:", ash.settings_ash.settings_dict)
            qcxmsdir=ash.settings_ash.settings_dict["qcxmsdir"]
        except:
            print(BC.WARNING,"Found no qcxmsdir variable in settings_ash module either.",BC.END)
            try:
                qcxmsdir = os.path.dirname(shutil.which('qcxms'))
                print(BC.OKGREEN,"Found qcxms in path. Setting qcxmsdir to:", qcxmsdir, BC.END)
            except:
                print("Found no qcxms executable in path. Exiting... ")
                ashexit()
    
    print("qcxmsdir:", qcxmsdir)
    #qcxmsdir should contain: qcxms, pqcxms q-batch, getres

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "call_QCxMS", theory=None)

    try:
        shutil.rmtree('qcxms-calc')
    except:
        pass
    os.mkdir('qcxms-calc')
    os.chdir('qcxms-calc')

    #Options: xtb for GFN1 xtb2 for GFN2
    if 'GFN1' in xtbmethod:
        xtbkw='xtb'
    elif 'GFN2' in xtbmethod:
        xtbkw='xtb2'
    else:
        print("Unknown xtbmethod", xtbmethod)
        exit()

    #Create inputfile: qcxms.in
    #TODO: To be extended. See manual: https://xtb-docs.readthedocs.io/en/latest/qcxms_doc/qcxms_run.html
    inputstring="""{}
ntraj {}  
tmax  {}    
tinit {} 
ieeatm {}""".format(xtbkw,ntraj, tmax, tinit, ieeatm)

    writestringtofile(inputstring, "qcxms.in")

    #Write XYZ file and then coord file
    fragment.write_xyzfile(xyzfilename="test.xyz", writemode='w')
    xyz2coord("test.xyz") #Writes coord file

    #Call program. Generates the ground-state trajectory
    process = sp.run([qcxmsdir + '/qcxms'])
    #Writes files: trjM, qcxms.gs and xtb.last

    #Call program 2nd time to set up all ionized trajectories. Reads qcxms.gs file from previous
    process = sp.run([qcxmsdir + '/qcxms'])
    #Writes dir: TMPQCXMS containing: TMP.1, TMP.2 etc. Each TMP. dir contains coord (same), qcxms.in, qcxms.start

    #Call program to run ionized trajectories

    #Setting OMP threads, equal to qcxms runs. Each run with 1 core though
    os.environ["OMP_NUM_THREADS"] = str(numcores)
    process = sp.run([qcxmsdir + '/pqcxms', '-t', 1]) #-j option did not work last time
    
    #Process the output. Define reactions and fragments associated

    #Each TMP dir contains fragments labelled like this: e.g. 1.1.xyz    or 1.1.xyz, 1.2.xyz, 1.3.xyz    qcxms.out, qcxms.res, ion.out, neutral.out

    #TODO: Need to go through each sim-run, read fragments found, get charge/mult for each and process what happened

    #TODO: Go through fragments and compare what is unique and what is not. Compare formula, charge, spin and rmsd-alignment.

    #Create dictionary of reactions and fragment. not sure of structure
    unique_fragments_reactions={}


    return unique_fragments_reactions
#Other ideas:

#OpenMM MD instead of QCEIXMS
#crest for each fragment as extra option
#Excited states. Test on Mo(CO)6