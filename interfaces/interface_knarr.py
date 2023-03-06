#Non-intrusive interface to Knarr
#Assumes that Knarr directory exists inside ASH
import numpy as np
import sys
import os
import copy
import shutil
import time


import ash
import ash.constants as constants
from ash.functions.functions_general import ashexit,print_time_rel,print_line_with_mainheader, BC,print_line_with_subheader1,print_line_with_subheader2
from ash.modules.module_coords import check_charge_mult, write_xyzfile
from ash.modules.module_freq import write_hessian, approximate_full_Hessian_from_smaller, calc_model_Hessian_ORCA, read_tangent, calc_hessian_xtb
from ash.modules.module_results import ASH_Results

#This makes Knarr part of python path
#Recommended way?
ashpath = os.path.dirname(ash.__file__)
sys.path.insert(0,ashpath+'/knarr')

from KNARRio.system_print import PrintHeader, PrintDivider, PrintCredit
from KNARRatom.utilities import InitializeAtomObject, InitializePathObject, RMS, RMS3
from KNARRjobs.path import DoPathInterpolation
from KNARRio.io import ReadTraj
from KNARRjobs.neb import DoNEB
import KNARRatom.atom

#LOG of Knarr-code modifications
#1. Various python2 print-statements to print-functions changes
#2. Various additions of int() in order to get integer of division products (Python2/3 change)
#3. Made variable  calculator.ISCION = True . Bad idea?

#Knarr settings for path-generation, NEB and optimizer
#These will be the reasonable defaults that can be overridden by special keywords in ASH NEB object
#RB modified springconst from 10 to 5
# Changed "IDPP_RMS_F": 0.005    and "IDPP_MAX_F": 0.01
#Changed again: was: "IDPP_MAX_F": 0.03, "IDPP_RMS_F": 0.005
path_parameters = {"METHOD": "DOUBLE", "INTERPOLATION": "IDPP", "NIMAGES": 8,
              "INSERT_CONFIG": None, "IDPP_MAX_ITER": 700,
              "IDPP_SPRINGCONST": 5.0, "IDPP_TIME_STEP": 0.01,
              "IDPP_MAX_MOVE": 0.1, "IDPP_MAX_F": 0.07, "IDPP_RMS_F": 0.009}

neb_settings = {"PATH": "neb.xyz",
              "CLIMBING": True,
              "TANGENT": "IMPROVED",
              "SPRINGTYPE": "DISTANCE",
              "PERP_SPRINGTYPE": "",
              "ENERGY_WEIGHTED": True, "SPRINGCONST": 1.0, "SPRINGCONST2": 10.0,
              "MIN_RMSD": True, "REMOVE_EXTERNAL_FORCE": True,
              "FREE_END": False, "FREE_END_TYPE": 'PERP', "FREE_END_ENERGY": 0.0,
              "FREE_END_ENERGY2": 0.0, "FREE_END_KAPPA": 0.0,
              "CONV_TYPE": "ALL", "TOL_SCALE": 10, "TOL_MAX_FCI": 0.026, "TOL_RMS_FCI": 0.013,
              "TOL_MAX_F": 0.26, "TOL_RMS_F": 0.13, "TOL_TURN_ON_CI": 1.0,
              "ZOOM": False,
              "TOL_TURN_ON_ZOOM": 0.5,
              "REPARAM": 0, "TOL_REPARAM": 0.0,
              "INTERPOLATION_TYPE": "LINEAR",
              "AUTO_ZOOM": True, "ZOOM_OFFSET": 1, "ZOOM_ALPHA": 0.5,
              "RESTART_OPT_ON_CI": True,
              "LBFGS_REPARAM_ON_RESTART": False
              }

optimizer = {"OPTIM_METHOD": "LBFGS", "MAX_ITER": 200, "TOL_MAX_FORCE": 0.01,
             "TOL_RMS_FORCE": 0.005, "TIME_STEP": 0.01, "MAX_MOVE": 0.1, "RESTART_ON_SCALING": True,
             "LBFGS_MEMORY": 20,
             "LBFGS_DAMP": 1.0,
             "FD_STEP": 0.001,
             "LINESEARCH": None}

#NEB-TS: NEB-CI + OptTS
#Threshold settings for CI-NEB part are the same as in the NEB-TS of ORCA
def NEBTS(reactant=None, product=None, theory=None, images=8, CI=True, free_end=False, maxiter=100, IDPPonly=False,
        conv_type="ALL", tol_scale=10, tol_max_fci=0.10, tol_rms_fci=0.05, tol_max_f=1.03, tol_rms_f=0.51,
        tol_turn_on_ci=1.0,  runmode='serial', numcores=1, charge=None, mult=None, printlevel=1, ActiveRegion=False, actatoms=None,
        interpolation="IDPP", idpp_maxiter=700, restart_file=None, TS_guess=None, mofilesdir=None, 
        OptTS_maxiter=100, OptTS_print_atoms_list=None, OptTS_convergence_setting=None, OptTS_conv_criteria=None, OptTS_coordsystem='tric',
        hessian_for_TS=None, modelhessian='unit', tsmode_tangent_threshold=0.1, subfrctor=1):

    print_line_with_mainheader("NEB+TS")
    module_init_time=time.time()
    if IDPPonly is True:
        print("Will first perform IDPP-only NEB job, followed by TSOpt job using geomeTRIC optimizer.")
    else:
        print("Will first perform loose NEB-CI job, followed by TSOpt job using geomeTRIC optimizer.")
    print("Hessian option:", hessian_for_TS)
    #Will use maximum number of CPU cores provided to either NEBTS or theory object
    #cores_for_TSopt=max([numcores,theory.numcores])
    numcores=int(numcores)
    theory.set_numcores(int(theory.numcores))
    #Keeping original setting
    original_theory_numcores=copy.copy(theory.numcores)
    max_cores=int(numcores*theory.numcores)
    cores_for_TSopt=max_cores

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, reactant, "NEBTS", theory=theory)

    #Printing parallelization info
    print(f"{numcores} CPU cores provided to parallelize the {images}-image NEB band optimization.")
    print(f"{theory.numcores} CPU cores will be used to parallelize theory level for each image during NEB.")
    print(f"{cores_for_TSopt} CPU cores will be used simultaneously during NEB.")
    print(f"{cores_for_TSopt} CPU cores will be used to parallize theory during Opt-TS part.")

    #CI-NEB step
    SP,energies_dict = NEB(reactant=reactant, product=product, theory=theory, images=images, CI=CI, free_end=free_end, maxiter=maxiter, IDPPonly=IDPPonly,
            conv_type=conv_type, tol_scale=tol_scale, tol_max_fci=tol_max_fci, tol_rms_fci=tol_rms_fci, tol_max_f=tol_max_f, tol_rms_f=tol_rms_f,
            tol_turn_on_ci=tol_turn_on_ci,  runmode=runmode, numcores=numcores, 
            charge=charge, mult=mult,printlevel=printlevel, ActiveRegion=ActiveRegion, actatoms=actatoms,
            interpolation=interpolation, idpp_maxiter=idpp_maxiter, 
            restart_file=restart_file, TS_guess=TS_guess, mofilesdir=mofilesdir)

    if SP == None:
        print("NEB-CI job failed. Exiting NEBTS.")
        return None
        #ashexit()

    #SP.write_xyzfile(xyzfilename='Saddlepoint-NEBCI-approx.xyz')
    print("NEB-CI job is complete. Now choosing Hessian option to use for Opt-TS job.")

    #Prepare Hessian option
    #Hessianfile should be a simple text file with 1 row per line, values space-separated and no header.
    #Default: 
    if hessian_for_TS == None:
        print("Using default hessian_for_TS option")
        #If dualtheory we just do a Numfreq in regular mode
        if isinstance(theory,ash.DualTheory):
            print("Dualtheory active. Doing Numfreq using regular mode.")
            #NOTE: Regular will probably involve a theory 2 correction. We could switch to theory1 solely instead here
            result_freq = ash.NumFreq(theory=theory, fragment=SP, printlevel=0, runmode=runmode, numcores=numcores)
            hessianfile="Hessian_from_dualtheory"
            shutil.copyfile("Numfreq_dir/Hessian",hessianfile)
            hessianoption='file:'+str(hessianfile)
        else:
            print("Doing Numfreq")
            result_freq = ash.NumFreq(theory=theory, fragment=SP, printlevel=0, npoint='1', runmode=runmode, numcores=numcores)
            hessianfile="Hessian_from_theory"
            shutil.copyfile("Numfreq_dir/Hessian",hessianfile)
            hessianoption='file:'+str(hessianfile)
            #print("Will calculate exact Hessian in the beginning of OptTS job.")
            #hessianoption="first"
    # Tell geomeTRIC to calculate exact Hessian in the beginning
    elif hessian_for_TS == '2point':
            print("Doing Numfreq 2-point approximation")
            result_freq = ash.NumFreq(theory=theory, fragment=SP, printlevel=0, npoint=2, runmode=runmode, numcores=numcores)
            hessianfile="Hessian_from_theory"
            shutil.copyfile("Numfreq_dir/Hessian",hessianfile)
            hessianoption='file:'+str(hessianfile)
    elif hessian_for_TS == '1point':
            print("Doing Numfreq 1-point approximation")
            result_freq = ash.NumFreq(theory=theory, fragment=SP, printlevel=0, npoint=1, runmode=runmode, numcores=numcores)
            hessianfile="Hessian_from_theory"
            shutil.copyfile("Numfreq_dir/Hessian",hessianfile)
            hessianoption='file:'+str(hessianfile)
    elif hessian_for_TS == 'first':
        hessianoption="first"
    # Tell geomeTRIC to calculate exact Hessian in eachianfile="Hessian_from_xt step
    elif hessian_for_TS == 'each':
        hessianoption="each"
    #xTB Hessian option
    elif hessian_for_TS == 'xtb':
        hessianfile = calc_hessian_xtb(fragment=SP, runmode='serial', actatoms=actatoms, numcores=max_cores, use_xtb_feature=True)
        hessianoption='file:'+str(hessianfile)
    #Cheap model Hessian
    #NOTE: None of these work well.  Need to use tangent to modify
    elif hessian_for_TS == 'model':
        print("hessian_for_TS option: model")
        if modelhessian == 'Zero':
            hess_size=SP.numatoms*3
            hessian=np.zeros((hess_size,hess_size))
            hessianfile="Hessian_Zero"
        else:
            print("Calling ORCA to get model Hessian")
            print(f"modelhessian: {modelhessian}")
            #Calling ORCA to get model Hessian (default Swart) for SP geometry
            hessian = calc_model_Hessian_ORCA(SP,model=modelhessian)
            #NOTE: We can do overlap of eigenvectors but we can currently not change what geometric does
            hessianfile="Hessian_from_ORCA_model"
        #Write Hessian to file
        write_hessian(hessian,hessfile=hessianfile)
        #Creating string 
        hessianoption='file:'+str(hessianfile)
    #Finding atoms that contribute the most to saddlepoint mode according to CI-NEB. Perform partial Hessian optimization
    elif hessian_for_TS == 'partial':
        print("hessian_for_TS option: partial")
        print(f"Using climbing image tangent to find dominant atoms in approximate TS mode (tsmode_tangent_threshold={tsmode_tangent_threshold})")

        #Getting tangent and atoms that contribute at least X to tangent where X is tsmode_tangent_threshold=0.1 (default)
        tangent = read_tangent("CItangent.xyz")
        TSmodeatoms = list(np.where(np.any(abs(tangent)>tsmode_tangent_threshold, axis=1))[0])

        print(f"Performing partial Hessian calculation using atoms: {TSmodeatoms}")
        #TODO: Make this work for QMMMTheory
        #TODO: Option to run this in parallel ?
        #Or just enable theory parallelization 
        #if isinstance(theory,ash.DualTheory): theory.switch_to_theory(2)
        result_freq = ash.NumFreq(theory=theory, fragment=SP, printlevel=0, npoint=2, hessatoms=TSmodeatoms, runmode=runmode, numcores=numcores)

        #Combine partial exact Hessian with model Hessian(Almloef, Lindh, Schlegel or unit)
        #Large Hessian is the actatoms Hessian if actatoms provided
        combined_hessian = approximate_full_Hessian_from_smaller(SP,result_freq.hessian,TSmodeatoms, large_atomindices=actatoms, restHessian=modelhessian)

        #Write combined Hessian to disk
        hessianfile="Hessian_from_partial"
        write_hessian(combined_hessian,hessfile=hessianfile)
        #Creating string 
        hessianoption='file:'+str(hessianfile)

    else:
        print("Unknown hessian_for_TS option")
        ashexit()

    #if Dualtheory switch to theory2 before TSOpt
    if isinstance(theory,ash.DualTheory): theory.switch_to_theory(2)

    #TSopt
    print(f"Now starting Optimizer job from NEB-CI saddlepoint with TSOpt=True with hessian option: {hessianoption}")
    print(f"Changing number of cores of Theory object from : {theory.numcores} cores ", end="")
    theory.set_numcores(cores_for_TSopt)
    print(f"to: {cores_for_TSopt} cores")

    ash.Optimizer(theory=theory, fragment=SP, charge=charge, mult=mult, coordsystem=OptTS_coordsystem, maxiter=OptTS_maxiter, 
                ActiveRegion=ActiveRegion, actatoms=actatoms, convergence_setting=OptTS_convergence_setting, 
                conv_criteria=OptTS_conv_criteria, print_atoms_list=OptTS_print_atoms_list, TSOpt=True,
                hessian=hessianoption, subfrctor=subfrctor)

    #TODO: Test if Optimizer converged or not. Currently there would be an error from geometric.
    # 
    # Finalprintout here with energies of all images, CI image pointed out and TS image also.
    #Also write-out NEB-CI image as Saddlepoint-NEBCI-approx.xyz and Saddlepoint-OptTS.xyz


    #Printing table of energies
    CI_num = max(energies_dict, key=energies_dict.get) #Getting CI number as HEI
    #Add TS geometry to energies_dict as -1
    energies_dict[-1] = SP.energy
    print("\n\nFinal energies of all NEB-CI images and final saddlepoint (in Eh and kcal/mol)")
    print("-"*80)

    for i in range(0,images+2):
        label="Image:"
        if free_end == False and (i == 0 or i == images+1):
            #If image was frozen
            relenergy=(energies_dict[i]-energies_dict[0])*ash.constants.hartokcal
            print(f"{label:<8} {i:<4}Energy:{energies_dict[i]:12.6f}  {relenergy:8.2f} (frozen)")
        elif i == CI_num:
            #Printing CI
            relenergy=(energies_dict[i]-energies_dict[0])*ash.constants.hartokcal
            print(f"{label:<8} {i:<4}Energy:{energies_dict[i]:12.6f}  {relenergy:8.2f} (CI)")
            #Printing TS
            label="TS"
            relenergy=(energies_dict[-1]-energies_dict[0])*ash.constants.hartokcal
            print(f"{label:<8} {label:<4}Energy:{energies_dict[-1]:12.6f}  {relenergy:8.2f} (TS)")
            SP_relenergy=relenergy
        else:
            #If regular image
            relenergy=(energies_dict[i]-energies_dict[0])*ash.constants.hartokcal
            print(f"{label:<8} {i:<4}Energy:{energies_dict[i]:12.6f}  {relenergy:8.2f}")            
    
    print("-"*80)
    print()

    #Writing final geometries for clarity
    SP.write_xyzfile(xyzfilename='Saddlepoint-OptTS.xyz')

    #Changing numcores back in case theory is reused
    theory.set_numcores(original_theory_numcores)
    print_time_rel(module_init_time, modulename='NEB-TS run', moduleindex=1)
    
    #Returning result object
    result = ASH_Results(label="NEBTS calc", energy=SP.energy, geometry=SP.coords,
        saddlepoint_fragment=SP, charge=charge, mult=mult, MEP_energies_dict=energies_dict,
        barrier_energy=SP_relenergy)
    return result
    #return SP

#ASH NEB function. Calls Knarr
def NEB(reactant=None, product=None, theory=None, images=8, CI=True, free_end=False, maxiter=100,
        conv_type="ALL", tol_scale=10, tol_max_fci=0.026, tol_rms_fci=0.013, tol_max_f=0.26, tol_rms_f=0.13,
        tol_turn_on_ci=1.0,  runmode='serial', numcores=1, IDPPonly=False,
        charge=None, mult=None,printlevel=1, ActiveRegion=False, actatoms=None,
        interpolation="IDPP", idpp_maxiter=700, 
        restart_file=None, TS_guess=None, mofilesdir=None):

    print_line_with_mainheader("Nudged elastic band calculation (via interface to KNARR)")
    module_init_time=time.time()

    if reactant==None or product==None or theory==None:
        print(BC.FAIL,"You need to provide reactant and product fragment and a theory to NEB", BC.END)
        ashexit()

    if runmode == 'serial' and numcores > 1:
        print(BC.FAIL,"Runmode is 'serial' but numcores > 1. Set runmode to 'parallel' to have NEB parallelize over images", BC.END)
        ashexit()
    elif runmode == 'parallel' and numcores == 1:
        print(BC.FAIL,"Runmode is 'parallel' but numcores == 1. You must provide more than 1 core to parallelize over images", BC.END)
        print(BC.FAIL,"It is recommended to provide as many cores as there are images", BC.END)
        ashexit()
    elif runmode == 'parallel' and numcores > 1:
        print(BC.WARNING,f"Runmode is 'parallel' and numcores == {numcores}.")
        print(BC.WARNING,f"Will launch Energy+gradient calculations using Singlepoint_parallel using {numcores} cores.", BC.END)
        if theory.numcores > 1:
            print(BC.WARNING,f"Warning: Theory parallelization is active and will utilize: {theory.numcores} cores.", BC.END)
            print(BC.WARNING,f"The NEB images will run in parallel by Python multiprocessing (using {numcores} cores) while each image E+Grad calculation is parallelized as well ({theory.numcores} per image)", BC.END)
            print(BC.WARNING,f"Make sure that you have {numcores} x {theory.numcores} = {numcores*theory.numcores} CPU cores available to this ASH job on the computing node", BC.END)
    elif runmode == 'serial' and numcores == 1:
        print (BC.WARNING,"NEB runmode is serial, i.e. running one image after another.", BC.END)
        if theory.numcores > 1:
            print(BC.WARNING,f"Theory parallelization is active and will utilize: {theory.numcores} CPU cores per image.",BC.END)
        else:
            print(BC.WARNING,"Warning: Theory parallelization is not active either (provide numcores keyword to Theory object).",BC.END)
    else:
        print("Unknown runmode, continuing.")
    print()

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, reactant, "NEB", theory=theory)
    numatoms = reactant.numatoms

    #Number of total images that Knarr wants. images input referring to intermediate images is now consistent with ORCA
    total_num_images=images+2
    

    #Zero-valued constraints list. We probably won't use constraints for now
    constr = np.zeros(shape=(numatoms * 3, 1))

    #ActiveRegion feature
    if ActiveRegion==True:
        print("Active Region option Active. Passing only active-region coordinates to Knarr.")
        if actatoms is None:
            print("You must include actatoms keyword (with list of atom indices) to NEB for ActiveRegion True")
            ashexit()
        R_actcoords, R_actelems = reactant.get_coords_for_atoms(actatoms)
        P_actcoords, P_actelems = product.get_coords_for_atoms(actatoms)
        new_reactant = ash.Fragment(coords=R_actcoords, elems=R_actelems)
        new_product = ash.Fragment(coords=P_actcoords, elems=P_actelems)

        #TSguess fragment provided
        if TS_guess != None:
            TS_actcoords, TS_actelems = TS_guess.get_coords_for_atoms(actatoms)
            new_TSguess = ash.Fragment(coords=TS_actcoords, elems=TS_actelems, printlevel=0)
            new_TSguess.write_xyzfile(xyzfilename="TSguess.xyz")
        #Create Knarr calculator from ASH theory.
        calculator = KnarrCalculator(theory, fragment1=new_reactant, fragment2=new_product, runmode=runmode, numcores=numcores,
                                     ActiveRegion=True, actatoms=actatoms, full_fragment_reactant=reactant,
                                     full_fragment_product=product,numimages=total_num_images, charge=charge, mult=mult,
                                     FreeEnd=free_end, printlevel=printlevel,mofilesdir=mofilesdir)

        # Symbols list for Knarr
        Knarr_symbols = [y for y in new_reactant.elems for i in range(3)]

        # New numatoms and constraints for active-region system
        numatoms = new_reactant.numatoms
        constr = np.zeros(shape=(numatoms * 3, 1))

        # Create KNARR Atom objects. Used in path generation
        react = KNARRatom.atom.Atom(coords=coords_to_Knarr(new_reactant.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                    ndof=numatoms * 3, constraints=constr, pbc=False)
        prod = KNARRatom.atom.Atom(coords=coords_to_Knarr(new_product.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                   ndof=numatoms * 3, constraints=constr, pbc=False)


    else:
        if TS_guess != None:
            #Writing XYZ-file for TSguess
            TS_guess.write_xyzfile(xyzfilename="TSguess.xyz")

        #Create Knarr calculator from ASH theory
        calculator = KnarrCalculator(theory, fragment1=reactant, fragment2=product, numcores=numcores,
                                     ActiveRegion=False, runmode=runmode,numimages=total_num_images, charge=charge, mult=mult,
                                     FreeEnd=free_end, printlevel=printlevel,mofilesdir=mofilesdir)

        # Symbols list for Knarr
        Knarr_symbols = [y for y in reactant.elems for i in range(3)]

        # Create KNARR Atom objects. Used in path generation
        react = KNARRatom.atom.Atom(coords=coords_to_Knarr(reactant.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                    ndof=numatoms * 3, constraints=constr, pbc=False)
        prod = KNARRatom.atom.Atom(coords=coords_to_Knarr(product.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                   ndof=numatoms * 3, constraints=constr, pbc=False)


    #Set Knarr settings in dictionary
    path_parameters["INTERPOLATION"]=interpolation
    path_parameters["IDPP_MAX_ITER"] = idpp_maxiter
    if TS_guess != None:
        path_parameters["INSERT_CONFIG"] = "TSguess.xyz"
    neb_settings["CLIMBING"]=CI
    neb_settings["FREE_END"] = free_end
    neb_settings["CONV_TYPE"] = conv_type
    neb_settings["TOL_SCALE"] = tol_scale
    neb_settings["TOL_MAX_FCI"] = tol_max_fci
    neb_settings["TOL_RMS_FCI"] = tol_rms_fci
    neb_settings["TOL_MAX_F"] = tol_max_f
    neb_settings["TOL_RMS_F"] = tol_rms_f
    neb_settings["TOL_TURN_ON_CI"] = tol_turn_on_ci
    optimizer["MAX_ITER"] = maxiter
    #Setting number of images of Knarr
    path_parameters["NIMAGES"]=total_num_images

    if ActiveRegion is True:
        print("Active Region feature active. Setting RMSD-alignment in NEB to false (required).")
        neb_settings["MIN_RMSD"] = False

    print()
    print_line_with_subheader2("Active NEB settings:")
    print()

    #images provided => Meaning intermediate images
    print("Number of images chosen:", images)
    print("Free_end option:", free_end)
    if free_end == True:
        print("Endpoints have been chosen to be free. Reactant and product geometries will thus also be active during NEB optimization.")
        print(f"There are {total_num_images} active images including endpoints.")
        print("Warning: Check that you have chosen an appropriate number of CPU cores for runmode=parallel")
    else:
        print("Endpoints are frozen. Reactant and product will only be calculated once in the beginning and then frozen.")
        print(f"{images} intermediate images will active and calculated during each primary NEB iteration (first iteration excluded)")
        print(f"There are {total_num_images} images including the frozen endpoints.")

    print("Restart file:", restart_file)
    print("TS guess insertion :", TS_guess)

    print()
    print("Interpolation path parameters:\n", path_parameters)
    print()
    print("NEB parameters:\n", neb_settings)
    print()
    print("Optimizer parameters:\n", optimizer)
    print()







    print("\nLaunching Knarr")
    print()
    PrintDivider()
    PrintDivider()
    PrintHeader()
    PrintCredit()
    PrintDivider()
    PrintDivider()

    if restart_file == None:
        print("Creating interpolated path.")

        if TS_guess != None:
            print(f"A TS guess : {TS_guess} was provided")
            print("Will use intermediate geometry in interpolation")

        # Generate path via Knarr_pathgenerator. ActiveRegion used to prevent RMSD alignment if doing actregion QM/MM etc.
        Knarr_pathgenerator(neb_settings, path_parameters, react, prod, ActiveRegion)
        print("Saving initial path as : initial_guess_path.xyz")
        shutil.copyfile("knarr_path.xyz","initial_guess_path.xyz")
        os.remove("knarr_path.xyz")
        print("\nReading initial path")
        #Reading initial path from XYZ file.
        rp, ndim, nim, symb = ReadTraj("initial_guess_path.xyz")

        path = InitializePathObject(nim, react)
        path.SetCoords(rp)
    else:
        print("Restart_file option active")
        print(f"Restart-file: {restart_file} will be read and used as guess path instead of interpolation trajectory")
        #Reading user-defined path from XYZ file.
        rp, ndim, nim, symb = ReadTraj(restart_file)
        path = InitializePathObject(nim, react)
        path.SetCoords(rp)
    
    
    print("Starting NEB")
    #Setting printlevel of theory during E+Grad steps  1=very-little, 2=more, 3=lots, 4=verymuch
    print("NEB printlevel is:", printlevel)
    theory.printlevel=printlevel
    print("Theory print level will now be set to:", theory.printlevel)
    if isinstance(theory, ash.QMMMTheory):
        theory.qm_theory.printlevel = printlevel
        theory.mm_theory.printlevel = printlevel



    #############################
    # CALLING NEB
    #############################
    
    if IDPPonly == True:
        print("IDPPonly option will do one NEB iteration on the IDPP path and then stop NEB part")
        optimizer["MAX_ITER"] = 1
        #Now starting NEB from path object, using neb_settings and optimizer settings
        print("neb_settings:", neb_settings)
        print("optimizer:", optimizer)
        DoNEB(path, calculator, neb_settings, optimizer)

        #Now finding highest energy image
        Saddlepoint_fragment = prepare_saddlepoint(path,neb_settings,reactant,calculator,ActiveRegion,actatoms,charge,mult, numatoms, "IDPP", write_tangent=False)
        print("WARNING: This is a highly approximate guess for the saddlepoint, based on the highest energy image from a single-iteration NEB.")
        #return Saddlepoint_fragment, calculator.energies_dict

        #Returning result object
        result = ASH_Results(label="NEB-IDPPonly calc", energy=Saddlepoint_fragment.energy, geometry=Saddlepoint_fragment.coords,
            saddlepoint_fragment=Saddlepoint_fragment, charge=charge, mult=mult, MEP_energies_dict=calculator.energies_dict,
            barrier_energy=None)
        return result

    #REGULAR NEB
    else:
        #Now starting NEB from path object, using neb_settings and optimizer settings
        print("neb_settings:", neb_settings)
        print("optimizer:", optimizer)
        DoNEB(path, calculator, neb_settings, optimizer)

        ###########################################################
        # CHECKING CONVERGENCE AND PREPARING FINAL OUTPUT
        ###########################################################
        if calculator.converged == False:
            print()
            print(f"Knarr failed to converge during the maxiter={maxiter} given.")
            print("Try restarting with different settings.")

            print_time_rel(module_init_time, modulename='Knarr-NEB run', moduleindex=1)

            #Returning result object with all attributes None
            result = ASH_Results(label="NEB-CI calc (fail)")
            return result

        else:
            print()
            print('KNARR successfully terminated')
            print()
            Saddlepoint_fragment = prepare_saddlepoint(path,neb_settings,reactant,calculator,ActiveRegion,actatoms,charge,mult, numatoms, "NEBCIapprox")
            print("WARNING: The NEB-CI saddlepoint is usually close to the true saddlepoint. Needs confirmation by Hessian.")
            print()

        print("\nThe Knarr-NEB code is based on work described in the following article. Please consider citing it:")
        print("Nudged elastic band method for molecular reactions using energy-weighted springs combined with eigenvector following\n \
    V. Ásgeirsson, B. Birgisson, R. Bjornsson, U. Becker, F. Neese, C: Riplinger,  H. Jónsson, J. Chem. Theory Comput. 2021,17, 4929–4945.\
    DOI: 10.1021/acs.jctc.1c00462")

        print_time_rel(module_init_time, modulename='Knarr-NEB run', moduleindex=1)

        if neb_settings["CLIMBING"] is True:
            #return Saddlepoint_fragment, calculator.energies_dict
            #Returning result object
            result = ASH_Results(label="NEB-CI calc", energy=Saddlepoint_fragment.energy, geometry=Saddlepoint_fragment.coords,
                saddlepoint_fragment=Saddlepoint_fragment, charge=charge, mult=mult, MEP_energies_dict=calculator.energies_dict,
                barrier_energy=None)
            return result
        else:
            #Returning result object
            result = ASH_Results(label="NEB calc", charge=charge, mult=mult, MEP_energies_dict=calculator.energies_dict)
            return result



#Path generator
def Knarr_pathgenerator(nebsettings,path_parameters,react,prod,ActiveRegion):
    sett = nebsettings
    type_of_method_string = path_parameters["METHOD"].upper()
    if type_of_method_string == "SINGLE":
        prod_is_needed = False
    elif type_of_method_string == "DOUBLE":
        prod_is_needed = True
    else:
        raise TypeError("Either choose single or double ended path generation")

    nim = path_parameters["NIMAGES"]
    path = InitializePathObject(nim, react)
    if prod_is_needed:
        # Check product
        if react.GetNDim() != prod.GetNDim():
            raise RuntimeError("Reactant / product do not match")
        if react.GetSymbols() != prod.GetSymbols():
            raise RuntimeError("Reactant / product do not match")
        path.SetConfig2(prod.GetCoords())

        # check insertion
        insertion = path_parameters["INSERT_CONFIG"]
        if insertion is not None:
            #pbc=main_control["PBC"]
            insertion = InitializeAtomObject(name="insertion", input_config=path_parameters["INSERT_CONFIG"])
            if insertion.GetSymbols() != react.GetSymbols():
                raise ValueError("Insertion does not match reactant / product")
            path.SetInsertionConfig(insertion.GetCoords())
    else:
        prod = None

    #Setting path.twodee True  prevents RMSD alignment in pathgeneration
    if ActiveRegion is True:
        print("Using ActiveRegion in NEB. Turning off RMSD alignment in Path generation")
        path.twodee = True
        #print("path istwodee", path.IsTwoDee())
    DoPathInterpolation(path, path_parameters)

#Convert coordinates list to Knarr-type array
def coords_to_Knarr(coords):
    coords_xyz=[]
    for i in coords:
        coords_xyz.append([i[0]]);coords_xyz.append([i[1]]);coords_xyz.append([i[2]])
    coords_xyz_np=np.array(coords_xyz)
    return coords_xyz_np

#Wrapper around ASH object passed onto Knarr
class KnarrCalculator:
    def __init__(self,theory,fragment1,fragment2,runmode='serial',printlevel=None, ActiveRegion=False, actatoms=None, numcores=1,
                 full_fragment_reactant=None, full_fragment_product=None, numimages=None, FreeEnd=False, charge=None, mult=None, mofilesdir=None ):
        self.numcores=numcores
        self.FreeEnd=FreeEnd
        self.numimages=numimages
        self.mofilesdir=mofilesdir
        self.printlevel=printlevel
        self.forcecalls=0
        self.iterations=-2  #Starting from -2 as R and P done first
        self.theory=theory
        self.charge=charge
        self.mult=mult
        #ASH fragments for reactant and product
        #Used for element list and keep track of full system if QM/MM
        self.fragment1=fragment1
        self.fragment2=fragment2
        #Full ASH fragments for reactant and product. Inactive part of reactant will be used for all images
        self.full_fragment_reactant=full_fragment_reactant
        self.full_fragment_product=full_fragment_product
        self.runmode=runmode
        self.ISCION=False
        self.ActiveRegion=ActiveRegion
        self.actatoms=actatoms
        self.full_coords_images_dict={}
        self.energies_dict={}
        self.gradient_dict={} #Keeps track of gradients for each image. Should only be active-space gradients
        self.converged=False 
        #Activating ORCA flag if theory or QM-region theory
        self.ORCAused = False
        if isinstance(self.theory, ash.ORCATheory):
            self.ORCAused = True
        elif isinstance(self.theory, ash.QMMMTheory):
            if isinstance(self.theory.qm_theory, ash.ORCATheory):
                self.ORCAused = True
        #Final tangent of saddlepoint. Will be available if job converges
        self.tangent=None
    #Function that Knarr will use to signal convergence and set self.converged to True. Otherwise it is False
    def status(self,converged):
        self.converged=converged
    def Compute(self,path, list_to_compute=None):
        if list_to_compute is None:
            return
        else:
            list_to_compute=list(list_to_compute)
            self.iterations+=1
        print()
        print("="*30)
        print("NEB ITERATION:", self.iterations)
        print("="*30)
        print()
        print(f"Images to be computed in this iteration: {list_to_compute}\n")

        counter=0
        F = np.zeros(shape=(path.GetNDimIm() * path.GetNim(), 1))
        E = np.zeros(shape=(path.GetNim(), 1))
        numatoms=int(path.ndofIm/3)

        if self.runmode=='serial':
            print("Starting NEB calculations in serial mode")
            for image_number in list_to_compute:
                print("\nComputing image: ", image_number)

                #Creating dir 
                try:
                    os.mkdir(f"image_{image_number}")
                except FileExistsError:
                    pass
                os.chdir(f"image_{image_number}")
                
                image_coords_1d = path.GetCoords()[image_number * path.ndimIm : (image_number + 1) * path.ndimIm]
                image_coords=np.reshape(image_coords_1d, (numatoms, 3))

                #Reading initial set of orbitals if requested
                if self.iterations == -1 or self.iterations == 0:
                    if self.mofilesdir != None:
                        if self.ORCAused == True:
                            imagefile_name="current_image"+str(image_number)+".gbw" #The imagefile to look for
                            path_to_imagefile=self.mofilesdir+"/"+imagefile_name #Full path to image file
                            if self.printlevel >= 1:
                                print(f"mofilesdir option active for iteration {self.iterations}. Looking inside {self.mofilesdir} for imagefile: {imagefile_name}")
                            if os.path.exists(path_to_imagefile):
                                if self.printlevel >= 1:
                                    print(f"File {path_to_imagefile} DOES exist. Will copy to image dir.")
                                shutil.copyfile(path_to_imagefile,"./"+imagefile_name)
                            else:
                                if self.printlevel >= 1:
                                    print(f"File {path_to_imagefile} does NOT exist. Continuing.")
                
                #Handling GBW files for ORCATheory and QMMMTheory (if using ORCA)
                if self.ORCAused == True:
                    if self.printlevel >= 1:
                        print("ORCATheory is being used")
                    current_image_file="current_image"+str(image_number)+".gbw"
                    if os.path.exists(current_image_file):
                        if self.printlevel >= 1:
                            print(f"File: {current_image_file} exists.")
                        if isinstance(self.theory,ash.QMMMTheory):
                            print(f"Copying {current_image_file} to {self.theory.qm_theory.filename}.gbw to be used.")
                            shutil.copyfile(current_image_file,self.theory.qm_theory.filename+".gbw")                        
                        else:
                            print(f"Copying {current_image_file} to {self.theory.filename}.gbw to be used.")
                            shutil.copyfile(current_image_file,self.theory.filename+".gbw")
                    else:
                        if self.printlevel >= 1:
                            print(f"current_image_file {current_image_file} DOES NOT exist")
                        if isinstance(self.theory,ash.QMMMTheory):
                            if os.path.exists(self.theory.qm_theory.filename+".gbw"):
                                if self.printlevel >= 1:
                                    print(f"A file {self.theory.qm_theory.filename}.gbw file does exist. Will use.")
                            else:
                                if self.printlevel >= 1:
                                    print(f"A file {self.theory.qm_theory.filename}.gbw file DOES NOT exist.")
                                    #If not image_0 let's try to copy GBW file from image_0 dir first
                                    if image_number != 0:
                                        print("Will try to copy GBW file from image_0 dir first")
                                        try:
                                            shutil.copyfile(f"../image_0/{self.qm_theory.filename}.gbw",f"./{self.qm_theory.filename}.gbw")
                                            print(f"Found a file : ../image_0/{self.qm_theory.filename}.gbw  Copying to dir: image_{image_number}")
                                        except:
                                            print(f"No {self.qm_theory.filename}.gbw file found in image_0 dir. Will use ORCATheory settings.")
                                    else:
                                        print("Will use ORCATheory settings")
                        else:
                            if os.path.exists(self.theory.filename+".gbw"):
                                if self.printlevel >= 1:
                                    print(f"A file {self.theory.filename}.gbw file does exist. Will use.")
                            else:
                                if self.printlevel >= 1:
                                    print(f"A file {self.theory.filename}.gbw file DOES NOT exist.")
                                    #If not image_0 let's try to copy GBW file from image_0 dir first
                                    if image_number != 0:
                                        print("Will try to copy GBW file from image_0 dir first")
                                        try:
                                            shutil.copyfile(f"../image_0/{self.theory.filename}.gbw",f"./{self.theory.filename}.gbw")
                                            print(f"Found a file in : ../image_0/{self.theory.filename}.gbw  Copying to dir: image_{image_number}")
                                        except:
                                            print(f"No {self.theory.filename}.gbw file found in image_0 dir. Will use ORCATheory settings.")
                                    else:
                                        print("Will use ORCATheory settings")
                if self.ActiveRegion == True:
                    currcoords=image_coords
                    # Defining full_coords as original coords temporarily
                    #full_coords = self.full_fragment_reactant.coords
                    #Creating deep copy of reactant coordinates as it will be modified
                    full_coords = copy.deepcopy(self.full_fragment_reactant.coords)

                    # Replacing act-region coordinates with coords from currcoords

                    for i, c in enumerate(full_coords):
                        if i in self.actatoms:
                            # Silly. Pop-ing first coord from currcoords until done
                            curr_c, currcoords = currcoords[0], currcoords[1:]
                            full_coords[i] = curr_c
                    full_current_image_coords = full_coords

                    #List of all image-geometries (full coords)
                    #full_coords_images_list.append(full_current_image_coords)

                    self.full_coords_images_dict[image_number] = copy.deepcopy(full_current_image_coords)

                    #Write full and active geometry to disk
                    write_xyzfile(self.full_fragment_reactant.elems, full_current_image_coords, "image_"+str(image_number)+"_Full", printlevel=self.printlevel, writemode='w')
                    write_xyzfile(self.fragment1.elems, image_coords, "image_"+str(image_number)+"_active", printlevel=self.printlevel, writemode='w')

                    #EnGrad calculation on full system
                    En_image, Grad_image_full = self.theory.run(current_coords=full_current_image_coords, charge=self.charge, mult=self.mult,
                                                                elems=self.full_fragment_reactant.elems, Grad=True, label="image_"+str(image_number))

                    if self.ORCAused == True:

                        if self.printlevel >= 1:
                            print(f"ORCA run done. Copying {self.theory.filename}.gbw to {current_image_file} for next time")
                        
                        if isinstance(self.theory,ash.QMMMTheory):
                            shutil.copyfile(self.theory.qm_theory.filename+".gbw",current_image_file)
                        else:
                            shutil.copyfile(self.theory.filename+".gbw",current_image_file)
                    if self.printlevel >= 2:
                        print("Energy of image {} is : {}".format(image_number,En_image))
                    #Trim Full gradient down to only act-atoms gradient
                    Grad_image = np.array([Grad_image_full[i] for i in self.actatoms])

                    #Keeping track of energies for each image in a dict
                    self.energies_dict[image_number] = En_image
                    #Keeping track of (active-region) gradients for each image in a dict
                    self.gradient_dict[image_number]=Grad_image

                else:

                    #Write geometry to disk
                    write_xyzfile(self.fragment1.elems, image_coords, "image_"+str(image_number), printlevel=self.printlevel, writemode='w')

                    En_image, Grad_image = self.theory.run(current_coords=image_coords, elems=self.fragment1.elems, Grad=True, charge=self.charge, mult=self.mult,
                        label="image_"+str(image_number))
                    
                    if self.ORCAused == True:
                        if self.printlevel >= 1:
                            print(f"ORCA run done. Copying {self.theory.filename}.gbw to {current_image_file} for next time")
                        if isinstance(self.theory,ash.QMMMTheory):
                            shutil.copyfile(self.theory.qm_theory.filename+".gbw",current_image_file)
                        else:
                            shutil.copyfile(self.theory.filename+".gbw",current_image_file)
                    
                    #Keeping track of energies for each image in a dict
                    self.energies_dict[image_number] = En_image
                    #Keeping track of  gradients for each image in a dict
                    self.gradient_dict[image_number]=Grad_image
                
                counter += 1
                #Energies array for all images
                En_eV=En_image*constants.hartoeV
                E[image_number]=En_eV
                #Forces array for all images
                #Convert ASH gradient to force and convert to ev/Ang instead of Eh/Bohr
                force = -1 * np.reshape(Grad_image,(int(path.ndofIm),1)) * 51.42210665240553
                F[image_number* path.ndimIm : (image_number + 1) * path.ndimIm] = force

                #Going up from image dir
                os.chdir('..')

        #PARALLEL
        elif self.runmode=='parallel':
            print("Starting NEB calculations in parallel mode")
            print("")

            all_image_fragments=[] #List of ASH fragments that will be passed onto Singlepoint_parallel
            
            #Looping over images, creating fragments
            for image_number in list_to_compute:
                counter += 1

                #Reading initial set of orbitals if requested, but only in teration -1 or 0 and copying to workerdir
                if self.iterations == -1 or self.iterations == 0:
                    #Creating directories for each image beforehand and adding initial GBW-files for each image
                    workerdir='Pooljob_'+"image_"+str(image_number) #Same name of dir that Singlepoint_parallel expects
                    if self.printlevel >= 1:
                        print(f"Creating worker directory: {workerdir}")
                    try:
                        os.mkdir(workerdir)
                    except:
                        pass
                    if self.mofilesdir != None:
                        if self.ORCAused == True:
                            imagefile_name="current_image"+str(image_number)+".gbw" #The imagefile to look for
                            path_to_imagefile=self.mofilesdir+"/"+imagefile_name #Full path to image file
                            if self.printlevel >= 1:
                                print(f"mofilesdir option active for iteration {self.iterations}. Looking inside {self.mofilesdir} for imagefile: {imagefile_name}")
                            if os.path.exists(path_to_imagefile):
                                if self.printlevel >= 1:
                                    print(f"File {path_to_imagefile} DOES exist")
                                    print(f"Copying file {path_to_imagefile} to dir {workerdir} as {self.theory.filename}.gbw")
                                shutil.copyfile(path_to_imagefile,workerdir+"/"+self.theory.filename+".gbw") #Copying to Pooljob_image_X as orca.gbw
                            else:
                                if self.printlevel >= 1:
                                    print(f"File {path_to_imagefile} does NOT exist. Continuing.")


                ################################################
                #FRAGMENT HANDLING FOR ACTIVE-REGION AND FULL
                ################################################
                #Getting 1D coords array from Knarr, converting to regular, creating ASH fragment
                image_coords_1d = path.GetCoords()[image_number * path.ndimIm : (image_number + 1) * path.ndimIm]
                image_coords=np.reshape(image_coords_1d, (numatoms, 3))
                if self.ActiveRegion == True:
                    print("Warning: NEB-parallel with ActiveRegion is highly experimental")
                    currcoords=image_coords
                    # Defining full_coords as original coords temporarily
                    #full_coords = self.full_fragment_reactant.coords
                    #Creating deep copy of reactant coordinates as it will be modified
                    full_coords = copy.deepcopy(self.full_fragment_reactant.coords)
                    # Replacing act-region coordinates with coords from currcoords
                    for i, c in enumerate(full_coords):
                        if i in self.actatoms:
                            # Silly. Pop-ing first coord from currcoords until done
                            curr_c, currcoords = currcoords[0], currcoords[1:]
                            full_coords[i] = curr_c
                    full_current_image_coords = full_coords

                    full_frag=ash.Fragment(coords=full_current_image_coords, elems=self.full_fragment_reactant.elems,charge=self.charge, mult=self.mult, label="image_"+str(image_number), printlevel=self.printlevel)
                    all_image_fragments.append(full_frag)
                else:
                    #NO active region
                    frag=ash.Fragment(coords=image_coords, elems=self.fragment1.elems,charge=self.charge, mult=self.mult, label="image_"+str(image_number), printlevel=self.printlevel)
                    all_image_fragments.append(frag)

            #Launching multiple ASH E+Grad calculations in parallel on list of ASH fragments: all_image_fragments
            result_par = ash.Singlepoint_parallel(fragments=all_image_fragments, theories=[self.theory], numcores=self.numcores, 
                allow_theory_parallelization=True, Grad=True, printlevel=self.printlevel)
            en_dict = result_par.energies_dict
            #Now looping over gradients present (done to avoid overwriting frozen-image gradients)
            #self.gradient_dict = result_par.gradients_dict
            for gradkey in result_par.gradients_dict:
                self.gradient_dict[gradkey] = result_par.gradients_dict[gradkey]

            #Keeping track of energies for each image in a dict
            for i in en_dict.keys():
                #i is image_X
                im=int(i.replace("image_",""))
                En_image=en_dict[i]
                if self.printlevel >= 2:
                    print("Energy of image {} is : {}".format(image_number,En_image))

                #Keeping track of images in Eh
                self.energies_dict[im] = En_image
                #Knarr energy array for all images in eV
                E[im]=En_image*constants.hartoeV

                #Forces array for all images
                #ActiveRegion: Trim Full gradient down to only act-atoms gradient
                if self.ActiveRegion is True:
                    Grad_image_full = self.gradient_dict[i]
                    #Trimming gradient if active region
                    Grad_image = np.array([Grad_image_full[i] for i in self.actatoms])
                else:
                    Grad_image = self.gradient_dict[i]

                #Convert ASH gradient to force and convert to ev/Ang instead of Eh/Bohr
                force = -1 * np.reshape(Grad_image,(int(path.ndofIm),1)) * 51.42210665240553
                #print("im bla", im* path.ndimIm : (im + 1) * path.ndimIm)
                F[im* path.ndimIm : (im + 1) * path.ndimIm] = force

        path.SetForces(F)
        path.SetEnergy(E)
        #Forcecalls
        path.AddFC(counter)
        print("NEB iteration calculations done\n")

        #Printing table of images
        print("Overview of images")        
        header=f"Image  Energy(Eh)  dE(kcal/mol)  State     RMSF(eV/Ang)    MaxF(eV/Ang)"
        print(header)
        print("-"*70)
        for i in sorted(self.energies_dict.keys()):
            #RMSF and MaxF in eV/Angstrom
            rms_f=RMSfunc(self.gradient_dict[i])*51.42210665240553
            max_f=np.max(self.gradient_dict[i])*51.42210665240553
            if self.FreeEnd == False and (i == 0 or i == self.numimages-1):
                relenergy=(self.energies_dict[i]-self.energies_dict[0])*ash.constants.hartokcal
                #print(f"Image: {i:<4}Energy:{self.energies_dict[i]:12.6f}  {relenergy:8.2f} (frozen) RMSF: {rms_f:6.4f} MaxF: {max_f:6.4f}")
                print(f"{i:>4}{self.energies_dict[i]:>12.6f}{relenergy:>11.2f}{'frozen':>12s}{rms_f:>12.4f}{max_f:>16.4f}")
            else:
                relenergy=(self.energies_dict[i]-self.energies_dict[0])*ash.constants.hartokcal
                #print(f"Image: {i:<4}Energy:{self.energies_dict[i]:12.6f}  {relenergy:8.2f}          RMSF: {rms_f:6.4f} MaxF: {max_f:6.4f}")
                print(f"{i:>4}{self.energies_dict[i]:>12.6f}{relenergy:>11.2f}{'active':>12s}{rms_f:>12.4f}{max_f:>16.4f}")
        print("-"*70)
        print()
        #Write out full MEP path in each NEB iteration.
        if self.ActiveRegion is True:
            if self.iterations >= 0:
                self.write_Full_MEP_Path(path, list_to_compute, E)
        
        #END OF COMPUTE HERE


    def write_Full_MEP_Path(self, path, list_to_compute, E):
        #Write out MEP for full coords in each iteration. Knarr writes out Active Part.
        if self.ActiveRegion is True:
            with open("knarr_MEP_FULL.xyz", "w") as trajfile:
                #Todo: This will fail if free_end=True
                #Todo: disable react and prod printing if free_end True

                #Writing reactant image. Only if FreeEnd is False (normal)
                if self.FreeEnd is False:
                    trajfile.write(str(self.full_fragment_reactant.numatoms) + "\n")
                    trajfile.write("Image 0. Energy: {} \n".format(self.energies_dict[0]))
                    for el, corr in zip(self.full_fragment_reactant.elems, self.full_fragment_reactant.coords):
                        trajfile.write(el + "  " + str(corr[0]) + " " + str(corr[1]) + " " + str(corr[2]) + "\n")

                #Writing all active images in this NEB iteration
                for imageid in list_to_compute:
                    #print("fc:", fc)
                    trajfile.write(str(self.full_fragment_reactant.numatoms) + "\n")
                    trajfile.write("Image {}. Energy: {} \n".format(imageid, E[imageid][0]))
                    #for el, cord in zip(self.full_fragment_reactant.elems, fc):
                    for el, cord in zip(self.full_fragment_reactant.elems, self.full_coords_images_dict[imageid]):
                        trajfile.write(el + "  " + str(cord[0]) + " " + str(cord[1]) + " " + str(cord[2]) + "\n")

                #Writing product image. Only if FreeEnd is False (normal)
                if self.FreeEnd is False:
                    trajfile.write(str(self.full_fragment_product.numatoms) + "\n")
                    trajfile.write("Image {} Energy: {} \n".format(self.numimages-1,self.energies_dict[self.numimages-1]))
                    for el, corp in zip(self.full_fragment_product.elems, self.full_fragment_product.coords):
                        trajfile.write(el + "  " + str(corp[0]) + " " + str(corp[1]) + " " + str(corp[2]) + "\n")


def prepare_saddlepoint(path,neb_settings,reactant,calculator,ActiveRegion,actatoms,charge,mult, numatoms, label, write_tangent=True):
    #Getting saddlepoint-structure and energy if CI-NEB
    if neb_settings["CLIMBING"] is True:

        if write_tangent is True:
            #Writing tangent to disk 
            write_xyzfile(reactant.elems, calculator.tangent, "CItangent", printlevel=2, writemode='w')

        if ActiveRegion == True:
            print("Getting saddlepoint geometry and creating new fragment for Full system")
            print("Has not been confirmed to work...")
            #Finding CI coords and energy
            CI = np.argmax(path.GetEnergy())
            print("Saddlepoint assumed to be image no.", CI)
            saddle_coords_1d=path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()]
            saddle_coords=np.reshape(saddle_coords_1d, (numatoms, 3))
            saddle_energy = path.GetEnergy()[CI][0]*constants.hartoeV

            #Combinining frozen region with optimized active-region for saddle-point
            # Defining full_coords as original coords temporarily
            full_saddleimage_coords = copy.deepcopy(reactant.coords)
            # Replacing act-region coordinates with coords from currcoords
            for i, c in enumerate(saddle_coords):
                if i in actatoms:
                    # Silly. Pop-ing first coord from currcoords until done
                    curr_c, saddle_coords = saddle_coords[0], saddle_coords[1:]
                    full_saddleimage_coords[i] = curr_c

            #Creating new ASH fragment for Full Saddle-point geometry
            Saddlepoint_fragment = ash.Fragment(coords=full_saddleimage_coords, elems=reactant.elems, connectivity=reactant.connectivity, charge=charge, mult=mult)
            Saddlepoint_fragment.set_energy(saddle_energy)
            #Adding atomtypes and charges if present.
            Saddlepoint_fragment.update_atomcharges(reactant.atomcharges)
            Saddlepoint_fragment.update_atomtypes(reactant.atomtypes)

            #Writing out Saddlepoint fragment file and XYZ file
            #Saddlepoint_fragment.print_system(filename=f'Saddlepoint-{label}.ygg')
            Saddlepoint_fragment.write_xyzfile(xyzfilename=f'Saddlepoint-{label}.xyz')

        else:
            #Finding CI coords and energy
            CI = np.argmax(path.GetEnergy())
            print("Saddlepoint assumed to be image no.", CI)
            saddle_coords_1d=path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()]
            saddle_coords=np.reshape(saddle_coords_1d, (reactant.numatoms, 3))
            saddle_energy = path.GetEnergy()[CI][0]/constants.hartoeV
            print(f"Creating new ASH fragment for {label} saddlepoint geometry")
            #Creating new ASH fragment
            Saddlepoint_fragment = ash.Fragment(coords=saddle_coords, elems=reactant.elems, connectivity=reactant.connectivity, charge=charge, mult=mult)
            Saddlepoint_fragment.set_energy(saddle_energy)
            #Writing out Saddlepoint fragment file and XYZ file
            #Saddlepoint_fragment.print_system(filename=f'Saddlepoint-{label}.ygg')
            Saddlepoint_fragment.write_xyzfile(xyzfilename=f'Saddlepoint-{label}.xyz')
        print(f"{label} Saddlepoint energy: {saddle_energy} Eh")
        return Saddlepoint_fragment

#Simple RMS function for np array
def RMSfunc(x):
    rms = 0.0
    for v in x.reshape(1, x.size).flatten():
        rms +=v*v
    return np.sqrt((1.0 / float(x.size)) * rms)