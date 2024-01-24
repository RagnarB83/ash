import os
import time
import KNARRsettings
import numpy as np
import random

from KNARRio.system_print import PrintHeader, PrintDivider, PrintCredit
from KNARRcalculator.calculator import Calculator
from KNARRjobs.point import DoPoint
from KNARRjobs.freq import DoFreq
from KNARRjobs.opt import DoOpt
from KNARRjobs.saddle import DoSaddle
from KNARRjobs.RMSD import DoRMSD
from KNARRjobs.path import DoPathInterpolation
from KNARRjobs.dynamics import DoDynamics
from KNARRjobs.neb import DoNEB

from KNARRio.input import ScanInput, FetchSettings
from KNARRio.utilities import AreThereDuplicateValues
from KNARRatom.utilities import InitializeAtomObject, InitializePathObject


# Author: Vilhjalmur Asgeirsson (2019--)
# University of Iceland

# TODO: CONCATENATE OPTIMIZE ROUTINES INTO ONE ROUTINE (saddle search and minimize etc)
# TODO: Make one routine calling all worker routines

def MAIN(current_job):
    current_job = current_job.upper().strip()
    input_file = "knarr.inp"

    # =============================================================
    # Initialize KNARR
    # =============================================================
    start_timer = time.time()
    PrintHeader()
    PrintCredit()
    PrintDivider()
    PrintDivider()

    # =============================================================
    # Check jobs and read input file for settings
    # =============================================================

    # Check if job exists
    do_job = 0
    job_found = False
    if current_job in KNARRsettings.job_types:
        job_found = True
        do_job = KNARRsettings.job_types[current_job]  # the job that will be done

    if not job_found:
        raise IOError("Unknown job-type (%s)" % current_job)

    # Search for input file
    if not os.path.isfile(input_file):
        raise IOError("No input file found. Please include knarr.inp in your working directory.")

    # Read input file
    settings, settings_line_ind, full_input = ScanInput(input_file)
    settings_list = []
    for sett in settings:
        sett = sett.upper().strip()
        sett_found = False
        if sett.upper() in KNARRsettings.settings_types:
            settings_list.append(KNARRsettings.settings_types[sett])
            sett_found = True

        if not sett_found:
            raise TypeError("Unknown settings-type (%s) in parameter file" % sett)

    # Sort settings_list and settings_line_ind together.

    settings_list, settings_line_ind = zip(*sorted(zip(settings_list, settings_line_ind)))

    if AreThereDuplicateValues(settings_list):
        raise IOError("Non-unique settings found in parameter file")

    # =============================================================
    # Set environmental variables
    # =============================================================
    print 'Initializing KNARR:'
    # READ MAIN PARAMETERS (settings = 0)
    sett = KNARRsettings.settings_types["MAIN"]
    main_control = FetchSettings(sett, settings_list, settings_line_ind, full_input)

    KNARRsettings.printlevel = main_control["PRINT_LEVEL"]
    KNARRsettings.seed = main_control["SEED"]
    random.seed(KNARRsettings.seed)
    print('Setting random seed as: %f' % KNARRsettings.seed)
    
    try:
        int(main_control["OMP_THREADS"])
    except:
        raise TypeError("Bad choice for OMP_THREADS")

    os.environ["OMP_NUM_THREADS"] = str(main_control["OMP_THREADS"])

    try:
        int(main_control["MKL_THREADS"])
    except:
        raise TypeError("Invalid type for MKL_THREADS")

    os.environ["OMP_MKL_THREADS"] = str(main_control["MKL_THREADS"])

    # =============================================================
    # Construct an atom object and fill it
    # =============================================================
    print('Reading reactant:')
    input_config = main_control["CONFIG_FILE"]
    if not os.path.isfile(input_config):
        raise IOError("Unable to find configuration file %s" % input_config)
    check_extension = input_config.split('.')[-1].upper()

    if check_extension == 'XYZ':
        KNARRsettings.extension = 0
    elif check_extension == 'CON':
        KNARRsettings.extension = 1
    else:
        raise NotImplementedError("File format (%s) is not recognized" % check_extension)

    react = InitializeAtomObject(name="Reactant", input_config=input_config,
                                 pbc=main_control["PBC"], twodee=main_control["TWO_DEE_SYSTEM"])
    react.SetOutputFile(main_control["OUTPUT_NAME"])
    try:
        print int(main_control["GLOBAL_DOF"])
        react.SetGlobalDof(int(main_control["GLOBAL_DOF"]))
    except:
        raise RuntimeError("Bad global DOF")

    if react.twodee == True:
        react.SetGlobalDof(0)

    print('...  reactant set')

    # =============================================================
    # Construct a calculator (settings = 1)
    # =============================================================
    print('Calculator:')
    sett = KNARRsettings.settings_types["CALCULATOR"]
    calculator_control = FetchSettings(sett, settings_list, settings_line_ind, full_input)

    name = calculator_control["CALCULATOR"]
    charge = calculator_control["CHARGE"]
    multiplicity = calculator_control["MULTIPLICITY"]
    template_file = calculator_control["CALCULATOR_TEMPLATE"]
    ncore = calculator_control["NCORE"]
    fd_step = float(calculator_control["FD_STEP"])

    # define globals for potentials
    KNARRsettings.boost = float(calculator_control["BOOST"])
    KNARRsettings.gauss_A = float(calculator_control["GAUSS_A"])
    KNARRsettings.gauss_B = float(calculator_control["GAUSS_B"])
    KNARRsettings.gauss_alpha = float(calculator_control["GAUSS_ALPHA"])
    KNARRsettings.ljrcut = float(calculator_control["LJ_RCUT"])
    KNARRsettings.ljsigma = float(calculator_control["LJ_SIGMA"])
    KNARRsettings.ljepsilon = float(calculator_control["LJ_EPSILON"])

    
    calc = Calculator(name=name, ncore=ncore, template_file=template_file,
                      fd_step=fd_step, charge=charge, multiplicity=multiplicity)
    calc.Setup()

    if react.GetPBC() and not calc.GetPBC():
        print("**Warning: Calculator does not support PBC. Are you sure this is OK?")

    print('... calculator initialized')
    # ==========================================================================
    # ==========================================================================
    # S T A R T I N G - J O B S
    # ==========================================================================
    # ==========================================================================

    # ================================
    # single point energy and gradient
    # ================================

    if do_job == 0:
        DoPoint(react, calc)
    # ================================
    # Frequency job
    # ================================

    elif do_job == 1:
        sett = KNARRsettings.settings_types["FREQ"]
        freq = FetchSettings(sett, settings_list, settings_line_ind, full_input)
        DoFreq(react, calc, freq)
    # ================================
    # Structural optimization
    # ================================

    elif do_job == 2:
        sett = KNARRsettings.settings_types["OPTIMIZER"]
        optim = FetchSettings(sett, settings_list, settings_line_ind, full_input)
        DoOpt(react, calc, optim)

    # ================================
    # Minimization of RMSD
    # ================================
    elif do_job == 3:
        product = InitializeAtomObject(name="product", input_config=main_control["CONFIG_FILE2"],
                                       pbc=main_control["PBC"])
        DoRMSD(react, product)
    # ================================
    # Interpolation / path generation
    # ================================
    elif do_job == 4:
        sett = KNARRsettings.settings_types["PATH"]
        pathgen = FetchSettings(sett, settings_list, settings_line_ind, full_input)

        type_of_method_string = pathgen["METHOD"].upper()
        if type_of_method_string == "SINGLE":
            prod_is_needed = False
        elif type_of_method_string == "DOUBLE":
            prod_is_needed = True
        else:
            raise TypeError("Either choose single or double ended path generation")

        nim = pathgen["NIMAGES"]
        path = InitializePathObject(nim, react)
        if prod_is_needed:
            # Get product
            prod = InitializeAtomObject(name="product", input_config=main_control["CONFIG_FILE2"],
                                        pbc=main_control["PBC"])

            # Check product
            if react.GetNDim() != prod.GetNDim():
                raise RuntimeError("Reactant / product do not match")
            if react.GetSymbols() != prod.GetSymbols():
                raise RuntimeError("Reactant / product do not match")
            path.SetConfig2(prod.GetCoords())

            # check insertion
            insertion = pathgen["INSERT_CONFIG"]
            if insertion is not None:
                insertion = InitializeAtomObject(name="insertion", input_config=pathgen["INSERT_CONFIG"],
                                                 pbc=main_control["PBC"])
                if insertion.GetSymbols() != react.GetSymbols():
                    raise ValueError("Insertion does not match reactant / product")
                path.SetInsertionConfig(insertion.GetCoords())
        else:
            prod = None

        DoPathInterpolation(path, pathgen)

    # ================================
    # Molecular dynamics
    # ================================
    elif do_job == 5:
        sett = KNARRsettings.settings_types["DYNAMICS"]
        dynamics = FetchSettings(sett, settings_list, settings_line_ind, full_input)
        DoDynamics(react, calc, dynamics)

    # =================================
    # Saddle point search
    # ================================
    elif do_job == 6:
        sett = KNARRsettings.settings_types["SADDLE"]
        saddle = FetchSettings(sett, settings_list, settings_line_ind, full_input)

        sett = KNARRsettings.settings_types["OPTIMIZER"]
        optimizer = FetchSettings(sett, settings_list, settings_line_ind, full_input)

        DoSaddle(react, calc, optimizer, saddle)
    # =================================
    # NEB
    # ================================
    elif do_job == 7:
        from KNARRio.io import ReadTraj
        sett = KNARRsettings.settings_types["OPTIMIZER"]
        optimizer = FetchSettings(sett, settings_list, settings_line_ind, full_input)
        sett = KNARRsettings.settings_types["NEB"]
        neb = FetchSettings(sett, settings_list, settings_line_ind, full_input)

        assert os.path.isfile(neb["PATH"]), ("Initial path %s not found for NEB" % neb["PATH"])

        rp, ndim, nim, symb = ReadTraj(neb["PATH"])
        path = InitializePathObject(nim, react)
        path.SetCoords(rp)

        # Make some checks before we enter
        if react.GetNDim() != ndim:
            raise IOError("Dimension mismatch in %s and %s" % (neb["PATH"], main_control["CONFIG_FILE"]))

        if react.GetSymbols() != symb:
            raise IOError("Chemical elements of path do not match with reactant")

        if np.sum(react.GetCoords() - rp[0:ndim]) > 0.01:
            print('**Warning: the reactant and first image of the path do not match. Are you sure about this?')

        if not neb["ZOOM"]:
            DoNEB(path, calc, neb, optimizer)

        else:
            from KNARRjobs.utilities import AutoZoom

            if not neb["CLIMBING"]:
                raise RuntimeError("Can not use zoom without CI")
            if neb["TOL_TURN_ON_CI"] <= neb["TOL_TURN_ON_ZOOM"]:
                raise RuntimeError("CI needs to be activated before zoom")

            # overwrite default parameters
            keep_tol_maxfci = neb["TOL_MAX_FCI"]
            keep_tol_rmsfci = neb["TOL_RMS_FCI"]
            keep_convtype = neb["CONV_TYPE"]

            neb["CONV_TYPE"] = "CIONLY"
            neb["TOL_MAX_FCI"] = neb["TOL_TURN_ON_ZOOM"]
            neb["TOL_RMS_FCI"] = neb["TOL_TURN_ON_ZOOM"] / 2.0

            DoNEB(path, calc, neb, optimizer, second_run=False)

            # restore default parameters
            neb["TOL_MAX_FCI"] = keep_tol_maxfci
            neb["TOL_RMS_FCI"] = keep_tol_rmsfci
            neb["CONV_TYPE"] = keep_convtype
            DoNEB(path, calc, neb, optimizer, second_run=True)



    else:
        raise NotImplementedError("Job requested has not been implemented")

    # =============================================================
    # Terminate execution of KNARR
    # =============================================================
    PrintDivider()
    stop_timer = time.time() - start_timer
    print('KNARR successfully terminated')
    print('Total run-time: %6.4lfs' % round(stop_timer, 4))

    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        raise RuntimeError("Expecting only one input argument")

    # current_job = 'opt'
    current_job = sys.argv[1]
    # current_job = "dynamics"
    OK = MAIN(current_job)
