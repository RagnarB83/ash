from utilities import FindElementInList


# Author: Vilhjalmur Asgeirsson, 2019.


def ScanInput(fname):
    listi = []
    settings = []
    settings_line_ind = []
    with open(fname) as f:
        f = f.readlines()
        for i, line in enumerate(f):
            if line in ['\n', '\r\n']:  # skip empty lines
                listi.append('')
                continue

            l0 = line[0].strip().upper()
            chrctr = l0[0][0]  # get first character - to check if line is a comment.
            if '%' in chrctr or '$' in chrctr:
                listi.append('')
                continue

            if '&' in line:
                k = line.split()
                settings.append(k[1].strip())
                settings_line_ind.append(int(i))

            listi.append(line.strip())

    return settings, settings_line_ind, listi


def convert_bool(val):
    if val.upper() == 'FALSE':
        val = False
    elif val.upper() == 'TRUE':
        val = True
    return val


def convert(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass


def FetchSettings(sett, settings_list, settings_line_ind, full_input):
    value = FindElementInList(sett, settings_list)
    if value is not None:
        control = UpdateSettings(sett, settings_line_ind[value], full_input)
    else:
        control = GetSettings(sett)

    return control


def GetSettings(settings):
    if settings == 0:
        # MAIN PARAMETERS
        parameters = {"OUTPUT_NAME": "knarr", "CONFIG_FILE": "reactant.xyz", "CONFIG_FILE2": "product.xyz",
                      "PRINT_LEVEL": 1, "PBC": False, "TWO_DEE_SYSTEM": False, "GLOBAL_DOF": 0,
                      "SEED" : 1234, "MKL_THREADS": 1, "OMP_THREADS": 1}

    elif settings == 1:
        # Calculator parameters
        parameters = {"CALCULATOR": "XTB", "CALCULATOR_TEMPLATE": None, "CHARGE": 0, "MULTIPLICITY": 1, "FD_STEP": 0.0001, "NCORE": 1, "BOOST": -1000.0, "GAUSS_A" : 1.0, "GAUSS_B" : -0.05, "GAUSS_ALPHA" : 12.0, "LJ_RCUT" : 100.0, "LJ_SIGMA" : 1.0, "LJ_EPSILON" : 1.0}

    elif settings == 2:
        # Optimization parameters
        parameters = {"OPTIM_METHOD": "LBFGS", "MAX_ITER": 1000, "TOL_MAX_FORCE": 0.01,
                      "TOL_RMS_FORCE": 0.005, "TIME_STEP": 0.01, "MAX_MOVE": 0.1, "RESTART_ON_SCALING": True,
                      "LBFGS_MEMORY": 20,
                      "LBFGS_DAMP" : 1.0,
                      "FD_STEP": 0.001,
                      "LINESEARCH": None}

    elif settings == 3:
        # path generation
        parameters = {"METHOD": "DOUBLE", "INTERPOLATION": "LINEAR", "NIMAGES": 6,
                      "INSERT_CONFIG": None, "IDPP_MAX_ITER": 2000,
                      "IDPP_SPRINGCONST": 10.0, "IDPP_TIME_STEP": 0.01,
                      "IDPP_MAX_MOVE": 0.1, "IDPP_MAX_F": 0.01, "IDPP_RMS_F": 0.005}

    elif settings == 4:
        # Dynamics job parameters
        parameters = {"THERMOSTAT": None, "ANDERSEN_COLLISION_FREQ": 10,
                      "ANDERSEN_COLLISION_STRENGTH": 1.0,
                      "LANGEVIN_FRICTION" : 0.01,
                      "TIME_STEP": 0.01, "SIMULATION_TIME": 1000,
                      "OUTPUT": "MD.xyz", "VELOCITY_DISTR": "MAXWELL_BOLTZMANN",
                      "TEMPERATURE": 298.15, "FORCE_TEMPERATURE": False,
                      "CONFORMATIONAL_SAMPLING": False, "SAMPLE_INTERVAL": 100,
                      "EXIT_IF_JUMP" : False, "EXIT_CRITICAL_TIME" : 100,
                      "NO_KB" : False, "PRINT_ITER": 1}


    elif settings == 5:
        # Saddle job parameters
        parameters = {"SADDLE_METHOD": "MMF", "MMF_TYPE": "LANCZOS", "L_GUESS": None,
                      "L_NMAX": 30, "L_JUMP": 5, "L_FDSTEP": 1e-5, "L_TOL": 0.001,
                      "COMPUTE_HESSIAN_EVERY" : 1, "DAMPING" : 1.0}

    elif settings == 6:
        parameters = {"PATH": "neb.xyz",
                      "CLIMBING": True,
                      "TANGENT": "IMPROVED",
                      "SPRINGTYPE": "DISTANCE",
                      "PERP_SPRINGTYPE": "",
                      "ENERGY_WEIGHTED": True, "SPRINGCONST": 1.0, "SPRINGCONST2": 10.0,
                      "MIN_RMSD": True, "REMOVE_EXTERNAL_FORCE": True,
                      "FREE_END": False, "FREE_END_TYPE": 'PERP', "FREE_END_ENERGY": 0.0,
                      "FREE_END_ENERGY2": 0.0, "FREE_END_KAPPA": 0.0,
                      "CONV_TYPE": "ALL", "TOL_SCALE": 10, "TOL_MAX_FCI": 0.026, "TOL_RMS_FCI": 0.013,
                      "TOL_MAX_F": 0.026, "TOL_RMS_F": 0.013, "TOL_TURN_ON_CI": 1.0,
                      "ZOOM" : False,
                      "TOL_TURN_ON_ZOOM": 0.5,
                      "REPARAM": 0, "TOL_REPARAM": 0.0,
                      "INTERPOLATION_TYPE": "LINEAR",
                      "AUTO_ZOOM": True, "ZOOM_OFFSET": 1, "ZOOM_ALPHA": 0.5,
                      "RESTART_OPT_ON_CI": True,
                      "LBFGS_REPARAM_ON_RESTART": False
                      }

    elif settings == 7:
        parameters = {"PARTIAL_HESSIAN" : True}

    else:
        raise RuntimeError("Hmm... can not find the job! Are you sure you know what you are asking for?")

    return parameters


def UpdateSettings(job, start_read, listi):
    parameters = GetSettings(job)

    for i in range(start_read + 1, len(listi)):

        if '&' in listi[i]:  # We read until next DO or until end_of_file
            break
        else:
            a = listi[i].split('=')  # read every instance from the list
            if len(a) > 1:
                keyw = a[0].upper().strip()  # the dict keyword
                val = a[1].strip()  # the dict value
                if keyw in parameters:
                    # convert to correct format
                    val = convert(val)
                    if isinstance(val, basestring):
                        val = convert_bool(val)
                    parameters[keyw] = val  # change value of kw
                else:
                    raise IOError("Keyword %s is not recognized" % keyw)
            else:
                if listi[i] != '':
                    raise IOError("Invalid format of input structure (%s)" % listi[i])

    return parameters
