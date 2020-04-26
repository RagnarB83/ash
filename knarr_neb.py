import os

#From Knarratom/atom.py

class Atom(object):

    def __init__(self, name="unknwn_obj", ndim=0, ndof=0, coords=None, symbols=None,
                 constraints=None, cell=None, pbc=None, twodee=False):
        self.name = name  # name

        self.ndim = ndim  # Number of dimensions
        self.ndof = ndof  # Number of active DOF
        self.nim = 1  # Number of images
        self.mass = [] #Mass of atoms
        self.coords = coords  # Full configuration
        self.forces = None  # Full atomic forces
        self.energy = None  # Energy
        self.energy0 = 0.0  # Energy of previous iteration

        self.symbols = symbols  # Chemical elementsXL
        self.pbc = pbc  # Periodic system?
        self.cell = cell  # Cell dimenions
        self.constraints = constraints  # Cartesian constraints
        self.moveable = None  # List of moveable atoms

        self.r = None  # Active configuration
        self.f = None  # Forces for active configuration
        self.h = None  # Hessian matrix for active configuration
        self.v = None  # velocity for active configuration
        self.v0 = None  # velocity of previous step for active configuration
        self.a = None  # Acceleration
        self.a0 = None

        self.twodee = twodee
        self.globaldof = 0
        self.forcecalls = 0  # Number of forcecalls executed on this atoms object
        self.setup = False
        self.output = "knarr"


#From Knarratom/utilities.py

def InitializeAtomObject(name="reactant", input_config="reactant.xyz", pbc=False,
                         twodee=False):
    from KNARRatom.atom import Atom
    atoms = Atom(name=name, pbc=pbc, twodee=twodee)
    atoms.ReadAtomsFromFile(input_config)
    return atoms

def InitializePathObject(nim, react):
    from KNARRatom.path import Path
    # Initialize a path from the reactant structure
    path = Path(name="linear_interp", nim=nim, config1=react.GetCoords(), pbc=react.GetPBC())
    path.SetNDimIm(react.GetNDim())
    path.SetNDofIm(react.GetNDof())
    path.SetNDim(react.GetNDim() * nim)
    path.SetSymbols(react.GetSymbols() * nim)
    path.SetConstraints(np.array(list(react.GetConstraints()) * nim))
    path.ndof = path.GetNDim() - int(path.GetConstraints().sum())
    path.SetMoveableAtoms()
    path.SetOutputFile(react.GetOutputFile())
    path.pbc = react.GetPBC()
    path.twodee = react.IsTwoDee()
    if path.pbc:
        path.SetCell(react.GetCell())
    return path
#KNARR/input.py

def ReadTraj(fname):
    if not os.path.isfile(fname):
        raise IOError("File %s not found " % fname)

    extension = fname.split('.')[-1]
    if extension.strip().upper() != 'XYZ':
        raise IOError('Only .xyz trajectories can be read by KNARR')

    first_line = ReadFirstLineOfFile(fname)
    assert int(first_line)
    natoms = int(first_line)
    ndim = natoms * 3

    # begin by reading the contents of the file to a list
    contents = []
    f = open(fname).readlines()
    for i, line in enumerate(f):
        contents.append(line)

    # get number of lines and hence number of images
    number_of_lines = i + 1
    nim = int((number_of_lines) / (natoms + 2))

    rp = []
    ind = 0
    for i in range(nim):
        symb = []
        ind = ind + 2
        for j in range(natoms):
            geom_line = contents[ind]
            geom_line = geom_line.split()
            symb.append(geom_line[0].strip())
            symb.append(geom_line[0].strip())
            symb.append(geom_line[0].strip())
            rp.append(float(geom_line[1]))
            rp.append(float(geom_line[2]))
            rp.append(float(geom_line[3]))
            ind += 1

    rp = np.reshape(rp, (nim * ndim, 1))

    return rp, ndim, nim, symb

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



def DoNEB(path, calculator, neb, optimizer, second_run=False):
    if second_run:
        PrintJob("Secondary Nudged Elastic Band Method")
    else:
        PrintJob('Nudged Elastic Band Method')
    # PrintCallBack('Linear', path)

    start_t = time.time()
    basename = path.GetOutputFile()
    if second_run == True:
        basename = basename + '_zoom'
    # ---------------------------------------------------------------------------------
    # Read optimizer input parameters
    # ---------------------------------------------------------------------------------
    method_string = optimizer["OPTIM_METHOD"].upper()
    if method_string in KNARRsettings.optimizer_types:
        opttype = KNARRsettings.optimizer_types[method_string]
    else:
        raise NotImplementedError("%s is not implemented" % method_string)
    maxiter = optimizer["MAX_ITER"]
    time_step = optimizer["TIME_STEP"]
    max_move = optimizer["MAX_MOVE"]
    fd_step = optimizer["FD_STEP"]
    lbfgs_memory = optimizer["LBFGS_MEMORY"]
    linesearch = optimizer["LINESEARCH"]
    reset_on_scaling = optimizer["RESTART_ON_SCALING"]
    lbfgs_damping = optimizer["LBFGS_DAMP"]

    if linesearch:
        raise NotImplementedError("Linesearch is not ready")

    # ---------------------------------------------------------------------------------
    # Read neb input parameters
    # --------------------------------------------------------------------------------
    doci = neb["CLIMBING"]
    restart_on_ci = neb["RESTART_OPT_ON_CI"]

    tangent_type_string = neb["TANGENT"].upper()
    if tangent_type_string == "ORIGINAL":
        tangent_type = 0
    elif tangent_type_string == "IMPROVED":
        tangent_type = 1
    else:
        raise TypeError("Unknown tangent type")

    spring_type_string = neb["SPRINGTYPE"].upper()
    if spring_type_string == "ORIGINAL":
        spring_type = 0
    elif spring_type_string == "DISTANCE":
        spring_type = 1
    elif spring_type_string == "IDEAL":
        spring_type = 2
    elif spring_type_string == "ELASTIC_BAND":
        spring_type = 3
        tangent_type = 0 #elastic band uses original tangent
    else:
        raise TypeError("Unknown spring type")

    perp_springtype_string = neb["PERP_SPRINGTYPE"].upper()
    if perp_springtype_string == "" or perp_springtype_string == "NO":
        perp_springtype = 0
    elif perp_springtype_string == "COS":
        perp_springtype = 1
    elif perp_springtype_string == "TAN":
        perp_springtype = 2
    elif perp_springtype_string == "COSTAN":
        perp_springtype = 3
    elif perp_springtype_string == "DNEB":
        perp_springtype = 4
    else:
        raise TypeError("Unknown perp. spring type")

    energy_weighted = neb["ENERGY_WEIGHTED"]
    springconst = neb["SPRINGCONST"]
    springconst2 = neb["SPRINGCONST2"]
    min_rmsd = neb["MIN_RMSD"]
    remove_extern_force = neb["REMOVE_EXTERNAL_FORCE"]
    free_end = neb["FREE_END"]

    free_end_type_str = neb["FREE_END_TYPE"].upper()
    if free_end_type_str == "PERP":
        free_end_type = 0
    elif free_end_type_str == "CONTOUR":
        free_end_type = 1
    elif free_end_type_str == "FULL":
        free_end_type = 2
    else:
        raise TypeError("Unknown free end type")

    free_end_energy1 = neb["FREE_END_ENERGY"]
    free_end_energy2 = neb["FREE_END_ENERGY2"]
    free_end_kappa = neb["FREE_END_KAPPA"]

    conv_type_string = neb["CONV_TYPE"].upper()
    if conv_type_string == "ALL":
        conv_type = 0
    elif conv_type_string == "CI_ONLY" or conv_type_string == "CIONLY":
        conv_type = 1
    else:
        raise TypeError("Unknown option for convergence monitoring")

    tol_scale = neb["TOL_SCALE"]
    tol_max_fci = neb["TOL_MAX_FCI"]
    tol_rms_fci = neb["TOL_RMS_FCI"]
    tol_max_f = neb["TOL_MAX_F"]
    tol_rms_f = neb["TOL_RMS_F"]
    tol_turn_on_ci = neb["TOL_TURN_ON_CI"]

    if tol_scale is not None and doci:

        if tol_scale <= 1:
            raise ValueError("Tol scale is too low")

        tol_max_f = tol_scale * tol_max_fci
        tol_rms_f = tol_scale * tol_rms_fci

    reparamfreq = neb["REPARAM"]
    tol_reparam = neb["TOL_REPARAM"]
    do_reparam = False
    lbfgs_reparam_on_restart = neb["LBFGS_REPARAM_ON_RESTART"]

    interp_type_str = neb["INTERPOLATION_TYPE"].upper()
    if interp_type_str == "LINEAR":
        interp_type = 0
    elif interp_type_str == "CUBIC":
        interp_type = 1
    else:
        raise TypeError("Unknown reparametrization type")

    autozoom = neb["AUTO_ZOOM"]
    zoom_offset = neb["ZOOM_OFFSET"]
    zoom_alpha = neb["ZOOM_ALPHA"]

    # ---------------------------------------------------------------------------------
    # Make checks for stupid input
    # --------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # Initialize NEB
    # --------------------------------------------------------------------------------
    maxf_all = 100.0
    step = 0.0
    ci = -1
    reset_opt = False
    reparam_only_once = True
    startci = False
    was_scaled = False
    converged = False
    stop_neb = False
    fire_param = GetFIREParam(time_step)
    path.PrintPath('Initial path:')

    # --------------------------------------------------------------------------------
    # Initialize second run of neb
    # --------------------------------------------------------------------------------
    if second_run:
        # Get Zoom Region
        CI = np.argmax(path.GetEnergy())
        Ereactant = path.GetEnergy()[0]
        Eproduct = path.GetEnergy()[-1]
        if autozoom:
            print('Searching for automatic zoom interval:')
            s = ComputeLengthOfPath(path.GetNDofIm(), path.GetNim(), path.GetR(), path.GetPBC(), path.GetCell())
            fneb, freal_perp, freal_paral = \
                ComputeEffectiveNEBForce(path.GetF(), 0, path.GetNDofIm(), path.GetNim(), ci, path.GetR(),
                                         path.GetEnergy(),
                                         tangent_type, spring_type, energy_weighted, springconst, springconst2,
                                         perp_springtype,
                                         free_end, free_end_type, free_end_energy1, free_end_energy2, free_end_kappa,
                                         startci, remove_extern_force,
                                         path.GetPBC(), path.GetCell())

            OK, A, B = AutoZoom(path.GetNim(), s, path.GetEnergy(), freal_paral, CI, zoom_alpha)

            if OK == 1:
                print('==> AutoZoom was succesful')
            else:
                print('AutoZoom unable to find a good guess:')
                print('Reverting back to manual selection with offset %i' % zoom_offset)
                A = CI - zoom_offset
                B = CI + zoom_offset

        else:
            A = CI - zoom_offset
            B = CI + zoom_offset

        if A < 0:
            A = 0

        if B > path.GetNim() + 1:
            B = path.GetNim() + 1

        if A == 0 and B == path.GetNim() + 1:
            raise RuntimeError("Too large zoom interval chosen, "
                               "nothing will be gained from the zoom. Please inspect the calculation")

        print('Zoom region is defined by images: %i -to- %i'(A, B))
        coords = path.GetCoords()[A * path.GetNDimIm():(B + 1) * path.GetNDimIm()]
        energy = path.GetEnergy()[A:B + 1]
        znim = len(energy)
        ztang = GetTangent(path.GetNDimIm(), znim, coords, energy, tangent_type,
                           path.GetPBC(), path.GetCell())
        s = ComputeLengthOfPath(path.GetNDimIm(), znim, coords,
                                path.GetPBC(), path.GetCell())

        # if no free-end was used in first run - we need to reduce number of images by 2
        if not free_end:
            nim = path.GetNim() - 2

        free_end = True
        free_end_type = 0

        path.nim = nim
        path.SetNDim(path.GetNim() * path.GetNDimIm())
        path.SetNDof(path.GetNim() * path.GetNDofIm())
        path.energy = None
        path.energy0 = None
        constraints = path.GetConstraints()
        path.SetConstraints(np.array(list(constraints[0:path.GetNDimIm()]) * path.GetNim()))
        path.SetMoveableAtoms()
        new_coords = GenerateNewPath(interp_type, path.GetNDimIm(), znim, path.GetNim(), s, coords, ztang)
        path.SetCoords(new_coords)

        free_end = True
        free_end_type = 0

    # ---------------------------------------------------------------------------------
    # Get end points
    # --------------------------------------------------------------------------------
    if not free_end:
        calculator.Compute(path, list_to_compute=[0, path.GetNim() - 1])
        print('Energy of end points: ')
        print('  Reactant: % 6.6f %s' % (path.GetEnergy()[0], KNARRsettings.energystring))
        print('  Product : % 6.6f %s' % (path.GetEnergy()[path.GetNim() - 1], KNARRsettings.energystring))

        path.UpdateF()
        maxf_reactant = np.max(abs(path.GetF()[0:path.GetNDofIm()]))
        maxf_product = np.max(abs(path.GetF()[(path.GetNim() - 1) * path.GetNDofIm()::]))
        if maxf_reactant > tol_max_f:
            print('**Warning: atom forces on reactant state may be too large. Max(|F|): %6.4f %s' %
                  (maxf_reactant, KNARRsettings.forcestring))
        if maxf_product > tol_max_f:
            print('**Warning: atom forces on product  state may be too large. Max(|F|): %6.4f %s' %
                  (maxf_product, KNARRsettings.forcestring))

        Ereactant = path.GetEnergy()[0]
        Eproduct = path.GetEnergy()[-1]

    if second_run:
        print('Reference energy: %6.6f' % Ereactant)

    # ---------------------------------------------------------------------------------
    # Start NEB optimization
    # --------------------------------------------------------------------------------
    print('\nStarting nudged elastic band optimization:')
    print(' %4ls %4s  %9ls %5ls %7ls %9ls %8ls' %
          ('it', 'dS', 'Energy', 'HEI', 'RMSF', 'MaxF', 'step'))

    for it in range(maxiter):

        # =======================================================
        # Reparametrization and minimization of rmsd
        # =======================================================
        if it > 0:
            if maxf_all < tol_reparam and reparam_only_once:
                do_reparam = True
                reparam_only_once = False

        if (reparamfreq > 0 or do_reparam):
            if ((it % reparamfreq == 0 or do_reparam) and (it > 0)):
                s = ComputeLengthOfPath(path.GetNDim(), path.GetNim(), path.GetCoords())
                tang = GetTangent(path.GetNDim(), path.GetNim(),
                                  path.GetCoords(), path.GetEnergy(), tangent_type)

                if (startci):
                    newr = MakeReparametrizationWithCI(path.GetNDim(), path.GetNim(), ci, s,
                                                       path.GetCoords(), tang, interp_type)
                else:
                    newr = MakeReparametrization(path.GetNDim(), path.GetNim(), s,
                                                 path.GetCoords(), tang, interp_type)
                path.SetCoords(newr)
                WritePath(basename + "_reparam_" + str(it) + "_.xyz", path.GetNDimIm(), path.GetNim(),
                          path.GetCoords(), path.GetSymbols())
                print('\nReparametrization of path\n')
                reset_opt = True
                do_reparam = False

        if min_rmsd and not path.IsConstrained() and not path.IsTwoDee():
            path.MinRMSD()

        # =======================================================
        # Compute energy and gradient
        # =======================================================
        if free_end:
            calculator.Compute(path)
            if it == 0 and not second_run:
                Ereactant = path.GetEnergy()[0]
                Eproduct = path.GetEnergy()[-1]
        else:
            calculator.Compute(path, list_to_compute=range(1, path.GetNim() - 1))
            path.SetEnergy(Ereactant, x=0)
            path.SetEnergy(Eproduct, x=path.GetNim() - 1)

        path.UpdateR()
        path.UpdateF()
        # =======================================================
        # Write trajectory files
        # =======================================================

        WriteTraj(basename + "_optimization.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())
        WritePath(basename + "_current.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())

        # =======================================================
        # Make checks - angle, possible intermediate images, etc.
        # =======================================================

        # =======================================================
        # Compute effective NEB force
        # =======================================================
        s = ComputeLengthOfPath(path.GetNDofIm(), path.GetNim(), path.GetR(),
                                path.GetPBC(), path.GetCell())

        fneb, freal_perp, freal_paral = \
            ComputeEffectiveNEBForce(path.GetF(), it, path.GetNDofIm(), path.GetNim(), ci, path.GetR(),
                                     path.GetEnergy(),
                                     tangent_type, spring_type, energy_weighted, springconst, springconst2,
                                     perp_springtype,
                                     free_end, free_end_type, free_end_energy1, free_end_energy2, free_end_kappa,
                                     startci, remove_extern_force,
                                     path.GetPBC(), path.GetCell(), path.IsTwoDee())

        # =======================================================
        # Compute maximum abs. and RMS force
        # =======================================================
        if startci:
            maxf_all = np.max(abs(freal_perp))
            freal_noci = np.concatenate((freal_perp[0:ci * path.GetNDofIm()],
                                         freal_perp[(ci + 1) * path.GetNDofIm()::]), axis=0)
            rmsf_noci = RMS(path.GetNDof() - path.GetNDofIm(), freal_noci)
            maxf_noci = np.max(abs(freal_noci))
            rmsf_ci = RMS(path.GetNDofIm(), path.GetF()[ci * path.GetNDofIm():(ci + 1) * path.GetNDofIm()])
            maxf_ci = np.max(abs(path.GetF()[ci * path.GetNDofIm():(ci + 1) * path.GetNDofIm()]))
        else:
            rmsf = RMS(path.GetNDof(), freal_perp)
            maxf = np.max(abs(freal_perp))

        # =======================================================
        # Output print
        # =======================================================
        if startci:
            print ("%4i %6.2lf  % 6.6lf %3li %8.4lf  %8.4lf %8.4lf %8.4lf %8.4lf"
                   % (it, s[-1], path.GetEnergy()[ci] - Ereactant, ci, rmsf_noci,
                      maxf_noci, rmsf_ci, maxf_ci, np.max(abs(step))))

        else:
            hei = np.argmax(path.GetEnergy())
            print ("%4i %6.2lf  % 6.6lf %3li  %8.4lf  %8.4lf %8.4lf"
                   % (it, s[-1], path.GetEnergy()[hei] - Ereactant, hei, rmsf, maxf, np.max(abs(step))))

        PiecewiseCubicEnergyInterpolation(basename + ".interp", path.GetNim(), s, path.GetEnergy(), freal_paral, it)

        # =======================================================
        # Check if calculation is converged...
        # =======================================================
        if doci:
            if conv_type == 1:
                if startci:
                    if tol_max_fci > maxf_ci and tol_rms_fci > rmsf_ci:
                        converged = True
                else:
                    if tol_max_fci > maxf and tol_rms_f > rmsf:
                        converged = True
            else:
                if startci:
                    if (tol_max_fci > maxf_ci and tol_rms_fci > rmsf_ci) and \
                            (tol_max_f > maxf_noci and tol_rms_f > rmsf_noci):
                        converged = True
                else:
                    if (tol_max_fci > maxf and tol_rms_fci > rmsf):
                        converged = True
        else:
            if (tol_max_f > maxf and tol_rms_f > rmsf):
                converged = True

        if it == maxiter - 1:
            stop_neb = True
            converged = False

        if converged or stop_neb:
            break

        # =======================================================
        # Climbing image block
        # =======================================================
        if (maxf < tol_turn_on_ci or tol_turn_on_ci == 0.0):
            if not startci and doci:
                checkmax = np.argmax(path.GetEnergy())
                if checkmax != 0 and checkmax != path.GetNim() - 1:
                    startci = True
                    ci = checkmax
                    if restart_on_ci:
                        reset_opt = True

                    print ('        Starting climbing image as image %i.\n' % ci),
                    print('%4ls  %4s  %9ls %5ls %6ls %9ls %9ls %9ls %6ls' %
                          ('it', 'dS', 'Energy', 'HEI', 'RMSF', 'MaxF', 'RMSF_CI', 'MaxF_CI', 'step'))
                    # print CI header here.
                    # print '%6s %4s %2s %10s %8s %13s %5s %11s  %11s %11s %13s' \
                    # % ("Optim.", "iter", "FC", "Objf", "dS", "maxE", "CI", "rmsF'(i)", "rmsF(CI)", "maxF'",
                    #   "maxF(CI)")
        if startci:
            if (it % 5) == 0:
                newci = np.argmax(path.GetEnergy())
                # cant allow it to be at the end points
                if newci != ci and newci != 0 and newci < path.GetNim() - 1:
                    ci = newci
                    if restart_on_ci:
                        reset_opt = True

        # =======================================================
        # Optimization block
        # =======================================================
        if opttype == 0:
            # (Global) Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                path.ZeroV()
            if was_scaled:
                time_step = time_step * 0.95

            step, velo = GlobalVPO(fneb, path.GetV(), time_step)
            path.SetV(velo)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)

        elif opttype == 1:
            # (Local) Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                path.ZeroV()
            if was_scaled:
                time_step = time_step * 0.95

            step, velo = LocalVPO(path.GetNDofIm(), path.GetNim(), fneb, path.GetV(), time_step)
            path.SetV(velo)
            step, was_scaled = LocalScaleStepByMax(path.GetNDofIm(), path.GetNim(), step, max_move)

        elif opttype == 2:
            # Andris Velocity projection optimization
            if it == 0 or reset_opt:
                reset_opt = False
                path.ZeroV()
            if was_scaled:
                time_step = time_step * 0.95

            step, velo = AndriVPO(path.GetNDof(), fneb, path.GetV(), time_step)
            path.SetV(velo)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)

        elif opttype == 4:
            # FIRE
            if it == 0 or reset_opt:
                reset_opt = False
                fire_param = GetFIREParam(time_step)
                path.ZeroV()
            if was_scaled:
                time_step = time_step * 0.95

            velo, time_step, fire_param = GlobalFIRE(fneb, path.GetV(), time_step, fire_param)
            path.SetV(velo)
            step, velo = EulerStep(path.GetV(), fneb, time_step)
            path.SetV(velo)
            step, was_scaled = GlobalScaleStepByMax(step, max_move)

        elif opttype == 5:
            # L-BFGS
            if it == 0 or reset_opt:
                reset_opt = False
                was_scaled = False
                sk = []
                yk = []
                rhok = []
                keepf = fneb.copy()
                keepr = path.GetR().copy()
                y = lambda x: ComputeEffectiveNEBForce(x, it, path.GetNDofIm(), path.GetNim(), ci, path.GetR(),
                                                       path.GetEnergy(),
                                                       tangent_type, spring_type, energy_weighted, springconst,
                                                       springconst2,
                                                       perp_springtype, free_end, free_end_type, free_end_energy1,
                                                       free_end_energy2, free_end_kappa,
                                                       startci, remove_extern_force,
                                                       path.GetPBC(), path.GetCell())
                step = TakeFDStepWithFunction(calculator, path, fd_step, fneb, y)
                step = GlobalScaleStepByMax(step, max_move)[0]
            else:
                sk, yk, rhok = LBFGSUpdate(path.GetR(), keepr, fneb, keepf,
                                           sk, yk, rhok, lbfgs_memory)
                keepf = fneb.copy()
                keepr = path.GetR().copy()
                step, negativecurv = LBFGSStep(fneb, sk, yk, rhok)
                step *= lbfgs_damping

                if negativecurv:
                    reset_opt = True

                step, was_scaled = GlobalScaleStepByMax(step, max_move)

        else:
            raise RuntimeError(
                "Choosen optimization method  %s is not available in structural opt." % method_string)

        if reset_on_scaling and was_scaled:
            reset_opt = True
            if lbfgs_reparam_on_restart:
                do_reparam = True

        path.SetR(path.GetR() + step)
        path.UpdateCoords()
        path.MIC()

    # ---------------------------------------------------------------------------------
    # END NEB OPTIMIZATION - Write output
    # ---------------------------------------------------------------------------------
    CI = np.argmax(path.GetEnergy())
    if converged:
        PrintConverged(it, path.GetFC())
        PrintDivider()
        print('Summary:')
        PrintDivider()
        print('%4ls %4ls %5ls %9ls %9ls' % ('Img.', 'dS', 'E', 'dE', 'MaxF'))
        for i in range(path.GetNim()):
            print('% 2i % 6.2f % 6.4f % 6.4f % 6.6f' % (
                i, s[i], path.GetEnergy()[i], path.GetEnergy()[i] - Ereactant,
                np.max(abs(freal_perp[i * path.GetNDofIm():(i + 1) * path.GetNDofIm()]))))

        WritePath(basename + "_MEP.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())

        if os.path.isfile(basename + "_current.xyz"):
            os.remove(basename + "_current.xyz")

        PrintAtomMatrix("\nSaddle point configuration:", path.GetNDimIm(),
                        path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()],
                        path.GetSymbols())

        PrintAtomMatrix("Atomic forces:", path.GetNDofIm(),
                        path.GetForces()[CI * path.GetNDofIm():(CI + 1) * path.GetNDofIm()],
                        path.GetSymbols())

        tang = GetTangent(path.GetNDofIm(), path.GetNim(), path.GetR(), path.GetEnergy(),
                          tangent_type, pbc=path.GetPBC(), cell=path.GetCell())

        PrintAtomMatrix("Tangent to path:", path.GetNDofIm(),
                        tang[CI * path.GetNDofIm():(CI + 1) * path.GetNDofIm()],
                        path.GetSymbols())

        WriteEnergyFile(basename + '.energy', path.GetEnergy(), path.GetNim())

        imaginary_mode = np.zeros(shape=(path.GetNDimIm(), 1))
        imaginary_mode[path.GetMoveableAtoms()[0:path.GetNDofIm()]] = \
            tang[CI * path.GetNDofIm():(CI + 1) * path.GetNDofIm()]
        GenerateVibrTrajectory(basename + '_mode.xyz', path.GetNDimIm(),
                               path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()],
                               path.GetSymbols(), imaginary_mode, npts=40, A=0.3)

        if KNARRsettings.extension == 0:
            WriteXYZ(basename + "_saddle.xyz", path.GetNDimIm(),
                     path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()], path.GetSymbols(),
                     energy=path.GetEnergy()[CI])
        else:
            WriteCon(basename + "_saddle.con", 1,
                     path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()],
                     path.GetSymbols()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()],
                     path.GetCell(),
                     path.GetConstraints()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()])

    if stop_neb:
        PrintMaxIter(maxiter)
        WritePath(basename + "_last_iter.xyz", path.GetNDimIm(), path.GetNim(), path.GetCoords(),
                  path.GetSymbols(), path.GetEnergy())

        if os.path.isfile(basename + "_current.xyz"):
            os.remove(basename + "_current.xyz")

    # Done.
    PrintJobDone('NEB job', time.time() - start_t)

    return path.GetEnergy()[CI], path.GetCoords()[CI * path.GetNDimIm():(CI + 1) * path.GetNDimIm()]







#From KNARR/KNARRjobs/neb.py
optimizer = {"OPTIM_METHOD": "LBFGS", "MAX_ITER": 1000, "TOL_MAX_FORCE": 0.01,
              "TOL_RMS_FORCE": 0.005, "TIME_STEP": 0.01, "MAX_MOVE": 0.1, "RESTART_ON_SCALING": True,
              "LBFGS_MEMORY": 20,
              "LBFGS_DAMP": 1.0,
              "FD_STEP": 0.001,
              "LINESEARCH": None}

settings_types = {"MAIN": 0,
                  "CALCULATOR": 1,
                  "OPTIMIZER": 2,
                  "PATH": 3,
                  "DYNAMICS": 4,
                  "SADDLE": 5,
                  "NEB": 6,
                  "FREQ": 7}

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
              "TOL_MAX_F": 0.026, "TOL_RMS_F": 0.013, "TOL_TURN_ON_CI": 1.0,
              "ZOOM": False,
              "TOL_TURN_ON_ZOOM": 0.5,
              "REPARAM": 0, "TOL_REPARAM": 0.0,
              "INTERPOLATION_TYPE": "LINEAR",
              "AUTO_ZOOM": True, "ZOOM_OFFSET": 1, "ZOOM_ALPHA": 0.5,
              "RESTART_OPT_ON_CI": True,
              "LBFGS_REPARAM_ON_RESTART": False
              }

#sett = KNARRsettings.settings_types["NEB"]
#neb = FetchSettings(sett, settings_list, settings_line_ind, full_input)
#optimizer dict with stuff
#neb dict with stuff. same as parameters_dict above
#from KNARRio.io import ReadTraj
rp, ndim, nim, symb = ReadTraj(neb["PATH"])
path = InitializePathObject(nim, react)
path.SetCoords(rp)

path_parameters = {"METHOD": "DOUBLE", "INTERPOLATION": "LINEAR", "NIMAGES": 6,
              "INSERT_CONFIG": None, "IDPP_MAX_ITER": 2000,
              "IDPP_SPRINGCONST": 10.0, "IDPP_TIME_STEP": 0.01,
              "IDPP_MAX_MOVE": 0.1, "IDPP_MAX_F": 0.01, "IDPP_RMS_F": 0.005}

#RB function

def Knarr_pathgenerator(nebsettings,path_parameters,nim,react):
    #sett = KNARRsettings.settings_types["PATH"]
    sett = nebsettings
    #pathgen = FetchSettings(sett, settings_list, settings_line_ind, full_input)
    #path_parameters
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
        insertion = path_parameters["INSERT_CONFIG"]
        if insertion is not None:
            insertion = InitializeAtomObject(name="insertion", input_config=path_parameters["INSERT_CONFIG"],
                                             pbc=main_control["PBC"])
            if insertion.GetSymbols() != react.GetSymbols():
                raise ValueError("Insertion does not match reactant / product")
            path.SetInsertionConfig(insertion.GetCoords())
    else:
        prod = None

    DoPathInterpolation(path, path_parameters)

#TEST

nim=8
#react?
Knarr_pathgenerator(neb_settings,path_parameters,nim,react)


DoNEB(path, calc, neb, optimizer)